"""
Test suite for tensor_engine. Run with: pytest tests/test_engine.py -v
(from the project root, after `pip install -e .`)
"""

import numpy as np
import pytest
from tensor_engine.engine import (
    EagerTensor, Context, Add, MatMul, ReLU, SoftmaxCrossEntropy, _unbroadcast,
)
from tensor_engine.graph_engine import (
    Graph, GraphTensor, add, matmul, relu, softmax_cross_entropy,
)


# ---------------------------------------------------------------------------
# shared gradient-checking helper (central difference, see math_reviews.ipynb
# derivation of why central difference has O(eps^2) error).
# ---------------------------------------------------------------------------
def assert_grad_close(fn_cls, inputs, target=None, eps=1e-6, atol=1e-4):
    extra = (target,) if target is not None else ()
    ctx = Context()
    out = fn_cls.forward(ctx, *inputs, *extra)
    grad_output = np.ones_like(out) if out.ndim > 0 else np.array(1.0)
    analytic = fn_cls.backward(ctx, grad_output)
    if not isinstance(analytic, tuple):
        analytic = (analytic,)

    for idx, (x, g_analytic) in enumerate(zip(inputs, analytic)):
        if g_analytic is None:
            continue
        numeric = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'])
        for _ in it:
            i = it.multi_index
            orig = x[i]
            x[i] = orig + eps
            out_p = fn_cls.forward(Context(), *inputs, *extra).sum()
            x[i] = orig - eps
            out_m = fn_cls.forward(Context(), *inputs, *extra).sum()
            x[i] = orig
            numeric[i] = (out_p - out_m) / (2 * eps)

        max_diff = np.abs(numeric - g_analytic).max()
        assert max_diff < atol, (
            f"{fn_cls.__name__} input[{idx}] gradient mismatch: "
            f"max|numeric - analytic| = {max_diff:.2e} (tol={atol:.0e})"
        )


# ---------------------------------------------------------------------------
# per-op unit tests
# ---------------------------------------------------------------------------
class TestAdd:
    def test_forward(self):
        out = Add.forward(Context(), np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]]))
        np.testing.assert_allclose(out, [[4.0, 6.0]])

    def test_gradient(self):
        np.random.seed(0)
        assert_grad_close(Add, [np.random.randn(3, 4), np.random.randn(3, 4)])

    def test_gradient_with_broadcasting(self):
        np.random.seed(1)
        assert_grad_close(Add, [np.random.randn(3, 4), np.random.randn(4)])


class TestMatMul:
    def test_forward(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = MatMul.forward(Context(), X, np.eye(2))
        np.testing.assert_allclose(out, X)

    def test_gradient(self):
        np.random.seed(2)
        assert_grad_close(MatMul, [np.random.randn(5, 3), np.random.randn(3, 4)])

    def test_gradient_batch_size_one(self):
        np.random.seed(3)
        assert_grad_close(MatMul, [np.random.randn(1, 3), np.random.randn(3, 4)])


class TestReLU:
    def test_forward_clips_negatives(self):
        out = ReLU.forward(Context(), np.array([-2.0, -0.5, 0.0, 0.5, 2.0]))
        np.testing.assert_allclose(out, [0.0, 0.0, 0.0, 0.5, 2.0])

    def test_gradient_away_from_zero(self):
        assert_grad_close(ReLU, [np.array([-2.0, -0.5, 0.3, 1.7, 4.0])])

    def test_all_units_dead(self):
        ctx = Context()
        x = np.array([-1.0, -2.0, -3.0])
        out = ReLU.forward(ctx, x)
        grad = ReLU.backward(ctx, np.ones_like(out))[0]
        np.testing.assert_array_equal(grad, np.zeros_like(x))


class TestSoftmaxCrossEntropy:
    def test_gradient(self):
        np.random.seed(4)
        logits = np.random.randn(6, 5)
        target = np.random.randint(0, 5, size=6)
        assert_grad_close(SoftmaxCrossEntropy, [logits], target=target)

    def test_probs_sum_to_one(self):
        ctx = Context()
        SoftmaxCrossEntropy.forward(ctx, np.array([[5.0, -3.0, 100.0], [0.0, 0.0, 0.0]]), np.array([0, 1]))
        probs, _ = ctx.saved_tensors
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-10)

    def test_extreme_logits_no_overflow(self):
        ctx = Context()
        loss = SoftmaxCrossEntropy.forward(ctx, np.array([[1000.0, 1.0, 0.0]]), np.array([0]))
        assert np.isfinite(loss)


# ---------------------------------------------------------------------------
# Integration tests -- both engine architectures must agree exactly
# ---------------------------------------------------------------------------
KNOWN_GOOD_LOSS = 1.1048456697548716
KNOWN_GOOD_W1_GRAD_NORM = 0.06618849436727453


def test_eager_tensor_capstone():
    np.random.seed(0)
    X, W1, b1 = EagerTensor(np.random.randn(4, 3)), EagerTensor(np.random.randn(3, 5) * 0.1), EagerTensor(np.zeros(5))
    W2, b2 = EagerTensor(np.random.randn(5, 3) * 0.1), EagerTensor(np.zeros(3))
    target = np.array([0, 1, 2, 1])

    loss = ((X @ W1 + b1).relu() @ W2 + b2).softmax_cross_entropy(target)
    loss.backward()

    assert np.isclose(loss.data, KNOWN_GOOD_LOSS)
    assert np.isclose(np.linalg.norm(W1.grad), KNOWN_GOOD_W1_GRAD_NORM)


def test_graph_tensor_capstone_matches_eager():
    np.random.seed(0)
    g = Graph()
    X = GraphTensor(np.random.randn(4, 3))
    W1, b1 = GraphTensor(np.random.randn(3, 5) * 0.1), GraphTensor(np.zeros(5))
    W2, b2 = GraphTensor(np.random.randn(5, 3) * 0.1), GraphTensor(np.zeros(3))
    target = np.array([0, 1, 2, 1])

    h1 = relu(g, add(g, matmul(g, X, W1), b1))
    logits = add(g, matmul(g, h1, W2), b2)
    loss = softmax_cross_entropy(g, logits, target)
    g.backward(loss)

    assert np.isclose(loss.data, KNOWN_GOOD_LOSS)
    assert np.isclose(np.linalg.norm(W1.grad), KNOWN_GOOD_W1_GRAD_NORM)


def test_zero_grad_prevents_doubled_gradient_eager():
    t = EagerTensor(np.array([1.0, 2.0]))
    t.grad = np.array([5.0, 5.0])  # simulate leftover state from a prior step
    t.zero_grad()
    np.testing.assert_array_equal(t.grad, [0.0, 0.0])


def test_zero_grad_resets_node_graph_tensor():
    # regression test for the doubled-gradient bug: without resetting .node,
    # a GraphTensor reused across two separate Graph instances would silently
    # accumulate onto a stale Node's leftover .grad.
    np.random.seed(0)

    def one_step(g, X, W1, b1, W2, b2, target):
        h1 = relu(g, add(g, matmul(g, X, W1), b1))
        logits = add(g, matmul(g, h1, W2), b2)
        return softmax_cross_entropy(g, logits, target)

    X = GraphTensor(np.random.randn(4, 3))
    W1, b1 = GraphTensor(np.random.randn(3, 5) * 0.1), GraphTensor(np.zeros(5))
    W2, b2 = GraphTensor(np.random.randn(5, 3) * 0.1), GraphTensor(np.zeros(3))
    target = np.array([0, 1, 2, 1])

    g1 = Graph()
    loss1 = one_step(g1, X, W1, b1, W2, b2, target)
    g1.backward(loss1)

    for p in (W1, b1, W2, b2):
        p.zero_grad()

    g2 = Graph()
    loss2 = one_step(g2, X, W1, b1, W2, b2, target)
    g2.backward(loss2)

    assert np.isclose(np.linalg.norm(W1.grad), KNOWN_GOOD_W1_GRAD_NORM)


def test_memory_tracking_frees_everything_by_end():
    np.random.seed(0)
    g = Graph()
    X = GraphTensor(np.random.randn(4, 3))
    W1, b1 = GraphTensor(np.random.randn(3, 5) * 0.1), GraphTensor(np.zeros(5))
    W2, b2 = GraphTensor(np.random.randn(5, 3) * 0.1), GraphTensor(np.zeros(3))
    target = np.array([0, 1, 2, 1])

    h1 = relu(g, add(g, matmul(g, X, W1), b1))
    logits = add(g, matmul(g, h1, W2), b2)
    loss = softmax_cross_entropy(g, logits, target)
    report = g.backward(loss, track_memory=True)

    assert report["initial_bytes"] > 0
    assert report["timeline"][-1][2] == 0  # fully freed by the end


@pytest.mark.parametrize(
    "batch,in_dim,hidden_dim,out_dim",
    [(4, 2, 4, 4), (64, 50, 128, 10), (1, 3, 4, 2)],
)
def test_two_layer_forward_backward_various_sizes(batch, in_dim, hidden_dim, out_dim):
    np.random.seed(0)
    X = EagerTensor(np.random.randn(batch, in_dim))
    W1, b1 = EagerTensor(np.random.randn(in_dim, hidden_dim) * 0.1), EagerTensor(np.zeros(hidden_dim))
    W2, b2 = EagerTensor(np.random.randn(hidden_dim, out_dim) * 0.1), EagerTensor(np.zeros(out_dim))
    target = np.random.randint(0, out_dim, size=batch)

    logits = (X @ W1 + b1).relu() @ W2 + b2
    loss = logits.softmax_cross_entropy(target)
    loss.backward()

    for p in (W1, b1, W2, b2):
        assert p.data.shape == p.grad.shape
        assert np.all(np.isfinite(p.grad))
    assert np.isfinite(loss.data)

    before = float(loss.data)
    lr = 0.5
    for p in (W1, b1, W2, b2):
        p.data = p.data - lr * p.grad
    loss2 = ((X @ W1 + b1).relu() @ W2 + b2).softmax_cross_entropy(target)
    assert float(loss2.data) < before