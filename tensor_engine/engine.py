import numpy as np
from collections import deque, defaultdict


def _unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Context:
    def __init__(self):
        self.saved_tensors = None

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class Function:
    """Base class for a differentiable op."""

    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args):
        ctx = Context()
        raw_inputs = [a.data if isinstance(a, EagerTensor) else a for a in args]
        out_data = cls.forward(ctx, *raw_inputs)

        tensor_args = [a for a in args if isinstance(a, EagerTensor)]
        out = EagerTensor(out_data, _children=set(tensor_args), _op=cls.__name__)

        def _backward():
            grads = cls.backward(ctx, out.grad)
            if not isinstance(grads, tuple):
                grads = (grads,)
            for arg, g in zip(tensor_args, grads):
                if g is not None:
                    arg.grad += g

        out._backward = _backward
        return out


class Add(Function):
    @staticmethod
    def forward(ctx, *args):
        if len(args) != 2:
            raise ValueError(f"Add expects 2 arguments, got {len(args)}")
        a, b = args
        ctx.save_for_backward(a, b)
        return a + b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = _unbroadcast(grad_output, a.shape)
        grad_b = _unbroadcast(grad_output, b.shape)
        return grad_a, grad_b


class Mul(Function):
    @staticmethod
    def forward(ctx, *args):
        if len(args) != 2:
            raise ValueError(f"Mul expects 2 arguments, got {len(args)}")
        a, b = args
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = _unbroadcast(grad_output * b, a.shape)
        grad_b = _unbroadcast(grad_output * a, b.shape)
        return grad_a, grad_b


class MatMul(Function):
    @staticmethod
    def forward(ctx, *args):
        if len(args) != 2:
            raise ValueError(f"MatMul expects 2 arguments, got {len(args)}")
        X, W = args
        ctx.save_for_backward(X, W)
        return X @ W

    @staticmethod
    def backward(ctx, grad_output):
        X, W = ctx.saved_tensors
        grad_X = grad_output @ W.T
        grad_W = X.T @ grad_output
        return grad_X, grad_W


class Sum(Function):
    @staticmethod
    def forward(ctx, *args):
        if len(args) != 1:
            raise ValueError(f"Sum expects 1 argument, got {len(args)}")
        x = args[0]
        ctx.save_for_backward(x)
        return np.array(x.sum())

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * np.ones_like(x)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, *args):
        if len(args) != 1:
            raise ValueError(f"Sigmoid expects 1 argument, got {len(args)}")
        x = args[0]
        s = 1.0 / (1.0 + np.exp(-x))
        ctx.save_for_backward(s)
        return s

    @staticmethod
    def backward(ctx, grad_output):
        s, = ctx.saved_tensors
        return grad_output * s * (1 - s)


class ReLU(Function):
    @staticmethod
    def forward(ctx, *args):
        if len(args) != 1:
            raise ValueError(f"ReLU expects 1 argument, got {len(args)}")
        x = args[0]
        mask = x > 0
        ctx.save_for_backward(mask)
        return np.where(mask, x, 0.0)

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output * mask


class Softmax(Function):
    @staticmethod
    def forward(ctx, *args):
        if len(args) != 1:
            raise ValueError(f"Softmax expects 1 argument, got {len(args)}")
        z = args[0]

        shifted = z - z.max(axis=1, keepdims=True) # stability
        exp = np.exp(shifted)
        s = exp / exp.sum(axis=1, keepdims=True)

        ctx.save_for_backward(s)
        return s

    @staticmethod
    def backward(ctx, grad_output):
        s, = ctx.saved_tensors

        # Jacobian: dS/dz = S * (δ_ij - S_j)
        # Vectorized: grad = S * g - sum(g * S, axis=1)
        dot = np.sum(grad_output * s, axis=1, keepdims=True)
        grad_input = s * (grad_output - dot)

        return grad_input


class CrossEntropy(Function):
    """Cross-entropy on already-computed probabilities -- not fused with softmax.

    Accepts either:
    - Integer class indices (hard labels): target shape (batch,)
    - probability distributions (soft labels): target shape (batch, C)
    """
    EPSILON = 1e-12

    @staticmethod
    def forward(ctx, *args):
        if len(args) != 2:
            raise ValueError(f"CrossEntropy expects 2 arguments, got {len(args)}")
        probs, target = args
        batch = probs.shape[0]

        if target.ndim == 1: # Integer class indices
            target_is_soft = False
            # gather probabilities at target indices
            correct_probs = probs[np.arange(batch), target]
            loss = -np.log(correct_probs + CrossEntropy.EPSILON).mean()
        else: # soft labels: target is (batch, C), sums to 1 per row
            target_is_soft = True
            loss = -(target * np.log(probs + CrossEntropy.EPSILON)).sum(axis=1).mean()

        ctx.save_for_backward(probs, target)
        ctx.target_is_soft = target_is_soft
        return np.array(loss)

    @staticmethod
    def backward(ctx, grad_output):
        probs, target = ctx.saved_tensors
        batch = probs.shape[0]

        if ctx.target_is_soft:
            y = target
        else:
            y = np.zeros_like(probs)
            y[np.arange(batch), target] = 1.0

        # dL/dp_i = -y_i / p_i, averaged over batch, chained
        grad_probs = -(y / (probs + CrossEntropy.EPSILON)) / batch
        grad_probs *= grad_output

        return grad_probs, None # no gradient to target


class SoftmaxCrossEntropy(Function):
    """Fused softmax + cross-entropy for numerical stability."""
    @staticmethod
    def forward(ctx, *args):
        """
        arg[1] logits: (batch, C) raw scores,
        arg[2] target: (batch,) integer class indices
        """
        if len(args) != 2:
            raise ValueError(f"SoftmaxCrossEntropy expects 2 arguments, got {len(args)}")
        logits, target = args

        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        batch = logits.shape[0]
        correct_probs = probs[np.arange(batch), target]
        loss = -np.log(correct_probs + 1e-12).mean()

        ctx.save_for_backward(probs, target)
        return np.array(loss)

    @staticmethod
    def backward(ctx, grad_output):
        probs, target = ctx.saved_tensors
        batch = probs.shape[0]

        # gradient of softmax + cross entropy = probs - one_hot(target)
        grad_logits = probs.copy()
        grad_logits[np.arange(batch), target] -= 1.0

        grad_logits /= batch
        grad_logits *= grad_output

        return grad_logits, None


class EagerTensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)

        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op

    def __repr__(self):
        return f"EagerTensor(shape={self.data.shape}, op='{self._op}')"

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    # -- arithmetic --
    def __add__(self, other):
        other = other if isinstance(other, EagerTensor) else EagerTensor(other)
        return Add.apply(self, other)

    def __mul__(self, other):
        other = other if isinstance(other, EagerTensor) else EagerTensor(other)
        return Mul.apply(self, other)

    def matmul(self, other):
        return MatMul.apply(self, other)
    __matmul__ = matmul

    def sum(self):
        return Sum.apply(self)

    # -- activation --
    def relu(self):
        return ReLU.apply(self)

    def softmax(self):
        return Softmax.apply(self)

    def sigmoid(self):
        return Sigmoid.apply(self)

    # -- loss --
    """target: raw numpy array (int class indices or soft-label probs),
        NOT a Tensor -- Function.apply only tracks Tensor args in the graph"""
    def cross_entropy(self, target):
        return CrossEntropy.apply(self, target)

    """Fused, numerically-stable path -- self is expected to be raw logits,
        NOT post-softmax probabilities"""
    def softmax_cross_entropy(self, target):
        return SoftmaxCrossEntropy.apply(self, target)

    def backward(self):
        visited = {self}
        pending = defaultdict(int)
        stack = [self]
        while stack:
            node = stack.pop()
            for inp in node._prev:
                pending[inp] += 1
                if inp not in visited:
                    visited.add(inp)
                    stack.append(inp)

        self.grad = np.ones_like(self.data)
        ready = deque(node for node in visited if pending[node] == 0)

        while ready:
            node = ready.popleft()
            node._backward()
            for inp in node._prev:
                pending[inp] -= 1
                if pending[inp] == 0:
                    ready.append(inp)