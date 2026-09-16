import numpy as np
from collections import deque, defaultdict
from .engine import Context, Add, MatMul, ReLU, SoftmaxCrossEntropy

class Edge:
    """directed dataflow dependency: 'node' produced the value that fills
    input slot 'inp_index' of whichever Node owns this Edge"""
    __slots__ = ('node', 'inp_index')

    def __init__(self, node, inp_index):
        self.node = node
        self.inp_index = inp_index


class Node:
    """explicit graph vertex. One per op application, plus one per leaf
    (a GraphTesnor with no producing op -- an input or a parameter)"""

    def __init__(self, op_name, fn_cls, ctx, inp_edges):
        self.tensor = None      # back-reference, set by Graph.apply / leaf creation
        self.op_name = op_name
        self.fn_cls = fn_cls    # None for leaf nodes
        self.ctx = ctx          # None for leaf nodes
        self.inp_edges = inp_edges
        self.grad = None
        self.num_consumers = 0  # how many nodes read this node's output

    def __repr__(self):
        return f"Node(op={self.op_name}, consumers={self.num_consumers})"


class GraphTensor:
    """Value wrapper. Unlike EagerTensor, this does not carry _prev/_backward
    itself -- the graph structure lives entirely in Node/Edge, owned by
    a Graph instance. GraphTensor just points at the Node that produced it"""

    def __init__(self, data, node=None):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.node = node # None -> leaf; a node gets created for
                         # it lazily, the first time it is used

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
        self.node = None

    def __repr__(self):
        return f"GraphTensor(shape={self.data.shape})"


class Graph:
    """owns every Node created during one forward pass.
    This is implicit in Tensor (spread across _prev sets) and explicit here"""

    def __init__(self):
        self.nodes = []

    def _leaf_node_for(self, t: GraphTensor) -> Node:
        if t.node is None:
            leaf = Node(op_name='leaf', fn_cls=None, ctx=None, inp_edges=[])
            leaf.tensor = t
            t.node = leaf
            self.nodes.append(leaf)
        return t.node

    def apply(self, fn_cls, *args):
        ctx = Context()
        raw_inputs = [a.data if isinstance(a, GraphTensor) else a for a in args]
        out_data = fn_cls.forward(ctx, *raw_inputs)

        tensor_args = [a for a in args if isinstance(a, GraphTensor)]
        inp_edges = []
        for i, a in enumerate(tensor_args):
            producer = self._leaf_node_for(a)
            producer.num_consumers += 1
            inp_edges.append(Edge(producer, i))

        node = Node(op_name=fn_cls.__name__, fn_cls=fn_cls, ctx=ctx, inp_edges=inp_edges)
        self.nodes.append(node)

        out = GraphTensor(out_data, node=node)
        node.tensor = out
        return out

    @staticmethod
    def _ctx_bytes(ctx):
        if ctx is None or ctx.saved_tensors is None:
            return 0
        total = 0
        for t in ctx.saved_tensors:
            if isinstance(t, np.ndarray):
                total += t.nbytes
        return total

    def backward(self, root: GraphTensor, track_memory: bool = False):
        root_node = root.node

        visited = {root_node}
        pending = defaultdict(int)
        stack = [root_node]
        while stack:
            v = stack.pop()
            for e in v.inp_edges:
                pending[e.node] += 1
                if e.node not in visited:
                    visited.add(e.node)
                    stack.append(e.node)

        root_node.grad = np.ones_like(root.data)
        ready = deque(v for v in visited if pending[v] == 0)

        memory_log = None
        live_bytes = 0
        if track_memory:
            live_bytes = sum(self._ctx_bytes(v.ctx) for v in visited)
            memory_log = {
                "initial_bytes": live_bytes,
                "peak_bytes": live_bytes,
                "timeline": [],  # (op_name, bytes_freed, bytes_still_held)
            }

        while ready:
            v = ready.popleft()

            if v.op_name == 'leaf':
                v.tensor.grad = v.grad
            else:
                grads = v.fn_cls.backward(v.ctx, v.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)
                for edge, g in zip(v.inp_edges, grads):
                    if g is None:
                        continue
                    producer = edge.node
                    if producer.grad is None:
                        producer.grad = np.zeros_like(g)
                    producer.grad = producer.grad + g

                if track_memory:
                    freed = self._ctx_bytes(v.ctx)
                    v.ctx.saved_tensors = None
                    live_bytes -= freed
                    memory_log["timeline"].append((v.op_name, freed, live_bytes))

            for e in v.inp_edges:
                pending[e.node] -= 1
                if pending[e.node] == 0:
                    ready.append(e.node)

        return memory_log

# -- thin convenience wrappers, mirroring TensorV2's dunder methods --
def add(graph, a, b):
    b = b if isinstance(b, GraphTensor) else GraphTensor(b)
    return graph.apply(Add, a, b)

def matmul(graph, a, b):
    return graph.apply(MatMul, a, b)

def relu(graph, a):
    return graph.apply(ReLU, a)

def softmax_cross_entropy(graph, a, target):
    return graph.apply(SoftmaxCrossEntropy, a, target)