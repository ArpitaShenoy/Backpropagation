import numpy as np
class ValueNode():
    def __init__(self, data, grad=None, op=None, _prev=(), backward_fn=None):
        self.data = data
        self.grad = np.zeros_like(self.data)
        self._op = op
        self._prev = list(_prev) if _prev is not None else []
        self._backward = backward_fn

    def __repr__(self):
        return f"{self.data}"

    def backward(self):
        topo = []
        visited = set()

        def build_topo(node):
            """
            Builds the topological sort recursively.
            """
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)
            
        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            if node._backward:
                node._backward()
        