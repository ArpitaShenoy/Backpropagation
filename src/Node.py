import numpy as np
class ValueNode():
    def __init__(self, data, grad=None, op=None, _prev=(), backward_fn=None):
        self.data = data
        self.grad = np.zeros_like(self.data)
        self._op = op
        self._prev = list(_prev) if _prev is not None else []
        self._backward = backward_fn

    def __add__(self, other):
        """Element-wise addition with broadcasting support."""
        other = other if isinstance(other, ValueNode) else ValueNode(other)
        out = ValueNode(self.data+other.data, _prev=(self,other), op="+")
        
        def _backward():
            # gradient for self (handling broadcasting)
            s_grad = out.grad

            while s_grad.ndim > self.data.ndim:
                s_grad = s_grad.sum(axis=0)
            
            for axis, size in enumerate(self.data.shape):
                if size==1:
                    s_grad = s_grad.sum(axis=axis, keepdims=True)
            self.grad += s_grad

            # gradient for other (handling broadcasting)
            o_grad = out.grad
            while o_grad.ndim > other.data.ndim:
                o_grad = o_grad.sum(axis=0)
            
            for axis, size in enumerate(other.data.shape):
                if size==1:
                    o_grad = o_grad.sum(axis=axis, keepdims=True)
            other.grad += o_grad
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        """Matrix Multiplication"""
        other = other if isinstance(other, ValueNode) else ValueNode(other)
        out = ValueNode(self.data @ other.data, _prev=(self, other), op="matmul")

        def _backward():
            # derivative w.r.t first matrix: grad @ other.T
            self.grad += out.grad @ other.data.T
            # derivative w.r.t second matrix: grad.T @ other
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out
    
    def __rmatmul__(self, other):
        other = other if isinstance(other,ValueNode) else ValueNode(other)
        return other @ self
    
    def __radd__(self, other):
        other = other if isinstance(other, ValueNode) else ValueNode(other)
        return self + other

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

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            if node._backward:
                node._backward()
        