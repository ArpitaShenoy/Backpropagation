import numpy as np
from .Node import ValueNode

class Linear():

    def __init__(self, M: int, N: int, weights=None, bias=None):
        self.M = M # in features
        self.N = N # out features
        self.weights = None
        self.bias = None
        self.update_parameters(weights, bias)
        self.x=None
        self.dw, self.db = None, None

    def update_parameters(self, weights, bias):
        """updates weights and biases required for a layer.
        If the weights aren't provided, this creates, a random weight matrix of MxN,
        which is derived from the linear layer shape.

        Args:
            weights: user provided weights
            bias   : user provided bias
        """

        # check for the type and convert if it isn't np.ndarray
        # and if weights and biases are None then create it
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights) if weights is not None else np.random.randn(self.M, self.N)

        if not isinstance(bias, np.ndarray):
            bias = np.array(bias) if bias is not None else np.random.randn(1, self.N)

        if bias.ndim == 1:
            if len(bias) == self.N:
                bias = bias[np.newaxis, :] # Correctly promotes (2,) to (1, 2)
            else:
                raise ValueError(f"bias length {len(bias)} must match N {self.N}")
        



        
        # check for the shape compatibility of the provided weight matrix
        # the in features(M) must match the n_rows else check by transposing it
        # (incase user provided weights in different order hoping it would be transposed later)
        # else raise error
        if weights.shape[0] == self.M:
            self.weights = ValueNode(data=weights)
            self.bias = ValueNode(data=bias)
        elif weights.shape[1] == self.M:
            self.weights = ValueNode(data=weights.T)
            self.bias = ValueNode(data=bias)
        else:
            raise ValueError(f"Shape mismatch: weights must be of shape({self.M},{self.N}) and bias should be either float or np.ndarray(1,)")


    @property 
    def parameters(self) -> tuple:
        return (self.weights, self.bias)
    
    def forward(self, x:np.ndarray) -> ValueNode:
        """Computes linear function.
        
        Args:
            x (np.ndarray) : inputs to the layer.
        """
        if isinstance(x, ValueNode):
            self.x = x
        else:
            self.x = ValueNode(data=x)

        self.wx = ValueNode(data=np.matmul(self.x.data,self.weights.data), 
                    op="matmul",
                    _prev=[self.weights,self.x], 
                    backward_fn=self._backward_wx)


        self.wx_b = ValueNode(data=self.wx.data+self.bias.data, op="+", _prev=[self.wx, self.bias],
                         backward_fn=self._backward_wx_b)
        return self.wx_b
    
    def __call__(self, x:np.ndarray):
        """When object(x) is performed, this is the method that gets called.
        
        Args:
            x (np.ndarray) : inputs to the layer
        """

        return self.forward(x=x)
    
    def _backward_wx(self):
        """Calculates the derivative w.r.t parameters and w.r.t inputs.
        
        Args:
            inp: derivative from the next layer.

        Returns: derivative of this layer w.r.t it's inputs
        """

        self.weights.grad += np.matmul(self.x.data.T, self.wx.grad)
        self.x.grad += np.matmul(self.wx.grad, self.weights.data.T)
        return self.x.grad
    
    def _backward_wx_b(self):
        self.wx.grad += self.wx_b.grad
        self.bias.grad += np.sum(self.wx_b.grad, axis=0, keepdims=True) 
    
    

class Sigmoid():

    def __init__(self):
        self.z = None
        self.x = None

    def forward(self, x: ValueNode) -> ValueNode:

        """Calculates the Sigmoid.
        
        Args:
            x (np.ndarray) : this is the input to the sigmoid coming from previous layer.
        """
        if isinstance(x, ValueNode):
            self.x = x
        else:
            self.x = ValueNode(data=x)

        self.z = ValueNode(data=1/(1+np.exp(-self.x.data)), op="sigmoid", _prev=[self.x],
                           backward_fn=self._backward)
        return self.z

    def __call__(self, x: ValueNode):
        """calls forward method when object(x) is called.
        
        Args:
            x (np.ndarray) : inputs to the layer
        """
        return self.forward(x=x)
    
    def _backward(self):
        """Calculates the derivative of the output of this layer w.r.t it's input.
        derivative of sigmoid(x) w.r.t x is sigmoid(x)*(1-sigmoid(x))
        
        Args:
            inp (np.ndarray) : derivative of the next layer.

        Returns: the derivative of this function
        """
        self.x.grad+= self.z.grad*(self.z.data*(1-self.z.data))



class ReLU():

    def __init__(self):
        self.z = None
        self.x = None

    def forward(self,x: ValueNode) -> ValueNode:
        """Calculates the ReLU(max(0,x)).
        
        Args:
            x (ValueNode) : The value node passed by the previous layer

        Returns (ValueNode) : ValueNode that holds the calculation of the max(0,x)
        """
        if isinstance(x, ValueNode):
            self.x = x
        else:
            self.x = ValueNode(data=x)

        self.z = ValueNode(data=self.x.data*(self.x.data>0),
                           op="ReLU",
                           _prev = [self.x],
                           backward_fn=self._backward)
        return self.z
    
    def __call__(self, x):
        """Calls the forward function"""
        return self.forward(x=x)
    
    def _backward(self):
        """Calculates the local derivative and pushes it to its child gradient.
        
        Local derivative = 1 where x.data>0
        """

        self.x.grad += self.z.grad*(self.x.data>0)


class Sequential:
    def __init__(self, *args):
        self.layers = args

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x):
        return self.forward(x=x)
    
    @property 
    def parameters(self) -> tuple:
        params = []
        for layer in self.layers:

            if isinstance(layer, Linear):
                params.append(layer.parameters)
        
        return params

        
                
