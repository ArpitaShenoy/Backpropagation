import numpy as np
from .Node import ValueNode

class Linear():
    """
    This is the linear class that performs affine transformation (XW+b).
    
    Args:
        M (int) : in features(number of features in the input data)
        N (int) : The number of distinct characteristics the layer is trying to "detect" or "extract" from the data. 
                  Each unit in N represents a unique "filter" or "perspective" on the input.
        weights (np.array) : predetermined weights if any, else the class will create one basis M and N
        bias (np.array) : same as weights, but bias has 
    """

    def __init__(self, M: int, N: int, weights=None, bias=None):
        self.M = M # in features
        self.N = N # out features
        self._weights = None
        self._bias = None
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

        # check the dimension of the bias whether it's 1D or 2D array. if not 2D, then convert it after 
        # verifying it's length
        if bias.ndim == 1:
            if len(bias) == self.N:
                bias = bias[np.newaxis, :] # Correctly promotes (2,) to (1, 2)
            else:
                raise ValueError(f"bias length {len(bias)} must match N {self.N}")
        elif bias.ndim == 2:
            if bias.shape[1] != self.N:
                raise ValueError(f"bias.shape[1]:{bias.shape[1]} must match N:{self.N}")
        else:
            raise ValueError(f"bias can only have 2D not more than that!")
        

        
        # check for the shape compatibility of the provided weight matrix
        # the in features(M) must match the n_rows else check by transposing it
        # (incase user provided weights in different order hoping it would be transposed later)
        # else raise error
        if weights.shape[0] == self.M:
            self._weights = ValueNode(data=weights)
        elif weights.shape[1] == self.M:
            self._weights = ValueNode(data=weights.T)
        else:
            raise ValueError(f"Shape mismatch: weights must be of shape({self.M},{self.N}) and bias should be either float or np.ndarray(1,)")
        
        self._bias = ValueNode(data=bias)

    # @property is set to make the parameters method an attribute of the class rather than a method.
    # everytime we access parameters we don't want to access weights and bias separately, instead we
    # want it bundled up together as a tuple. but we want it to be separate nodes in the rest of the 
    # code. 
    @property 
    def parameters(self) -> tuple:
        """Bundles up weights and bias together as tuples and returns the same. This method allows users
        to only call parameters as an attribute rather than accessing weights and biases separately.
        Also safeguards the weights and biases from modification.

        This is what is passed to the optimizer.
        """
        return (self._weights, self._bias)
    
    def forward(self, x:np.ndarray) -> ValueNode:
        """Computes affine transformation.
        
        Args:
            x (np.ndarray) : inputs to the layer.
        """
        # check if x is np.ndarray
        if isinstance(x, np.ndarray):
            self.x = ValueNode(data=x)
        elif isinstance(x, ValueNode):
            self.x = x
        else:
            raise TypeError(f"X must be a numpy array. Other datatype isn't acceptable")

        return (self.x @ self._weights) + self._bias
    
    def __call__(self, x:np.ndarray):
        """When object(x) is performed, this is the method that gets called.
        
        Args:
            x (np.ndarray) : inputs to the layer
        """

        return self.forward(x=x)
        
    

class Sigmoid():
    """
    Performs Sigmoid operation and also holds the backward method, which calculates the 
    derivative of the node that holds this operation.
    """

    def __init__(self):
        self.z = None
        self.x = None

    def forward(self, x: ValueNode) -> ValueNode:

        """Calculates the Sigmoid.
        
        Args:
            x (ValueNode) : this is the input to the sigmoid coming from previous layer.
        """
        if isinstance(x, np.ndarray):
            self.x = ValueNode(data=x)
        elif isinstance(x, ValueNode):
            self.x = x
        else:
            raise TypeError(f"input must be a numpy array.")

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
    """
    Rectified Linear Unit, which simply passes the data where the input is greater than 0.
    """

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
        elif isinstance(x, np.ndarray):
            self.x = ValueNode(data=x)
        else:
            raise TypeError(f"input must be of type numpy array.")

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
    """
    This is similar to PyTorch's Sequential Layer which weaves the layers together and makes it easier
    to call just one forward method which then loops over all, instead of us, calling each layer one by 
    one.
    """
    def __init__(self, *args):
        self.layers = args

    def forward(self,x) -> ValueNode:
        """
        Calls all the layers in a loop and returns the final Node.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x):
        return self.forward(x=x)
    
    @property 
    def parameters(self) -> tuple:
        """
        Bundles up and returns the list of parameters of all the linear layer defined in the
        sequential class.
        """

        params = []
        for layer in self.layers:

            if isinstance(layer, Linear):
                params.append(layer.parameters)
        
        return params

        
                
