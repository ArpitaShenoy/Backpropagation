import numpy as np

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
            if not weights:
                weights = np.random.randn(self.M,self.N)
            else:
                weights = np.array(weights)
            
        if not isinstance(bias, np.ndarray):
            if not bias:
                bias = np.random.randn(self.N)
            elif len(bias) == self.N:
                bias = np.array([bias])
            else:
                raise ValueError(f"bias must be of length {self.N}")
        else: 
            # check the shape if np.ndarray was provided
            if len(bias) != self.N:
                raise ValueError(f"bias must be of length {self.N}")



        
        # check for the shape compatibility of the provided weight matrix
        # the in features(M) must match the n_rows else check by transposing it
        # (incase user provided weights in different order hoping it would be transposed later)
        # else raise error
        if weights.shape[0] == self.M:
            self.weights = weights
            self.bias = bias
        elif weights.shape[1] == self.M:
            self.weights = weights.T
            self.bias = bias
        else:
            raise ValueError(f"Shape mismatch: weights must be of shape({self.M},{self.N}) and bias should be either float or np.ndarray(1,)")


    @property 
    def parameters(self) -> tuple:
        return (self.weights, self.bias)
    
    def forward(self, x:np.ndarray) -> np.ndarray:
        """Computes linear function.
        
        Args:
            x (np.ndarray) : inputs to the layer.
        """
        self.x = x
        return np.matmul(self.x,self.weights)+self.bias
    
    def __call__(self, x:np.ndarray):
        """When object(x) is performed, this is the method that gets called.
        
        Args:
            x (np.ndarray) : inputs to the layer
        """

        return self.forward(x=x)
    
    def backward(self, inp: np.ndarray):
        """Calculates the derivative w.r.t parameters and w.r.t inputs.
        
        Args:
            inp: derivative from the next layer.

        Returns: derivative of this layer w.r.t it's inputs
        """

        # reshape the inp which is currently (2,) to (1,2)
        # and input to (2,1)
        self.x = self.x.reshape(-1,1)
        inp = inp.reshape(1,-1)
        self.dw = np.matmul(self.x, inp)
        self.db = np.sum(inp, axis=0)

        return np.matmul(inp, self.weights.T)
    

class Sigmoid():

    def __init__(self):
        self.z = None
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:

        """Calculates the Sigmoid.
        
        Args:
            x (np.ndarray) : this is the input to the sigmoid coming from previous layer.
        """
        self.x = x
        self.z = 1/(1+np.exp(-self.x))
        return self.z

    def __call__(self, x: np.ndarray):
        """calls forward method when object(x) is called.
        
        Args:
            x (np.ndarray) : inputs to the layer
        """
        return self.forward(x=x)
    
    def backward(self, inp:np.ndarray) -> np.ndarray:
        """Calculates the derivative of the output of this layer w.r.t it's input.
        derivative of sigmoid(x) w.r.t x is sigmoid(x)*(1-sigmoid(x))
        
        Args:
            inp (np.ndarray) : derivative of the next layer.

        Returns: the derivative of this function
        """
        inp

        return inp*(self.z*(1-self.z))



    
                
