import numpy as np
from .Node import ValueNode

class MSELoss():
    """
    Calculates Mean Squared Error of the prediction w.r.t the ground truth.
    """
    
    def __init__(self):
        self.loss = None
        self.preds = None
        self.targs = None
    
    def forward(self, preds: ValueNode, targs: np.ndarray):
        """Calculates the loss.
        
        Args:
            preds (np.ndarray) : prediction probabilities from the model
            targs (np.ndarray) : ground truth labels
        """
        # check if the prediction shape matches the targs shape
        if isinstance(preds, ValueNode):
            self.preds = preds
        elif isinstance(preds, np.ndarray):
            self.preds = ValueNode(data=preds)
        else:
            raise TypeError(f"preds must be of type np.ndarray")
        
        if not isinstance(targs, np.ndarray):
            targs = np.array(targs)
        
        
        # check if the sizes match, if not check if the transposed targs matches the size with preds
        # if both fail, then raise ValueError
        if self.preds.data.size == targs.size:
            # calculate the mean squared error
            self.targs = targs
            self.loss = ValueNode(data=np.sum(((self.targs-self.preds.data)**2))/self.preds.data.size,
                                  op="mse",
                                  _prev=[self.preds],
                                  backward_fn=self._backward)
        elif self.preds.data.size == targs.T.size:
            self.targs = targs.T
            self.loss = ValueNode(data=np.sum(((self.targs-self.preds.data)**2))/self.preds.data.size,
                                  op="mse",
                                  _prev=[self.preds],
                                  backward_fn=self._backward)
        else:
            raise ValueError(f"Please check the sizes, {self.preds.data.size}!={self.targs.size}")

        

        return self.loss


    def __call__(self, x: np.ndarray, y: np.ndarray):
        """calls forward method when object(x) is called."""
        return self.forward(x,y)
    
    def _backward(self):
        """Calculates the derivative of this layer with resepect to inputs.
        
        Returns (np.ndarray): derivative of the loss
        """

        self.preds.grad += self.loss.grad*(2/self.preds.data.size)*(self.preds.data-self.targs)


    
    def backward(self):
        """This is called when loss.backward() is called, which calls our backward,
         and creates a final node with gradient set to 1. which is the derivative of loss w.r.t loss.
        """
        self.loss.backward()

        
