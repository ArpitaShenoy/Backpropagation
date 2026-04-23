import numpy as np
from .Node import ValueNode

class MSELoss():
    
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
            self.x = preds
        else:
            self.x = ValueNode(data=preds)
        if not isinstance(targs, np.ndarray):
            targs = np.array(targs)
        if preds.data.size == targs.size:
            # calculate the mean squared error
            self.targs = targs
            self.loss = ValueNode(data=np.sum(((targs-preds.data)**2))/preds.data.size,
                                  op="mse",
                                  _prev=[preds],
                                  backward_fn=self._backward)
        elif preds.data.size == targs.T.size:
            self.targs = targs.T
            self.loss = ValueNode(data=np.sum(((targs-preds.data.T)**2))/preds.data.size,
                                  op="mse",
                                  _prev=[preds],
                                  backward_fn=self._backward)
        else:
            raise ValueError(f"Please check the sizes, {preds.data.size}!={targs.size}")

        # save the predictions to calculate the derivatives later
        self.preds = preds
        

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

        
