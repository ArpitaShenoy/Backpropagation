import numpy as np

class MSELoss():
    
    def __init__(self):
        self.loss = None
        self.preds = None
        self.targs = None
    
    def forward(self, preds: np.ndarray, targs: np.ndarray):
        """Calculates the loss.
        
        Args:
            preds (np.ndarray) : prediction probabilities from the model
            targs (np.ndarray) : ground truth labels
        """
        # check if the prediction shape matches the targs shape
        if not isinstance(targs, np.ndarray):
            targs = np.array(targs)
        if preds.shape == targs.shape:
            # calculate the mean squared error
            self.targs = targs
            self.loss = np.sum(((targs-preds)**2))/len(targs)
        elif preds.shape == targs.T.shape:
            self.targs = targs.T
            self.loss = np.sum(((targs-preds.T)**2))/len(targs.T)

        # save the predictions to calculate the derivatives later
        self.preds = preds
        

        return self.loss


    def __call__(self, x: np.ndarray, y: np.ndarray):
        """calls forward method when object(x) is called."""
        return self.forward(x,y)
    
    def backward(self) -> np.ndarray:
        """Calculates the derivative of this layer with resepect to inputs.
        
        Returns (np.ndarray): derivative of the loss
        """
        dout_dinp = 2/len(self.targs)*(self.preds-self.targs)

        return dout_dinp
        
