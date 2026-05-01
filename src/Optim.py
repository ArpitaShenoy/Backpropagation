import numpy as np

class SGD:
    """
    Updates the parameters of all the linear layers and resets their gradients to 0.

    This method takes in parameters of linear layers and a learning rate(lr) which defaults to 0.05.
    """
    def __init__(self, params,lr=None):
        self.params = params
        self.lr = lr

    def step(self):
        """Adjusts the parameters by multiplying it's gradient by a factor of learning rate.
        
        If learning rate is not provided, it defaults to 0.05
        """
        self.lr = 0.05 if not self.lr else self.lr

        for param in self.params:
            param[0].data -= self.lr*param[0].grad
            param[1].data -= self.lr*param[1].grad

    def zero_grad(self):
        """Iteratively zeroes the gradients of the weights and biases."""

        for param in self.params:
            param[0].grad = np.zeros_like(param[0].grad)
            param[1].grad = np.zeros_like(param[1].grad)