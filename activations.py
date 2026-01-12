import numpy as np

# sigmoid activation function
def sigmoid(inp: np.array):
    """Calculates sigmoid function: 1/(1+exp(-inp)). This function squashes the inputs between 0 and 1.
    inputs must be of type np.array.
    
    Args:
        inp (np.array) : input for which the sigmoid needs to be applied.
    
    Returns:
        (np.array) : sigmoid calculation
    """
    # check for instances if they are np.array
    if isinstance(inp, np.array):
        return 1/(1+np.exp(-inp))
    else:
        raise TypeError
    
# sigmoid derivative
def sigmoid_derivative(inp: np.array) -> np.array:
    """Calculates derivative of sigmoid function for backpropagation.
    The derivative of sigmoid is sigmoid(x).(1-sigmoid(x))
    
    Args:
        inp (np.array) : values whose derivatives need to be calculated.
    
    Returns:
        (np.array) : derivative
    """
    # check for the instance
    if isinstance(inp, np.array):
        return inp*(1-inp)
    

    