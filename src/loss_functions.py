import numpy as np

# MSE loss function
def mse_loss(inp: np.array, targ: np.array):
    """Calculates the mse loss. sum(0.5*(inp-targ)**2).
    
    Args:
        inp (np.array) : This is the output of the previous layer/ probabilities calculated by model
        targ (np.array): This is the ground truth labels.

    Returns:
        (np.array) : Returns the total error calculated.
    """
    # check for the instance
    if isinstance(inp, np.array) and isinstance(targ, np.array):
        return np.sum(0.5*(inp-targ)**2) # 0.5 is used just to cancel out the square value when getting the derivative
    else:
        raise TypeError

# derivative of mse
def mse_derivative(inp: np.array, targ: np.array):
    """Calculates the derivative of MSE Loss which is just inp-targ
    
    Args:
        inp (np.array) : The probabilities calculated by the model
        targ (np.array): The ground truth label

    Returns:
        (np.array) : the derivatives
    """
    # check instances
    if isinstance(inp, np.array) and isinstance(targ, np.array):
        return inp-targ
    else:
        raise TypeError