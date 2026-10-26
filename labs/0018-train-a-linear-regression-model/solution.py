# --- cell 1 ---
import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    x_t = X.T
    x_new = x_t @ X
    x_new = np.linalg.pinv(x_t @ X)
    b = 0.0
    W = x_new @ x_t @ y
    '''
    x_t = X.T
    lambda_val = 0.1
    I = np.eye(X.shape[1])
    x_new = np.linalg.inv(x_t @ X + lambda_val * I)
    b = 0.0
    W = x_new @ x_t @ y
    '''
    return W, b