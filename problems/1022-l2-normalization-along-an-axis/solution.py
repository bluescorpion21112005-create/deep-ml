import numpy as np

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> list:
    """
    L2-normalize x along the given axis.

    Args:
        x: input NumPy array
        axis: axis along which to normalize
        eps: small constant for numerical stability

    Returns:
        Normalized array as a nested Python list.
    """
    # Your code here
    sqrt = x ** 2
    sum_sqrt = np.sum(sqrt, axis=axis, keepdims=True)
    norm = np.sqrt(sum_sqrt + eps)
    nomr_x = x /norm
    return nomr_x.tolist()
