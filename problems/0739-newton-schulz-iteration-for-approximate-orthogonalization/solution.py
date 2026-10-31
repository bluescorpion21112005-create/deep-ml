import numpy as np

def newton_schulz(M, num_iters: int, a: float, b: float, c: float):
    """
    Apply Newton-Schulz iterations to approximately orthogonalize M.
    Returns the resulting matrix as a nested list of floats.
    """
    # Your code here
    X = np.array(M, dtype=float)
    norm = np.linalg.norm(X)
    if norm == 0:
        return X.tolist()

    X = X / norm
    for _ in range(num_iters):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.tolist()