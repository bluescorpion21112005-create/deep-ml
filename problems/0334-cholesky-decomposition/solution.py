import numpy as np

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    
    Args:
        A: A symmetric positive-definite matrix (2D list or numpy array)
    
    Returns:
        L: Lower triangular matrix such that A = L @ L.T as a 2D list,
           or -1 if decomposition is not possible
    """
    # Your code here
    A = np.array(A, dtype=float)
    L = np.zeros_like(A)
    n = len(A)
    for j in range(n):
        for i in range(j, n):
            sum_val = 0.0
            for k in range(j):
                sum_val += L[i][k] * L[j][k]
            if i == j:
                val = A[i][i] - sum_val
                if val < 0:
                    return -1
                L[i][i] = np.sqrt(val)

            else:
                L[i][j] = (A[i][j] - sum_val) / L[j][j]

    return L.tolist()