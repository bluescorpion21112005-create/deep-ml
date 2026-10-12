import numpy as np

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    # Your code here
    A = np.array(matrix, dtype=np.float64)

    eigenvalues = np.linalg.eigvals(A)
    eigenvalues = np.sort(eigenvalues)
    eigenvalues = np.real(eigenvalues)

    is_pos_def = bool(np.all(eigenvalues > 1e-10))

    rounded_eigenvalues = np.round(eigenvalues, 4).tolist()

    return {
        'is_positive_definite': is_pos_def,
        'eigenvalues': rounded_eigenvalues,
    }