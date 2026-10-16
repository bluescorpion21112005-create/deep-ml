import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10):
    eignvalues = np.linalg.eigvalsh(hessian)

    if np.any(np.abs(eignvalues) <= tol):
        return None
    
    if np.all(eignvalues>0):
        return -1
    
    elif np.all(eignvalues<0):
        return 1
    else:
        return 0
