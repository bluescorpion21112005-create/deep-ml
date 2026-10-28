import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    
    # 1. Compute A^T A
    m11 = a*a + c*c
    m12 = a*b + c*d
    m22 = b*b + d*d
    
    # 2. Find Eigenvalues (squared singular values)
    trace = m11 + m22
    det = m11*m22 - m12*m12
    
    disc = trace*trace - 4*det
    if disc < 0:
        disc = 0.0 # Handle floating point precision issues
    sqrt_disc = np.sqrt(disc)
    
    lam1 = (trace + sqrt_disc) / 2.0
    lam2 = (trace - sqrt_disc) / 2.0
    
    # Ensure no negative eigenvalues due to precision
    lam1 = max(lam1, 0.0)
    lam2 = max(lam2, 0.0)
    
    s1 = np.sqrt(lam1)
    s2 = np.sqrt(lam2)
    s = np.array([s1, s2])
    
    # 3. Find V (Right singular vectors)
    if abs(m12) > 1e-10:
        v1 = np.array([m12, lam1 - m11])
    else:
        if m11 >= m22:
            v1 = np.array([1.0, 0.0])
        else:
            v1 = np.array([0.0, 1.0])
            
    norm_v1 = np.linalg.norm(v1)
    if norm_v1 > 0:
        v1 = v1 / norm_v1
    else:
        v1 = np.array([1.0, 0.0])
        
    v2 = np.array([-v1[1], v1[0]])
    
    # 4. Find U (Left singular vectors)
    if s1 > 1e-10:
        u1 = (A @ v1) / s1
    else:
        u1 = np.array([1.0, 0.0])
        
    if s2 > 1e-10:
        u2 = (A @ v2) / s2
    else:
        u2 = np.array([-u1[1], u1[0]])
        
    # 5. Sign correction for deterministic output
    if u1[np.argmax(np.abs(u1))] < 0:
        u1 = -u1
        v1 = -v1
    if u2[np.argmax(np.abs(u2))] < 0:
        u2 = -u2
        v2 = -v2
        
    # Construct matrices
    U = np.column_stack((u1, u2))
    Vs = np.column_stack((v1, v2))
    
    # The problem expects V such that A = U @ diag(s) @ V.
    # Therefore, V is Vs.T (the transpose of the standard right singular vector matrix).
    V_return = Vs.T
    
    return U, s, V_return