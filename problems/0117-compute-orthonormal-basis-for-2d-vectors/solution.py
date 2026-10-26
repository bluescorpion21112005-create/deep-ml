import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    # Your code here
    basis = []
    for v in vectors:
        v_k = np.array(v, dtype=float)
        for q in basis:
            l = np.dot(v_k, q)
            v_k = v_k - l * q
        norm_v = np.linalg.norm(v_k)
        if norm_v > tol:
            basis.append(v_k / norm_v)
    return basis

