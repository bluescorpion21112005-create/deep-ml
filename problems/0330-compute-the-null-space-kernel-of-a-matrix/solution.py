import numpy as np

def compute_null_space(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    U, sigma, Vh = np.linalg.svd(A)
    
    # 1. Matritsaning haqiqiy rangini topamiz (tol dan katta bo'lganlar soni)
    rank = np.sum(sigma > tol)
    
    # 2. Vh ning rank dan keyingi barcha qatorlarini olamiz va transposeni qilamiz
    null_space_basis = Vh[rank:].T
    
    return null_space_basis