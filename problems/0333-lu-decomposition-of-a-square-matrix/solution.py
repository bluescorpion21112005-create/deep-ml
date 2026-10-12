import numpy as np

def lu_decomposition(A: list) -> tuple:
    # 1. Listni NumPy massiviga o'girish (float turida, aks holda butun sonlar xato beradi)
    A_np = np.array(A, dtype=np.float64)
    n = A_np.shape[0]
    
    # 2. L va U matritsalarini yaratish
    L = np.eye(n)          # L ning diagonali 1 ga teng bo'lishi shart
    U = np.zeros((n, n))   # U boshida hammasi 0
    
    # 3. Doolittle algoritmi
    for j in range(n):
        # --- U ning j-ustunini hisoblash ---
        for i in range(j + 1):
            s = 0
            for k in range(i):
                s += L[i, k] * U[k, j]
            U[i, j] = A_np[i, j] - s
            
        # --- L ning j-ustunini hisoblash ---
        for i in range(j + 1, n):
            s = 0
            for k in range(j):
                s += L[i, k] * U[k, j]
            L[i, j] = (A_np[i, j] - s) / U[j, j]
            
    return L, U