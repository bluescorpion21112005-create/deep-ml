import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
    # Boshlang'ich x ni aniqlash
    if x_ini is None:
        x = np.zeros_like(b, dtype=float)
    else:
        x = x_ini.astype(float).copy()
        
    rows = len(b)

    # n marta takrorlaymiz (rows emas!)
    for _ in range(n):
        for i in range(rows):
            sigma = 0.0
            
            # Qolgan elementlarni qo'shamiz (j != i bo'lganda)
            for j in range(rows):
                if i != j:
                    sigma += A[i, j] * x[j]

            # Yangi x[i] ni hisoblaymiz
            x[i] = (b[i] - sigma) / A[i, i]
            
    # Natijani qaytaramiz (nollar emas!)
    return x