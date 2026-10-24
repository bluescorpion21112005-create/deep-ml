import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
    # O'nlik kasrlar bilan ishlash uchun float turini ta'minlaymiz
    b = b.astype(float)
    
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.astype(float).copy()
        
    r = b - (A @ x)
    p = r.copy() # Nusxa olish shart! (In-place yangilash uchun)
    
    rsold = r @ r # np.dot(r, r) dan tezroq ishlaydi
    tol_sq = tol ** 2 # Kvadrat ildizdan qochish uchun
    
    for i in range(n):
        a_p = A @ p
        alpha = rsold / (p @ a_p)
        
        x += alpha * p      # In-place yangilash (Tezroq)
        r -= alpha * a_p    # In-place yangilash (Tezroq)
        
        rsnew = r @ r       # Buni erta hisoblaymiz
        
        if rsnew <= tol_sq: # Kvadrat ildizsiz tekshirish
            return x
            
        beta = rsnew / rsold
        p *= beta           # p ni beta ga ko'paytirish
        p += r              # p = r + beta * p ga teng
        
        rsold = rsnew

    return x