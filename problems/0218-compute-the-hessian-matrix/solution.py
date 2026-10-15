from typing import Callable

def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
    n = len(point)
    hessian = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Har bir hisoblash uchun asl nuqtaning nusxasini olamiz
            p_i_plus = point.copy()
            p_i_minus = point.copy()
            p_j_plus = point.copy()
            p_j_minus = point.copy()
            
            # i va j indekslarga qadam qo'shamiz/ayiramiz
            p_i_plus[i] += h
            p_i_minus[i] -= h
            p_j_plus[j] += h
            p_j_minus[j] -= h
            
            if i == j:
                # Diagonal elementlar (sof ikkinchi hosila)
                f_plus = f(p_i_plus)
                f_minus = f(p_i_minus)
                f_center = f(point)
                hessian[i][j] = (f_plus - 2 * f_center + f_minus) / (h ** 2)
            else:
                # Diagonal bo'lmagan elementlar (aralash hosila)
                # 4 ta burchak nuqtasini yaratamiz
                pp = point.copy() # +i, +j
                pm = point.copy() # +i, -j
                mp = point.copy() # -i, +j
                mm = point.copy() # -i, -j
                
                pp[i] += h; pp[j] += h
                pm[i] += h; pm[j] -= h
                mp[i] -= h; mp[j] += h
                mm[i] -= h; mm[j] -= h
                
                hessian[i][j] = (f(pp) - f(pm) - f(mp) + f(mm)) / (4 * (h ** 2))
                
    return hessian