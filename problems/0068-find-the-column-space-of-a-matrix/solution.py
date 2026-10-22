import numpy as np

def matrix_image(A):
    rows, cols = A.shape
    independent_cols = [] # Mustaqil ustunlarni saqlash uchun bo'sh ro'yxat
    
    for i in range(cols):
        # Vaqtinchalik matritsa yasaymiz: oldingi mustaqillar + hozirgi ustun
        # (A[:, i] - bu A ning i-ustuni)
        temp_matrix = A[:, independent_cols + [i]]
        
        # Agar yangi matritsaning rangi (rank) oldingi rankdan katta bo'lsa,
        # demak bu ustun yangi ma'lumot olib kelyapti
        if np.linalg.matrix_rank(temp_matrix) == len(independent_cols) + 1:
            independent_cols.append(i)
            
    # Topilgan mustaqil ustunlarni asl matritsadan ajratib olamiz
    return A[:, independent_cols]