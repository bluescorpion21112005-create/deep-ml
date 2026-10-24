import numpy as np

def cramers_rule(A, b):
    # Your code here
    A = np.array(A, dtype=float)
    B = np.array(b, dtype=float)

    det_a = np.linalg.det(A)

    if np.isclose(det_a, 0.0):
        return -1

    n = len(b)
    solutions = []
    for i in range(n):
        a_copy = A.copy()
        a_copy[:, i] = b
        det_a1 = np.linalg.det(a_copy)
        solutions.append(det_a1 / det_a)
    return np.array(solutions)