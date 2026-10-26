import numpy as np
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	B = np.array(B, dtype=float)
	C = np.array(C, dtype=float)
	C = np.linalg.inv(C)
	P = C @ B
	return P
