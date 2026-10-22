import numpy as np

def rref(matrix):
	#Your code here
	A = matrix.astype(float)
	row, cols = A.shape

	current_row = 0
	for col in range(cols):
		if current_row >= cols:
			break
		
		pivot_idx = None
		for r in range(current_row, row):
			if abs(A[r, col]) > 1e-10:
				pivot_idx = r
				break

		if pivot_idx is None:
			continue
		A[[current_row, pivot_idx]] = A[[pivot_idx, current_row]]

		pivot_val = A[current_row, col]
		A[current_row] = A[current_row]/ pivot_val

		for r in range(row):
			if r != current_row:
				factor = A[r, col]

				A[r] = A[r] - factor * A[current_row]

		current_row += 1

	A[np.abs(A) < 1e-10] = 0.0

	return A
