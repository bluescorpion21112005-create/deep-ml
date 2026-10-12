import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
	"""
	Perform QR decomposition using Gram-Schmidt process.
	
	Args:
		A: An m x n matrix represented as list of lists
	
	Returns:
		Tuple of (Q, R) where Q is orthogonal and R is upper triangular
	"""
	# Your code here
	a_np = np.array(A, dtype=np.float64)
	m, n = a_np.shape

	q = np.zeros((m, n))
	r = np.zeros((n, n))


	for j in range(n):
		v = a_np[:, j].copy()
		for i in range(j):
			r[i, j] = np.dot(q[:, i], a_np[:, j])
			v = v - r[i, j] * q[:, i]

		r[j ,j] = np.linalg.norm(v)
		q[:, j] = v / r[j, j]

	return q.tolist(), r.tolist()