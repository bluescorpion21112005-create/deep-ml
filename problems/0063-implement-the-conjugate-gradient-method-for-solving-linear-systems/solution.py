import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
	"""
	Solve the system Ax = b using the Conjugate Gradient method.

	:param A: Symmetric positive-definite matrix
	:param b: Right-hand side vector
	:param n: Maximum number of iterations
	:param x0: Initial guess for solution (default is zero vector)
	:param tol: Convergence tolerance
	:return: Solution vector x
	"""
	# calculate initial residual vector
	if x0 is None:
		x = np.zeros_like(b)
	else:
		x = x0.copy()	
	r = b - (A @ x)
	p = r
	rsold = np.dot(p, r)
	for i in range(n):
		a_p = A @ p
		alpha = rsold / np.dot(p, a_p)
		x = x + alpha * p
		r = r - alpha * a_p
		if np.linalg.norm(r) <= tol:
			return x
		rsnew = np.dot(r, r)
		beta = rsnew / rsold
		p = r + beta * p
		rsold = rsnew
	return x
	
