import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	# Your code here
	P = np.array(P)
	Q = np.array(Q)
	mask = P > 0

	H_P = -np.sum(P[mask] * np.log(P[mask]))


	if np.any((P > 0) & (Q == 0)):
		H_P_Q = float('inf')
	else:
		H_P_Q = -np.sum(P[mask] * np.log(Q[mask]))


	return (round(float(H_P), 6), round(float(H_P_Q), 6))