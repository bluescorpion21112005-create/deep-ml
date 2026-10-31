import numpy as np

def suggest_rank(delta_W: np.ndarray, energy_threshold: float) -> int:
	"""
	Return the smallest rank k such that the top-k singular values of delta_W
	capture at least `energy_threshold` of the total squared-singular-value energy.
	"""
	# Your code here
	U, s, Vh = np.linalg.svd(delta_W)
	energy = s ** 2
	total_energy = np.sum(energy)

	if total_energy == 0:
		return 0
	target = total_energy * energy_threshold
	current_sum = 0
	for i, e in enumerate(energy):
		current_sum += e
		if current_sum >= target:
			return i + 1
	return target