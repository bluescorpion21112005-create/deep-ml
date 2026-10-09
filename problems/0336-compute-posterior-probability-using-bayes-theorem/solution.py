def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
	"""
	Calculate posterior probabilities using Bayes' Theorem.
	
	Args:
		priors: Prior probabilities P(H_i) for each hypothesis
		likelihoods: Likelihoods P(E|H_i) for each hypothesis
		
	Returns:
		Posterior probabilities P(H_i|E) for each hypothesis
	"""
	numerators = [p * l for p, l in zip(priors, likelihoods)]

	total_prob = sum(numerators)

	if total_prob == 0:
		return [0.0] * len(priors)

	posterios = [num / total_prob for num in numerators]
	return posterios
	
