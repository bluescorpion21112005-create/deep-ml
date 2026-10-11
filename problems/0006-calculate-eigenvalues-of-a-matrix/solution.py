import math
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	a = matrix[0][0]
	b = matrix[0][1]
	c = matrix[1][0]
	d = matrix[1][1]

	trace = a + d
	det = a * d - b * c
	discriminant = (trace ** 2) - (4 * det)

	sqrt_disc = math.sqrt(discriminant)

	lambda1 = (trace + sqrt_disc) / 2
	lambda2 = (trace - sqrt_disc) / 2

	eigenvalues = sorted([lambda1, lambda2], reverse=True)
	return eigenvalues