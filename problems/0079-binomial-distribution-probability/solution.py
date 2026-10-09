import math

def binomial_probability(n: int, k: int, p: float) -> float:
    """
    Calculate the probability of exactly k successes in n Bernoulli trials.
    
    Args:
        n: Total number of trials
        k: Number of successes
        p: Probability of success on each trial
    
    Returns:
        Probability of k successes
    """
    # Your code here
    if k < 0 or k > n:
        return 0.0

    comb = math.comb(n, k)

    succ = p ** k

    fin = (1 - p) ** (n - k)

    result = comb * succ * fin

    return result