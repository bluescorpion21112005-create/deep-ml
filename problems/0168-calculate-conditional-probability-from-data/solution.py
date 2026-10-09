def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    # Your code here
    x_count = 0
    xy_count = 0
    for x1, y1 in data:
      if x1 == x:
        x_count += 1
        if y1 == y:
          xy_count += 1
    if x_count == 0:
      return 0.0

    return xy_count / x_count