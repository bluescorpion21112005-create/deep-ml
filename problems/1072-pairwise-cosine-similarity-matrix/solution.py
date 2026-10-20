import numpy as np

def pairwise_cosine_similarity(X):
    # Your code here
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    eps = 1e-10

    x_normalized = X / (norms + eps)

    similarity_matrix = np.dot(x_normalized, x_normalized.T)

    return np.round(similarity_matrix, 4).tolist()