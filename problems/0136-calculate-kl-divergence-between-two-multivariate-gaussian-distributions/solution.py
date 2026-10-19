import numpy as np

def multivariate_kl_divergence(mu_p: np.ndarray, Cov_p: np.ndarray, mu_q: np.ndarray, Cov_q: np.ndarray) -> float:
    """
    Computes the KL divergence between two multivariate Gaussian distributions.
    
    Parameters:
    mu_p: mean vector of the first distribution
    Cov_p: covariance matrix of the first distribution
    mu_q: mean vector of the second distribution
    Cov_q: covariance matrix of the second distribution

    Returns:
    KL divergence as a float
    """
    # Your code here
    d = len(mu_p)

    inv_cov_q = np.linalg.inv(Cov_q)

    trace_term = np.trace(inv_cov_q @ Cov_p)
    mu_diff = mu_q - mu_p
    mahalanobis = mu_diff.T @ inv_cov_q @ mu_diff

    det_q = np.linalg.det(Cov_q)
    det_p = np.linalg.det(Cov_p)
    log_det_term = np.log(det_q / det_p)


    kl = 0.5 * (trace_term + mahalanobis - d + log_det_term)

    return float(kl)