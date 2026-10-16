import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
    # 1-had: log(sigma_q / sigma_p)
    term1 = np.log(sigma_q / sigma_p)
    
    # 2-had: (sigma_p^2 + (mu_p - mu_q)^2) / (2 * sigma_q^2)
    # Diqqat: sigma_p kvadrat va sigma_q kvadrat o'rinlari to'g'ri bo'lishi kerak!
    term2 = (sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2)
    
    # 3-had: 1/2
    term3 = 1 / 2
    
    # Yakuniy formula
    kl = term1 + term2 - term3
    
    return kl