# --- cell 1 ---
import numpy as np

class MyReducer:
    """
    Implement your own dimensionality reduction to 10 dimensions.
    
    Your goal: Project high-dimensional data to 10 dimensions while
    preserving structure for classification.
    
    A k-NN classifier will be trained on your reduced data to evaluate quality.
    """
    
    def __init__(self):
        self.n_components = 10
        # Add any attributes you need to store learned parameters
    
    def fit(self, X):
        """
        Learn the reduction from training data.
        
        Args:
            X: Training data, shape (n_samples, n_features)
        
        Returns:
            self
        """
        # TODO: Analyze X and store what you need for transform()
        self.mean = np.mean(X, axis=0)
        X_c = X - self.mean
        n = X.shape[0]
        cov = (1 / (n - 1)) * (X_c.T @ X_c)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        self.components = eigvecs[:, :self.n_components]
        return self
    
    def transform(self, X):
        """
        Apply the learned reduction to data.
        
        Args:
            X: Data to transform, shape (n_samples, n_features)
        
        Returns:
            X_reduced: shape (n_samples, 10)
        """
        # TODO: Project X to 10 dimensions using parameters from fit()
        x_new = X - self.mean
        result = x_new @ self.components
        return result
        
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)