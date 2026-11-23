
import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    X = np.array(X, dtype=float)
    X_centered = X - np.mean(X, axis=0)
    return (X_centered.T @ X_centered) / (X.shape[0] - 1)
