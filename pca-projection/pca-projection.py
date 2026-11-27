import numpy as np

def pca_projection(X: list, k: int) -> list:
    X = np.array(X, dtype=float)

    X_centered = X - np.mean(X, axis=0)

    cov = X_centered.T @ X_centered / (X.shape[0] - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    indices = np.argsort(eigenvalues)[::-1][:k]
    components = eigenvectors[:, indices]

    for i in range(k):
        idx = np.argmax(np.abs(components[:, i]))
        if components[idx, i] < 0:
            components[:, i] *= -1

    return (X_centered @ components).tolist()