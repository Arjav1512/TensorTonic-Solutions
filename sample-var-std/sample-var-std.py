import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    a = np.asarray(x, dtype=float)
    var = float(np.sum((a - a.mean()) ** 2) / (a.size - 1))
    return {"variance": var, "standard_deviation": float(np.sqrt(var))}