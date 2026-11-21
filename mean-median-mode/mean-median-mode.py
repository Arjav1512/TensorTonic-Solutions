from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    c = Counter(x)
    m = max(c.values())
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "mode": float(min(k for k, v in c.items() if v == m)),
    }