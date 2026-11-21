import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x = np.asarray(x, dtype=float)
    result =  1 / (1 + np.exp(-x))
    if result.ndim == 0:
        return float(result)

    return result
