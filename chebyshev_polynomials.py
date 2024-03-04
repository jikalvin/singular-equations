import numpy as np
from scipy.special import chebyu, chebyt

# Vectorize the Chebyshev function
vectorized_chebyt = np.vectorize(chebyt)

def chebyshev_first_kind(x, n):
    return np.cos(n * np.arccos(x))

def chebyshev_second_kind(x, n):
    return np.where(np.abs(x) <= 1, chebyt(n, x), 0.0).item()

def chebyshev_third_kind(x, n):
    return chebyu(n, x)

def chebyshev_fourth_kind(x, n):
    return np.sin((n + 1) * np.arccos(x)) / np.sin(np.arccos(x))
