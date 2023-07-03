import numpy as np


def suggest_corr(self, trial, symbol, dim):
    """Suggest a correlation matrix."""
    R = np.zeros(shape=(dim, dim), dtype=object)
    for i in range(dim):
        for j in range(dim):
            R[i, j] = trial.suggest_float(f"{symbol}_{i}_{j}", -1.0, 1.0)
    R = (R.T @ R) / dim
    np.fill_diagonal(R, 1.0)
    return R
