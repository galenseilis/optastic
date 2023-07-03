import numpy as np
import optuna

import suggestions

class NormToNorm:
    """IO multivariate Gaussian uncertainty."""

    def __init__(self, model, X, Y, loss=None):
        """
        model: Trained SKlearn regressor.
        X numpy.ndarray: Input data.
        Y numpy.ndarry: Output data.
        """

        if loss is None:

            def loss(x, y):
                """Default MSE."""
                return np.mean(np.power(x - y, 2))

        X_dim = X.shape[1]
        Y_dim = Y.shape[1]

        def objective(trial):
            X_corr = suggestions.suggest_corr(trial, "R_x", X_dim)
            Y_corr = suggestions.suggest_corr(trial, "R_y", Y_dim)

        self.objective = objective

    def fit(self, n_trials=3, *args, **kwargs):
        """Fit the stochastic parameters."""
        study = optuna.create_study(*args, **kwargs)
        study.optimize(self.objective, n_trials=n_trials)
        return study

