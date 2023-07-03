import numpy as np
import optuna

import suggestions


class NormToNorm:
    """IO multivariate Gaussian uncertainty."""

    def __init__(self, model, X, Y, loss):
        """
        model: Trained SKlearn regressor.
        X numpy.ndarray: Input data.
        Y numpy.ndarry: Output data.
        """

        X_mean = np.mean(X, axis=0)
        X_residual = X - X_mean
        X_min_residual = np.min(X_residual, axis=0)
        X_max_residual = np.max(X_residual, axis=0)
        X_max_abs_residuals = np.max(np.abs(X_residual), axis=0)

        Y_mean = np.mean(Y, axis=0)
        Y_residual = Y - Y_mean
        Y_min_residual = np.min(Y_residual, axis=0)
        Y_max_residual = np.max(Y_residual, axis=0)
        Y_max_abs_residuals = np.max(np.abs(Y_residual), axis=0)

        def objective(trial):
            # Sample input error distribution
            X_corr = suggestions.suggest_corr(trial, "x_corr", X.shape[1])
            X_mu = [
                trial.suggest_float(f"x_mu_{k}", i, j)
                for k, (i, j) in enumerate(zip(X_min_residual, X_max_residual))
            ]
            X_sigma = [
                trial.suggest_float(f"x_sigma_{k}", 0.0, s)
                for k, s in enumerate(X_max_abs_residuals)
            ]
            X_cov = np.outer((X_sigma,) * 2) * X_corr
            X_sample = np.random.multivariate_normal(X_mu, X_cov, size=X.shape[0])

            # Sample output error distribution
            Y_corr = suggestions.suggest_corr(trial, "y_corr", Y.shape[1])
            Y_mu = [
                trial.suggest_float(f"y_mu_{k}", i, j)
                for k, (i, j) in enumerate(zip(Y_min_residual, Y_max_residual))
            ]
            Y_sigma = [
                trial.suggest_float(f"y_sigma_{k}", 0.0, s)
                for k, s in enumerate(Y_max_abs_residuals)
            ]
            Y_cov = np.outer((Y_sigma,) * 2) * Y_corr
            Y_sample = np.random.multivariate_normal(Y_mu, Y_cov, size=Y.shape[0])

            # Make noisy prediction
            Y_hat = model.predict(X + X_sample) + Y_sample

            return loss(Y, Y_hat)

        self.objective = objective

    def fit(self, n_trials=3, *args, **kwargs):
        """Fit the stochastic parameters."""
        self.study = optuna.create_study(*args, **kwargs)
        self.study.optimize(self.objective, n_trials=n_trials)
        return self.study


# TODO: Real-to-classification for misclassication problems.
# TODO: Real-to-non-negative integer regression for count data
