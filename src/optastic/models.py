import numpy as np
import optuna

class NormToNorm:
	'''IO multivariate Gaussian uncertainty.'''
	def __init__(self, model, X, Y, loss=None):
	'''
	model: Trained SKlearn regressor.
	X numpy.ndarray: Input data.
	Y numpy.ndarry: Output data.
	'''
		if loss is None:
			def loss(x,y):
				'''Default MSE.'''
				return np.mean(np.power(x-y,2))
				
		X_dim = X.shape[1]
		Y_dim = Y.shape[1]
		def objective(trial):
			X_corr = self._suggest_corr(trial, 'x', X_dim)
		self.objective
		
	def fit(self, n_trials=3, *args, **kwargs):
		'''Fit the stochastic parameters.'''
		study = optuna.create_study(*args, **kwargs)
		study.optimize(self.objective, n_trials=n_trials)
		return study

	# TODO: Move this to "suggestions" module
	def _suggest_corr(self, trial, symbol, dim):
		'''Suggest a correlation matrix.'''
		R = np.zeros(shape=(dim,dim), dtype=object)
		for i in range(dim):
			for j in range(dim):
				R[i,j] = trial.suggest_float(
				f'{symbol}_{i}_{j}', 
				-1.0, 
				1.0
				)
		return (R.T @ R) / dim
