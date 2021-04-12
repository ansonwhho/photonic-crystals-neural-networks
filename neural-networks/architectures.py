"""
Anson Ho, 2021

Different neural network architectures for training

"""

import tensorflow as tf
from tensorflow import keras
import preprocessing
from sklearn.model_selection import GridSearchCV
import numpy as np 
import pandas as pd

def get_model(in_dim=10, out_dim=1, learn_rate=1e-3, loss='mse', layers=2, neurons=32):

	"""
	Sequential model with specified dimensions
	Defaults set to random values for grid_search
	(see grid_search)
	"""

	model = keras.Sequential()
	model.add(keras.layers.Input(in_dim))

	for i in range(layers):
		model.add(keras.layers.Dense(neurons, input_dim=in_dim, activation='relu'))
		model.add(keras.layers.GaussianNoise(0.1)) # Noise for reproducibility

	model.add(keras.layers.Dense(out_dim))

	# SGD most stable - see grid_search
	opt = keras.optimizers.SGD(lr=learn_rate, momentum=0.9, decay=0.001)

	model.compile(
		optimizer=opt,
		loss=loss
		)

	return model

def grid_search(X, y, cv=5):
	"""
	Grid search given features and targets to find 
	optimal hyperparameters. Focusses on 

	Can also search through fancier things in this
	systematic fashion, e.g. normalisation, optimisers,
	activation, batch size, weight initialisation.
	For this project chose to use SGD and reLU. 
	- SGD most reliable at converging from experimentation
	- reLU for simpler calculations in backpropagation, and
	for better performance in predicting values over
	large range
	- Weight initialisation for dense layers is glorot 
	uniform by default in keras
	"""
	
	in_dim = len(X.columns)
	out_dim = len(y.columns)

	model = keras.wrappers.scikit_learn.KerasRegressor(build_fn=get_model)

	learn_rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
	loss = ['mae', 'mse']
	neurons = [8, 16, 32, 64, 128]
	layers = [1, 2, 3, 4] # Hidden layers
	epochs = [50, 100, 300, 1000, 2000]
	# normalisation = ['min-max', 'z-score']

	# param_grid = {
	# 	# 'learn_rates': learn_rates,
	# 	# 'loss': loss,
	# 	# 'neurons': neurons, 
	# 	'layers': layers
	# 	# 'epochs': epochs,
	# 	# 'normalisation': normalisation
	# 	}

	param_grid = dict(learn_rate=learn_rates, loss=loss, neurons=neurons, layers=layers, epochs=epochs)

	grid = GridSearchCV(
		estimator=model,
		param_grid=param_grid,
		cv=cv,
		verbose=2
		)

	fit_model = grid.fit(X, y)

	# # Also fancier things to search through
	# # in more systematic fashion
	# # e.g. optimisers, activation, 
	# # batch size, weight initialisation
	# optimisers = ['adam', 'SGD']
	# activation = ['sigmoid', 'relu', 'tanh']

	print("Best: {} using {}".format(fit_model.best_score_, fit_model.best_params_))
	means = fit_model.cv_results_['mean_test_score']
	stds = fit_model.cv_results_['std_test_score']
	params = fit_model.cv_results_['params']

	for mean, stdev, param in zip(means, stds, params):
		print("{} ({}) with: {}".format(mean, stdev, param))

def main():
	inputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/training-sets/combined-sets/2021-04-11_combined-set.csv"
	all_params = ['GBP', 'avgLoss', 'bandwidth', 'delay', 'loss_at_ng0', 'ng0', 'p1', 'p2', 'p3', 'r0', 'r1', 'r2', 'r3', 's1', 's2', 's3']
	
	# Load data
	input_params = all_params[6:]
	output_params = all_params[:6]
	# output_params = [all_params[2]]
	df = pd.read_csv(inputCSV, names=all_params)

	# Normalise input and output
	dataFrames = preprocessing.pipeline(df, input_params, output_params, "z-score", "inout")
	X_train, X_train_vals = dataFrames[0]
	X_test, X_test_vals = dataFrames[1]
	y_train, y_train_vals = dataFrames[2]
	y_test, y_test_vals = dataFrames[3]

	# in_dim, out_dim, learn_rate, loss, layers, neurons
	# in_dim = len(input_params)
	# out_dim = len(output_params)

	grid_search(X_train, y_train)

if __name__ == "__main__":
	main()
