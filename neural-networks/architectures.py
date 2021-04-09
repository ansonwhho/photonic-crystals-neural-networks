"""
Anson Ho, 2021

Different neural network architectures for training

"""

import tensorflow as tf
from tensorflow import keras

def get_model(in_dim, out_dim, learn_rate, loss, layers, neurons):

	"""
	Sequential model with specified dimensions
	"""

	model = keras.Sequential()
	model.add(keras.layers.Input(in_dim))

	for i in range(layers):
		model.add(keras.layers.Dense(neurons, input_dim=in_dim, activation='relu'))
		model.add(keras.layers.GaussianNoise(0.1)) # Noise for reproducibility

	model.add(keras.layers.Dense(out_dim))

	# SGD most stable - see grid_search
	opt = keras.optimizers.SGD(lr=learn_rate)

	model.compile(
		optimizer=opt,
		loss=loss
		)

	return model

# def single_output(in_dim, learn_rate, loss, layers, neurons):

# 	"""
# 	Sequential model with single output
# 	"""

# 	model = keras.Sequential()
# 	model.add(keras.layers.Input(in_dim))

# 	for i in range(layers):
# 		model.add(keras.layers.Dense(neurons, input_dim=in_dim, activation='relu'))
# 		model.add(keras.layers.GaussianNoise(0.1)) # Noise for reproducibility

# 	model.add(keras.layers.Dense(1))

# 	# SGD most stable - see grid_search
# 	opt = keras.optimizers.SGD(lr=learn_rate)

# 	model.compile(
# 		optimizer=opt,
# 		loss=loss
# 		)

# 	return model

def grid_search(X, y):
	"""
	Grid search given features and targets to find 
	optimal hyperparameters. 
	- SGD most reliable at converging from experimentation
	- reLU for simpler calculations in backpropagation, and
	for better performance in predicting values over
	large range
	"""

	learn_rates = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
	loss = ['mae', 'mse']
	neurons = [8, 16, 32, 64, 128]
	layers = [1, 2, 3, 4]
	epochs = [50, 100, 150, 200, 250]

	param_grid = dict()
	# grid = 

	model = keras.wrappers.scikit_learn.KerasRegressor

	# # Also fancier things to search through
	# # in more systematic fashion
	# # e.g. normalisation, optimisers, activation, 
	# # batch size, weight initialisation
	# optimisers = ['adam', 'SGD']
	# activation = ['sigmoid', 'relu', 'tanh']

	return best_score, best_params

if __name__ == "__main__":
	grid_search()
