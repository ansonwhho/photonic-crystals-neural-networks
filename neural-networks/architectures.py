"""
Anson Ho, 2021

Different neural network architectures for training
Simple method: keras Sequential API
More complex: keras Functional API

Network hyperparameters:
- Learning rate
- Optimizer 
- No. of neurons
- No. of layers

"""

import tensorflow as tf
from tensorflow import keras

def seq_model(in_dim, out_dim, learn_rate, loss):

	# Sequential model with specified dimensions
	# and optimizer

	model = keras.Sequential()
	model.add(keras.layers.Dense(16, input_dim=in_dim, activation='relu'))
	model.add(keras.layers.Dense(16, input_dim=in_dim, activation='relu'))
	# model.add(keras.layers.Dense(16, activation='sigmoid'))
	model.add(keras.layers.Dense(out_dim))

	# SGD most stable
	opt = keras.optimizers.SGD(lr=learn_rate)

	model.compile(
		optimizer=opt,
		loss=loss
		)

	return model

def grid_search(vals, ):
	"""
	Grid search given array vals to find optimal hyperparameters
	"""

	learn_rates = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
	optimisers = ['adam', 'SGD']
	loss_function = ['mae', 'mse']
	neurons = [8, 16, 32, 64, 128]
	epochs = [10, 50, 100, 150, 200]
	activation = ['sigmoid', 'relu', 'tanh']
	weight_init = []
	normalisation = []

	return best_score, best_params


# class seq_model(tf.keras.Sequential):

# 	# Model using keras Sequential API

# 	def __init__(self):
# 		super().__init__()
# 		self.optimizer = 'SGD' # Most stable
# 		self.learn_rate = 0.001
# 		self.in_dim = 10
# 		self.out_dim = 6

# 	def add_layer(self, type, neurons, activation):
# 		self.add(layers.)

# 	def call(self, inputs):

