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
	model.add(keras.layers.Dense(16, input_dim=in_dim, activation='sigmoid'))
	model.add(keras.layers.Dense(out_dim))

	# SGD most stable
	opt = keras.optimizers.SGD(lr=learn_rate)

	model.compile(
		optimizer=opt,
		loss=loss
		)

	return model

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


# class func_model(tf.keras.Model):
	
	# Model using keras Functional API
