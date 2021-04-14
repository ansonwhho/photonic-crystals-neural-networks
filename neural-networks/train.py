"""
Anson Ho, 2021

Trains a neural network to predict figures of merit
given a set of input parameters for the W1 PhCW

Figures of merit (FOMs) i.e. output parameters: 
- Gain-bandwidth product (GBP)
- Average loss (avgLoss)
- Bandwidth
- Delay
- Loss at ng0
- ng0

Input parameters: 
- p_i: shift of ith row of holes parallel to waveguide
- r_i: radius for holes in ith row
- s_i: shift of ith row of holes perpendicular to waveguide

"""

import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import preprocessing
import architectures
import predict

def train_history(history):

	"""
	Visualises loss and validation loss vs epochs
	i.e. training history
	"""

	plot_title = 'Loss against epochs'
	x_label = 'Epochs'
	y_label = 'Loss'

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title(plot_title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.legend(['loss', 'val_loss'], loc = 'upper left')
	plt.show()

def main(): 

	"""
	Creates a trained neural network and saves it
	to a foo.h5 file
	"""

	# PREPROCESSING
	# inputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/training-sets/run-sets/vary-one-param/2021-03-24_p3_set-1-edit.csv"
	inputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/training-sets/combined-sets/2021-04-13_combined-set.csv"
	all_params = ['GBP', 'avgLoss', 'bandwidth', 'delay', 'loss_at_ng0', 'ng0', 'p1', 'p2', 'p3', 'r0', 'r1', 'r2', 'r3', 's1', 's2', 's3']
	normaliser = "z-score"
	norm_settings = "inout"
	
	# Load data
	input_params = all_params[6:]
	# output_params = all_params[:6]
	output_params = [all_params[0]]
	df = pd.read_csv(inputCSV, names=all_params)
	
	# Normalise input and output
	dataFrames = preprocessing.pipeline(df, input_params, output_params, normaliser, norm_settings)
	X_train, X_train_vals = dataFrames[0]
	X_test, X_test_vals = dataFrames[1]
	y_train, y_train_vals = dataFrames[2]
	y_test, y_test_vals = dataFrames[3]

	# X_train, X_train_vals = dataFrames[0]
	# X_test, X_test_vals = dataFrames[1]
	# y_train = dataFrames[2]
	# y_test = dataFrames[3]

	# BUILD MODEL
	# Set model and training hyperparameters
	# (see grid_search in architectures.py for more details)
	# Architecture
	in_dim = len(input_params)
	out_dim = len(output_params)
	layers = 4
	neurons = 128

	# Training
	learn_rate = 2e-1
	loss = 'mae'
	epochs = 300
	val_split = 0.1
	patience = 25 # use high patience if avgLoss

	model = architectures.get_model(in_dim, out_dim, learn_rate, loss, layers, neurons)
	early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
	
	history = model.fit(
		X_train, y_train,
		epochs=epochs,
		verbose=0,
		validation_split=val_split,
		callbacks=[early_stop]
		)

	# EVALUATE MODEL
	# Visualise training
	print("TRAINING HYPERPARAMETERS")
	print("INPUTS: ", input_params)
	print("OUTPUTS: ", output_params)
	print("EPOCHS: ", epochs)
	print("LEARNING RATE: ", learn_rate)
	print("NO. OF LAYERS: ", layers)
	print("NEURONS PER LAYER: ", neurons)
	print()

	train_history(history)
	predict.eval_preds(model, X_test, y_test, output_params)

	# # SAVE MODEL
	# date = "2021-04-13" # YYYY-MM-DD
	# version = "1"
	# fileName = "/Users/apple/desktop/photonic-crystals-neural-networks/models/train_{}_v{}.h5".format(date, version)
	# model.save(fileName)

if __name__ == "__main__":
	main()

