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
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
# from sklearn import preprocessing
import matplotlib.pyplot as plt

import preprocessing
import architectures as arc
import predict

def normalizer(df):

	"""
	Min-max normalisation, manually written because
	I want to be able to denormalise the outputs later
	(see denormalizer method)
	"""

	col_max_min = {} # For denormalisation

	norm_df = df.copy()
	df_len = df.shape[0]

	for col in df.columns:
		col_max = df[col].max()
		col_min = df[col].min()

		# Min-max scaling
		if col_max == col_min: # Check division by zero
			if col_max == 0:
				norm_df[col] = 0
			else:
				norm_df[col] = 1 / df_len
		else: 
			norm_df[col] = (df[col] - col_min) / (col_max - col_min)
		
		col_max_min[col] = (col_max, col_min)

	
	return norm_df, col_max_min

def denormalizer(norm_df, col_max_min):

	"""
	Reverses min-max normalisation to obtain
	values in outputs in original scale
	"""
	denorm_df = norm_df.copy()
	
	for col in norm_df.columns: 
		col_max = col_max_min[col][0]
		col_min = col_max_min[col][1]
		denorm_df[col] = norm_df[col] * (col_max - col_min) + col_min

	return denorm_df

def preprocessing(dataFrame, all_params, input_params, output_params):

	# Remove data with invalid refractive index
	# i.e. remove data when GBP auto set to zero
	dataFrame_valid = dataFrame[dataFrame.GBP != 0.0000]

	X = dataFrame_valid[input_params] # Features
	y = dataFrame_valid[output_params] # Targets

	# Split data into train, validation sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

	# Normalize targets since different outputs have vastly 
	# different ranges, which would lead to uneven training
	# by the network

	# out_dfs = X_train, X_test, y_train, y_test
	out_dfs = normalizer(X_train), normalizer(y_train), normalizer(X_test), normalizer(y_test)

	return out_dfs

def visual_data(dataFrame):

	# Verify the distribution of results
	pd.plotting.scatter_matrix(dataFrame, diagonal='kde')
	plt.show()

def train_model(features, targets, epochs, split, learn_rate, loss):

	"""
	Trains model with specified hyperparameters
	"""

	feature_dim = features.shape[1]
	target_dim = targets.shape[1]

	model = arc.seq_model(feature_dim, target_dim, learn_rate, loss)

	return model

def save_model(model, date, ver, fileName):
	fileName = "train_{}_v{}.h5".format(date, version)
	model.save(fileName)

def main(): 

	"""
	Creates a trained neural network and saves it
	to a foo.h5 file
	"""

	# Load data
	# inputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/training-sets/run-sets/vary-one-param/2021-03-24_p3_set-1-edit.csv"
	inputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/training-sets/combined-sets/2021-04-07_combined-set.csv"
	all_params = ['GBP', 'avgLoss', 'bandwidth', 'delay', 'loss_at_ng0', 'ng0', 'p1', 'p2', 'p3', 'r0', 'r1', 'r2', 'r3', 's1', 's2', 's3']
	input_params = all_params[6:] # p1 onwards
	# output_params = all_params[:6]
	output_params = [all_params[5]] # single output param

	df = pd.read_csv(inputCSV, names=all_params)
	# df_valid = df[df.GBP != 0.0000]
	# visual_data(df_valid[input_params])

	dataFrames = preprocessing(df, all_params, input_params, output_params)

	# print(dataFrames)

	norm_X_train, X_train_maxmin = dataFrames[0]
	norm_y_train, y_train_maxmin = dataFrames[1]
	norm_X_test, X_test_maxmin = dataFrames[2]
	norm_y_test, y_test_maxmin = dataFrames[3]

	# norm_X_train, X_train_maxmin = dataFrames[0]
	# norm_y_train = dataFrames[1]
	# norm_X_test, X_test_maxmin = dataFrames[2]
	# norm_y_test = dataFrames[3]	

	# print(norm_X_train)

	# Build model
	learn_rate = 5e-4
	loss = 'mae' # Mean absolute error
	epochs = 100
	val_split = 0.1

	model = train_model(norm_X_train, norm_y_train, epochs, val_split, learn_rate, loss)

	history = model.fit(
		norm_X_train, norm_y_train,
		epochs=epochs,
		verbose=0,
		validation_split=val_split
		)

	norm_y_pred = pd.DataFrame(model.predict(norm_X_test), columns=output_params)

	# Compare predictions with actual
	diff = norm_y_pred.to_numpy() - norm_y_test.to_numpy()
	percent = diff / norm_y_pred.to_numpy() * 100
	percentDF = pd.DataFrame(percent)
	diffDF = pd.DataFrame(diff)
	print(percentDF)

	print("NORMALISED y PREDICTIONS")
	print()
	print(norm_y_pred)
	print()
	print("NORMALISED y ACTUAL")
	print()
	print(norm_y_test)
	print()
	print("DIFFERENCE")
	print()
	print(pd.DataFrame(diff))
	print()
	print("MODEL PERFORMANCE")
	print("EPOCHS: ", epochs)
	print("LEARNING RATE: ", learn_rate)
	print("NO. OF LAYERS: ", 1)
	print("NEURONS PER LAYER: ", 16)
	print("ACTIVATION: ", "sigmoid")
	print("MEAN: ", np.mean(diff))
	print("STD: ", np.std(diff))

	# Visualise training
	plot_title = 'Percentage error for ng0'
	x_label = 'Percentage error'
	# y_label = 'Loss'

	# plt.plot(history.history['loss'])
	# plt.plot(history.history['val_loss'])
	# plt.title(plot_title)
	# plt.xlabel(x_label)
	# plt.ylabel(y_label)
	# plt.legend(['loss', 'val_loss'], loc = 'upper left')
	# plt.show()

	# plt.figure();
	bins = [0, 1, 3, 10, 30, 50, 100, 300, 1000]
	print(np.histogram(percentDF, bins=bins))
	percentDF.plot.hist(bins=bins)
	plt.title(plot_title)
	plt.xscale('log')
	plt.xlabel(x_label)
	plt.show()

	# date = 2021-03-28 # YYYY-MM-DD
	# version = 1
	# fileName = "train_{}_v{}.h5".format(date, version)
	# model.save(fileName)

if __name__ == "__main__":
	main()

