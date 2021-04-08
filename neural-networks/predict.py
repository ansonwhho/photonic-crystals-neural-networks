"""
Anson Ho, 2021

Loads trained networks and makes predictions on new data
Evaluates predictions
"""

import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from itertools import product

def prediction_set():

	"""
	Generates data for prediction systematically
	throughout the input parameter space
	"""

	# initPars = {'r0': 0, 'r1': 0, 'r2': 0, 'r3': 0, 's1': 0, 's2': 0, 's3': 0, 'p1': 0, 'p2': 0, 'p3': 0}
	

	# round used to bypass floating point error
	rRange = [round(x*0.1, 1) for x in range(1, 5+1, 1)] # radii from 0.1 to 0.5
	sRange = [round(x*0.1, 1) for x in range(-5, 5+1, 5)]
	pRange = [round(x*0.1, 1) for x in range(-5, 5+1, 5)]

	# Array for all inputs
	inputArray = [rRange for i in range(4)] + [sRange for i in range(3)] + [pRange for i in range(3)]

	# print(inputPars)

	outputData = []

	# Generates datasets for prediction
	for r0, r1, r2, r3, s1, p1 in product(*inputArray):
		inputPars = {'r0': r0, 'r1': r1, 'r2': r2, 'r3': r3, 's1': s1, 'p1': p1}
		outputData.append(inputPars)

	# Convert to CSV
	outputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/models/predict-grid.csv"

	df = pd.DataFrame(outputData)
	df.to_csv(outputCSV)

	print(outputData)

def predict(model, X_pred):
	
	pred = pd.DataFrame(model.predict(X_pred))

	return pred

def eval_preds(model, X_test, y_test):

	"""
	Evaluates accuracy of predictions
	"""

	y_pred = pd.DataFrame(model.predict(X_test))

	# Compare predictions with actual
	diff = y_pred.to_numpy() - y_test.to_numpy()

	print("NORMALISED y PREDICTIONS")
	print()
	print(y_pred)
	print()
	print("NORMALISED y ACTUAL")
	print()
	print(y_test)
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

	plt.show()

def main(): 

	"""
	Makes predictions given a set of inputs
	"""

	all_params = ['GBP', 'avgLoss', 'bandwidth', 'delay', 'loss_at_ng0', 'ng0', 'p1', 'p2', 'p3', 'r0', 'r1', 'r2', 'r3', 's1', 's2', 's3']
	input_params = all_params[6:] # p1 onwards
	output_params = all_params[:6]
	# output_params = all_params[:1] # GBP only

	# Load model
	h5_filepath = "/path/to/foo.h5"
	model = keras.models.load_model(h5_filepath)

	# norm_y_pred = pd.DataFrame(model.predict(norm_X_test), columns=output_params)

	# # Compare predictions with actual
	# diff = norm_y_pred.to_numpy() - norm_y_test.to_numpy()
	# # print(pd.DataFrame(diff))
	# print("MEAN: ", np.mean(diff))
	# print("STD: ", np.std(diff))

if __name__ == "__main__":
	prediction_set()

