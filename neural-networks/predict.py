"""
Anson Ho, 2021

Loads trained networks and makes predictions on new data
Evaluates predictions
"""

import pandas as pd 
import numpy as np
from itertools import product
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import preprocessing
import random


def generic_preds():

	"""
	Generates data for prediction systematically
	throughout the input parameter space.
	'Generic' since no specific FOM is being 
	targeted (see specific_preds). 
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

def specific_preds(num_pred_sets):

	"""
	Generates data for prediction randomly
	within a certain range, to test a
	particular FOM. Ranges are set based
	on best values training data, or from
	literature. 'Specific' because looks at
	a specific FOM (see generic_preds).
	"""

	outputData = []

	for i in range(num_pred_sets):
		inputPars = {'r0': 0, 'r1': 0, 'r2': 0, 'r3': 0, 's1': 0, 's2': 0, 's3': 0, 'p1': 0, 'p2': 0, 'p3': 0}
		inputPars['r0'] = random.uniform(0.2, 0.26)
		inputPars['r1'] = random.uniform(0.35, 0.4)
		inputPars['r2'] = random.uniform(0.22, 0.28) + random.randint(0, 1)
		inputPars['r3'] = random.uniform(0.2, 0.4)
		inputPars['s1'] = random.uniform(-0.2, 0.2)
		inputPars['s2'] = random.uniform(-0.1, 0.1)
		inputPars['s3'] = random.uniform(-0.1, 0.1)
		inputPars['p1'] = random.uniform(-0.001, 0.001)
		inputPars['p2'] = random.uniform(-0.001, 0.001)
		inputPars['p3'] = random.uniform(-0.001, 0.001)

		outputData.append(inputPars)

	# Convert to CSV
	outputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/models/GBP-pred-1.csv"

	df = pd.DataFrame(outputData)
	df.to_csv(outputCSV)

def predict_save(model, X, output_params, outputCSV):

	pred = pd.DataFrame(model.predict(X))
	pred.to_csv(outputCSV)

def eval_preds(model, X_test, y_test, output_params):

	"""
	Evaluates accuracy of predictions
	"""

	# Target predictions
	y_pred = pd.DataFrame(model.predict(X_test), columns=output_params)

	# Compare predictions with test values
	diff = y_pred.to_numpy() - y_test.to_numpy()
	diffDF = pd.DataFrame(diff)
	percent = np.absolute(diff / y_pred.to_numpy()) * 100
	percentDF = pd.DataFrame(percent)

	print("MODEL PERFORMANCE")
	print("y PREDICTIONS")
	print(y_pred)
	print("NORMALISED y ACTUAL")
	print(y_test)
	print("DIFFERENCE")
	print(diffDF)
	print("PERCENTAGE DIFFERENCE")
	print(percentDF)
	print()
	print("MEAN % DIFFERENCE: ", np.mean(percent))
	print("STD % DIFFERENCE: ", np.std(percent))

	# Visualise performance
	plot_title = "Predictions vs actual"
	x_label = "y predictions"
	y_label = "y actual"

	# bins = [0, 1, 3, 10, 30, 50, 100, 300, 1000]
	# print(np.histogram(percentDF, bins=bins))
	# percentDF.plot.hist(bins=bins)
	# plt.xscale('log')	
	# plt.show()

	plt.scatter(y_pred, y_test, s=10, marker="d", facecolors="none", edgecolors="red")
	plt.title(plot_title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
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

	# eval_preds()

if __name__ == "__main__":
	specific_preds(10000)

