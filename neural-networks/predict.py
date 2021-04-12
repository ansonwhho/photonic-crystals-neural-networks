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

	#{'r0': 0.305814, 'r1': 0.302837, 'r2': 0.225891, 'r3': 0.396836, 's1': 0.089282, 's2': 0.24393, 's3': 0.477042, 'p1': -0.437686, 'p2': -0.250646, 'p3': -0.20826, 'bandwidth': 0.0436, 'ng0': 14.5394, 'avgLoss': 174.5804, 'GBP': 0.6286, 'loss_at_ng0': 265.4624, 'delay': 11.9783}

	for i in range(num_pred_sets):
		inputPars = {'r0': 0, 'r1': 0, 'r2': 0, 'r3': 0, 's1': 0, 's2': 0, 's3': 0, 'p1': 0, 'p2': 0, 'p3': 0}
		inputPars['r0'] = random.uniform(0.2, 0.4)
		inputPars['r1'] = random.uniform(0.2, 0.4)
		inputPars['r2'] = random.uniform(0.2, 0.4)
		inputPars['r3'] = random.uniform(0.2, 0.4)
		inputPars['s1'] = random.uniform(-0.5, 0.5)
		inputPars['s2'] = random.uniform(-0.5, 0.5)
		inputPars['s3'] = random.uniform(-0.5, 0.5)
		inputPars['p1'] = random.uniform(-0.5, 0.5)
		inputPars['p2'] = random.uniform(-0.5, 0.5)
		inputPars['p3'] = random.uniform(-0.5, 0.5)

		outputData.append(inputPars)

	# Convert to CSV
	outputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/predict-sets/random-pred-2.csv"

	df = pd.DataFrame(outputData)
	df.to_csv(outputCSV)

def predict_save(model, X, output_params, outputCSV):

	pred = pd.DataFrame(model.predict(X), columns=output_params)
	pred_norm, col_mean_SD = preprocessing.zScoreNorm(pred)
	pred_norm.to_csv(outputCSV)

	return col_mean_SD

def eval_preds(model, X_test, y_test, output_params):

	"""
	Evaluates accuracy of predictions
	"""

	# Target predictions
	y_pred = pd.DataFrame(model.predict(X_test), columns=output_params)
	num_out = len(output_params)

	# Compare predictions with test values
	diff = y_pred.to_numpy() - y_test.to_numpy()
	diffDF = pd.DataFrame(diff)
	percent = np.absolute(diff / y_pred.to_numpy()) * 100
	percentDF = pd.DataFrame(percent)

	print("MODEL PERFORMANCE")
	print("y PREDICTIONS")
	print(y_pred)
	print("y ACTUAL")
	print(y_test)
	print("DIFFERENCE")
	print(diffDF)
	print("PERCENTAGE DIFFERENCE")
	print(percentDF)
	print()
	print("MEAN % DIFFERENCE: ", [percent[:, i].mean() for i in range(num_out)])
	print("STD % DIFFERENCE: ", [percent[:, i].std() for i in range(num_out)])

	# Visualise performance
	plot_title = "Predictions vs actual"
	x_label = "y predictions"
	y_label = "y actual"

	# bins = [0, 1, 3, 10, 30, 50, 100, 300, 1000]
	# print(np.histogram(percentDF, bins=bins))
	# percentDF.plot.hist(bins=bins)
	# plt.xscale('log')	
	# plt.show()

	if num_out == 1:
		plt.scatter(y_pred, y_test, s=10, marker="d", facecolors="none", edgecolors="red")
		plt.title(plot_title)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.show()

def main(): 

	"""
	Makes predictions given a set of inputs
	"""

	# Load data to predict on
	all_params = ['GBP', 'avgLoss', 'bandwidth', 'delay', 'loss_at_ng0', 'ng0', 'r0', 'r1', 'r2', 'r3', 's1', 's2', 's3', 'p1', 'p2', 'p3']
	input_params = all_params[6:] # p1 onwards
	# output_params = all_params[:6]
	output_params = [all_params[0]] # GBP only
	inputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/predict-sets/random-pred-2.csv"
	df = pd.read_csv(inputCSV, index_col=0)
	# print(df)

	# Load model
	h5_filepath = "/Users/apple/desktop/photonic-crystals-neural-networks/models/train_2021-04-11_v2.h5"
	model = keras.models.load_model(h5_filepath)

	# Make predictions
	outputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/models/2021-04-11_v2_random-pred-2-PREDICTIONS.csv"
	# predict_save(model, df, output_params, outputCSV)
	df_norm, col_mean_SD = preprocessing.zScoreNorm(df)
	pred = pd.DataFrame(model.predict(df_norm), columns=output_params)
	pred.to_csv(outputCSV)

if __name__ == "__main__":
	# specific_preds(10000000)
	main()

