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

# def accuracy():
# 	score = 0
# 	return score

# def plot():
# 	plt.show()

# def predict(model):
# 	model.predict()

# def eval_preds():

# 	# Evaluates accuracy of predictions

# 	norm_y_pred = pd.DataFrame(model.predict(norm_X_test), columns=output_params)

# 	# Compare predictions with actual
# 	diff = norm_y_pred.to_numpy() - norm_y_test.to_numpy()

# 	print("NORMALISED y PREDICTIONS")
# 	print()
# 	print(norm_y_pred)
# 	print()
# 	print("NORMALISED y ACTUAL")
# 	print()
# 	print(norm_y_test)
# 	print()
# 	print("DIFFERENCE")
# 	print()
# 	print(pd.DataFrame(diff))
# 	print()
# 	print("MODEL PERFORMANCE")
# 	print("EPOCHS: ", epochs)
# 	print("LEARNING RATE: ", learn_rate)
# 	print("NO. OF LAYERS: ", 1)
# 	print("NEURONS PER LAYER: ", 16)
# 	print("ACTIVATION: ", "sigmoid")
# 	print("MEAN: ", np.mean(diff))
# 	print("STD: ", np.std(diff))

# 	plt.show()

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
	main()

