"""
Anson Ho, 2021

Preprocesses data to be used for training of the neural network

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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def removeInvalid(df):

	"""
	Removes experiments with invalid
	group index from dataframe
	i.e. remove data when GBP auto set to zero
	"""

	df_new = df[df.GBP != 0.0000]

	return df_new

def minMaxNorm(df):

	"""
	Min-max normalisation, manually written
	so can denormalise the outputs later
	(see minMaxDenorm)
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

def minMaxDenorm(norm_df, col_max_min):

	"""
	Reverses min-max normalisation to obtain
	values in outputs in original scale
	(see minMaxNorm)
	"""

	denorm_df = norm_df.copy()
	
	for col in norm_df.columns: 
		col_max = col_max_min[col][0]
		col_min = col_max_min[col][1]
		denorm_df[col] = norm_df[col] * (col_max - col_min) + col_min

	return denorm_df

def pipeNormInOut(dataFrame, input_params, output_params):

	"""
	Preprocessing pipeline with normalised inputs
	and outputs - may be desirable if many outputs
	with vastly different ranges, so network does
	not excessively favour one target
	"""

	df = removeInvalid(dataFrame)

	X = df[input_params] # Features
	y = df[output_params] # Targets

	# Split data into train, validation sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

	out_dfs = minMaxNorm(X_train), minMaxNorm(X_test), minMaxNorm(y_train), minMaxNorm(y_test)

	return out_dfs

def pipeNormIn(dataFrame, input_params, output_params):

	"""
	Preprocessing pipeline with normalised inputs
	but without normalised outputs
	(see pipeNormInOut)
	"""	

	df = removeInvalid(dataFrame)

	X = df[input_params] # Features
	y = df[output_params] # Targets

	# Split data into train, validation sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

	out_dfs = minMaxNorm(X_train), minMaxNorm(X_test), y_train, y_test

	return out_dfs

def pipeNormOut(dataFrame, input_params, output_params):

	"""
	Preprocessing pipeline with normalised outputs
	but without normalised inputs
	(see pipeNormInOut)
	"""

	df = removeInvalid(dataFrame)

	X = df[input_params] # Features
	y = df[output_params] # Targets

	# Split data into train, validation sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

	out_dfs = X_train, X_test, minMaxNorm(y_train), minMaxNorm(y_test)

	return out_dfs

def pipeNormNone(dataFrame, input_params, output_params):

	"""
	Preprocessing pipeline without normalisation
	(see pipeNormInOut)
	"""
	df = removeInvalid(dataFrame)

	X = df[input_params] # Features
	y = df[output_params] # Targets

	# Split data into train, validation sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

	return X_train, X_test, y_train, y_test

def visual_data(dataFrame):

	pd.plotting.scatter_matrix(dataFrame, diagonal='kde')
	plt.show()
