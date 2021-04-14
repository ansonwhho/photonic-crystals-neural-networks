"""
Anson Ho, 2021

Preprocesses data to be used for training of the neural network

Figures of merit (FOMs) i.e. output parameters: 
- Group-index bandwidth product (GBP)
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
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import itertools
import re

def removeInvalid(df, input_params, output_params):

	"""
	Removes experiments with invalid
	group index from dataframe
	i.e. remove data when GBP auto set to zero

	(Update 2021-04-12): Found bug in constraintsFix.py
	A lot of the previous training data is not
	physically viable. The hallmark of these 
	is that the input params have > 6 s.f. and
	hence will be written in scientific notation.
	Thus remove these values
	Potentially reduces the accuracy of trained
	networks significantly
	"""

	# Fix values that gave too many recursions
	for col in df.columns:
		num_str = str(df[col])

		# Reject if in scientific notation
		if num_str[0] != "0": 
			if ((num_str[0] != "-") or (num_str[1] != "0")):
				df.GBP = 0.0000

	# Remove data with invalid GBP or 
	df_new = df[df.GBP != 0.0000]

	# print(df_new)


	X = df_new[input_params] # Features
	y = df_new[output_params] # Targets

	return X, y

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
		col_max_min[col] = (col_max, col_min)

		# Min-max scaling
		if col_max == col_min: # Check division by zero
			if col_max == 0:
				norm_df[col] = 0
			else:
				norm_df[col] = 1 / df_len
		else: 
			norm_df[col] = (df[col] - col_min) / (col_max - col_min)
	
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

def zScoreNorm(df):

	"""
	Z-score normalisation, manually written
	so can denormalise the outputs later
	(see zScoreDenorm). Theoretically should
	be better at handling outliers than
	min-max normalisation. 
	"""

	col_mean_SD = {} # For denormalisation

	norm_df = df.copy()
	df_len = df.shape[0]

	for col in df.columns:
		col_mean = df[col].mean()
		col_SD = df[col].std()
		col_mean_SD[col] = (col_mean, col_SD)

		# Check division by zero
		if col_SD == 0:
			norm_df[col] = 1 / df_len
		else: 
			norm_df[col] = (df[col] - col_mean) / col_SD

	return norm_df, col_mean_SD

def zScoreDenorm(norm_df, col_mean_SD):

	"""
	Reverses z-score normalisation to obtain
	values in outputs in original scale
	(see zScoreNorm)
	"""

	denorm_df = norm_df.copy()
	
	for col in norm_df.columns: 
		col_mean = col_mean_SD[col][0]
		col_SD = col_mean_SD[col][1]
		denorm_df[col] = norm_df[col] * col_SD + col_mean

	return denorm_df

def pipeline(dataFrame, input_params, output_params, normaliser, norm_settings):
	"""
	Preprocessing pipeline - norm_settings determines
	which of inputs and outputs are normalised.
	Normalising outputs may be desirable if there are
	many outputs with vastly different ranges, so that 
	the network does not excessively favour one target.

	normaliser: "min-max", "z-score"
	norm_settings: "inout", "in", "out", "none"
	"""

	X, y = removeInvalid(dataFrame, input_params, output_params)

	# Split data into train, validation sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

	if normaliser == "min-max":
		if norm_settings == "inout":
			out_dfs = minMaxNorm(X_train), minMaxNorm(X_test), minMaxNorm(y_train), minMaxNorm(y_test)	
		elif norm_settings == "in":
			out_dfs = minMaxNorm(X_train), minMaxNorm(X_test), y_train, y_test
		elif norm_settings == "out":
			out_dfs = X_train, X_test, minMaxNorm(y_train), minMaxNorm(y_test)
		elif norm_settings == "none":
			out_dfs = X_train, X_test, y_train, y_test

	elif normaliser == "z-score":
		if norm_settings == "inout":
			out_dfs = zScoreNorm(X_train), zScoreNorm(X_test), zScoreNorm(y_train), zScoreNorm(y_test)	
		elif norm_settings == "in":
			out_dfs = zScoreNorm(X_train), zScoreNorm(X_test), y_train, y_test
		elif norm_settings == "out":
			out_dfs = X_train, X_test, zScoreNorm(y_train), zScoreNorm(y_test)
		elif norm_settings == "none":
			out_dfs = X_train, X_test, y_train, y_test

	return out_dfs

def which_param(dataFrame):

	"""
	Finds parameters with good/bad FOMs from MPB
	calculations, so that the neural network can 
	be trained on a smaller sample space.

	parameters: 'GBP', 'avgLoss', 'bandwidth', 'delay', 'loss_at_ng0', 'ng0', 
	'p1', 'p2', 'p3', 'r0', 'r1', 'r2', 'r3', 's1', 's2', 's3'

	Look at values within certain thresholds

	"""

	all_params = ['GBP', 'avgLoss', 'bandwidth', 'delay', 'loss_at_ng0', 'ng0', 
		'p1', 'p2', 'p3', 'r0', 'r1', 'r2', 'r3', 's1', 's2', 's3']
	
	# Load data
	input_params = all_params[6:]
	# output_params = all_params[:6]

	df = dataFrame[dataFrame.GBP != 0.0000]

	# Set thresholds
	high = 0.999
	low = 0.001

	GBP_good, GBP_bad = df["GBP"].quantile(high), df["GBP"].quantile(low)
	avgLoss_good, avgLoss_bad = df["avgLoss"].quantile(low), df["avgLoss"].quantile(high)
	bandwidth_good, bandwidth_bad = df["bandwidth"].quantile(high), df["bandwidth"].quantile(low)
	delay_good, delay_bad = df["delay"].quantile(high), df["delay"].quantile(low)
	loss_at_ng0_good, loss_at_ng0_bad = df["loss_at_ng0"].quantile(low), df["loss_at_ng0"].quantile(high)
	ng0_good, ng0_bad = df["ng0"].quantile(high), df["ng0"].quantile(low)

	# Find good values
	# all_good = (df.GBP >= GBP_good and df.avgLoss)

	df_good = df[(df.GBP >= GBP_good) 
		# & (df.avgLoss <= avgLoss_good) 
		# & (df.bandwidth >= bandwidth_good)
		# & (df.delay >= delay_good)
		# & (df.loss_at_ng0 <= loss_at_ng0_good)
		# & (df.ng0 >= ng0_good)
		]

	df_bad = df[(df.GBP <= GBP_bad) 
		& (df.avgLoss >= avgLoss_bad) 
		& (df.bandwidth <= bandwidth_bad)
		& (df.delay <= delay_bad)
		& (df.loss_at_ng0 >= loss_at_ng0_bad)
		& (df.ng0 <= ng0_bad)]

	# 'GBP', 'avgLoss', 'bandwidth', 'delay', 'loss_at_ng0', 'ng0'
	interesting_params = ['delay', 'ng0', 'bandwidth'] + input_params
	good_interesting = df_good[interesting_params]
	bad_interesting = df_bad[interesting_params]

	# df_good_in.plot.kde()

	# g = sns.FacetGrid(df_good_in, col=input_params, height=2, col_wrap=5)

	# n_rows = 2
	# n_cols = 5
	# fig, axs = plt.subplots(n_rows, n_cols)
	# for i, col in enumerate(df_good_in.columns):
	# 	sns.kdeplot(df_good_in[col],ax=axs[i//n_cols,i%n_cols])
	# plt.show()

	print(good_interesting)
	# print(df_bad)
	# print(df_good_in)
	# print(df_bad_in)

def analyse_preds(df, output_params):

	"""
	For exploratory data analysis.
	Visualise data that optimises
	a particular or several FOMs. 
	"""

	all_params = ['GBP', 'avgLoss', 'bandwidth', 'delay', 'loss_at_ng0', 'ng0', 
		'p1', 'p2', 'p3', 'r0', 'r1', 'r2', 'r3', 's1', 's2', 's3']
	input_params = all_params[6:]

	# Set thresholds
	high = 0.99
	low = 0.001

	GBP_good, GBP_bad = df["GBP"].quantile(high), df["GBP"].quantile(low)
	# avgLoss_good, avgLoss_bad = df["avgLoss"].quantile(low), df["avgLoss"].quantile(high)
	# bandwidth_good, bandwidth_bad = df["bandwidth"].quantile(high), df["bandwidth"].quantile(low)
	# delay_good, delay_bad = df["delay"].quantile(high), df["delay"].quantile(low)
	# loss_at_ng0_good, loss_at_ng0_bad = df["loss_at_ng0"].quantile(low), df["loss_at_ng0"].quantile(high)
	# ng0_good, ng0_bad = df["ng0"].quantile(high), df["ng0"].quantile(low)

	df_good = df[(df.GBP >= GBP_good) 
	# & (df.avgLoss <= avgLoss_good) 
	# & (df.bandwidth >= bandwidth_good)
	# & (df.delay >= delay_good)
	# & (df.loss_at_ng0 <= loss_at_ng0_good)
	# & (df.ng0 >= ng0_good)
	]

	print(df_good)

	# outputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/models/designs/candidates/2021-04-11_candidate-set-2-TEST.csv"
	

def csv_row(CSV_file, row_num):

	"""
	Finds the elements given a CSV file and row
	Use for large datasets where Excel cannot
	manage the size
	"""

	with open(CSV_file, 'r') as file:
		print(next(itertools.islice(csv.reader(file), row_num, None)))

def main():
	# inputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/training-sets/run-sets/vary-one-param/2021-03-24_p3_set-1-edit.csv"
	inputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/training-sets/combined-sets/2021-04-13_combined-set.csv"
	# inputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/models/predictions/2021-04-11_v2_random-pred-1-PREDICTIONS.csv"
	# inputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/predict-sets/random-pred-1.csv"
	# inputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/models/designs/candidates/2021-04-11_candidate-set-2-TEST.csv"

	# csv_row(inputCSV, 944038)

	all_params = ['GBP', 'avgLoss', 'bandwidth', 'delay', 'loss_at_ng0', 'ng0', 
		'p1', 'p2', 'p3', 'r0', 'r1', 'r2', 'r3', 's1', 's2', 's3']
	
	# Load data
	input_params = all_params[6:]
	# output_params = all_params[:6]
	output_params = [all_params[0]]
	df = pd.read_csv(inputCSV, names=all_params)
	print(df)

	# # df_norm, col_mean_SD = zScoreNorm(df)

	removeInvalid(df, input_params, output_params)
	# analyse_preds(df, output_params)

	# which_param(df)	

	# df_norm, col_mean_SD = zScoreNorm(df)
	# df_denorm = zScoreDenorm(df_norm, col_mean_SD)

	# print(df)
	# print(df_norm)
	# print(col_mean_SD)
	# print(df_denorm)

	# # Normalise input and output
	# dataFrames = preprocessing.pipeline(df, input_params, output_params, "min-max", "inout")
	# X_train, X_train_maxmin = dataFrames[0]
	# X_test, X_test_maxmin = dataFrames[1]
	# y_train, y_train_maxmin = dataFrames[2]
	# y_test, y_test_maxmin = dataFrames[3]

	# # Only normalise input
	# dataFrames = preprocessing.pipeline(df, input_params, output_params, "in")
	# X_train, X_train_maxmin = dataFrames[0]
	# y_train = dataFrames[1]
	# X_test, X_test_maxmin = dataFrames[2]
	# y_test = dataFrames[3]

	# # Sanity check
	# print(X_train)
	# print(X_test)
	# print(y_train)
	# print(y_test)

	# pd.plotting.scatter_matrix(dataFrame, diagonal='kde')
	# plt.show()

if __name__ == "__main__":
	main()
