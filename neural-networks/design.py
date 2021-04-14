"""
Anson Ho, 2021

For analysing and designing PhCWs based on the
predictions made by the neural network. 

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import itertools
import preprocessing

def candidate_designs(input_df, pred_df, opt_param, high, low, outputCSV):
	
	"""
	Takes in a dataframe of predictions and selects
	the best performing designs given a certain
	parameter to optimise.

	high and low are thresholds, e.g. high = 0.99
	is used to define the 99th percentile
	opt_param: 'GBP', 'avgLoss', 'bandwidth', 'delay', 'loss_at_ng0', 'ng0'
	"""

	# Find outputs that satisfy threshold
	if opt_param == 'GBP':
		GBP_good, GBP_bad = pred_df["GBP"].quantile(high), pred_df["GBP"].quantile(low)
		candidate_preds = pred_df[pred_df.GBP >= GBP_good]
	elif opt_param == 'avgLoss':
		avgLoss_good, avgLoss_bad = pred_df["avgLoss"].quantile(low), pred_df["avgLoss"].quantile(high)
		candidate_preds = pred_df[pred_df.avgLoss <= avgLoss_good]
	elif opt_param == 'bandwidth':
		bandwidth_good, bandwidth_bad = pred_df["bandwidth"].quantile(high), pred_df["bandwidth"].quantile(low)
		candidate_preds = pred_df[pred_df.bandwidth >= bandwidth_good]
	elif opt_param == 'delay':
		delay_good, delay_bad = pred_df["delay"].quantile(high), pred_df["delay"].quantile(low)
		candidate_preds = df[df.delay >= delay_good]
	elif opt_param == 'loss_at_ng0':
		loss_at_ng0_good, loss_at_ng0_bad = pred_df["loss_at_ng0"].quantile(low), pred_df["loss_at_ng0"].quantile(high)
		candidate_preds = pred_df[pred_df.loss_at_ng0 <= loss_at_ng0_good]
	elif opt_param == 'ng0':
		ng0_good, ng0_bad = pred_df["ng0"].quantile(high), pred_df["ng0"].quantile(low)	
		candidate_preds = pred_df[pred_df.ng0 >= ng0_good]

	# Obtain input params for given outputs
	indexes = list(candidate_preds.index)
	candidate_df = input_df.iloc[indexes,:]

	# Save outputs for testing with MPB
	candidate_df.to_csv(outputCSV)
	# print(candidate_preds)

# def test_candidates(candidate_df):


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

	# Set thresholds
	high = 0.9999
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

	outputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/models/predictions/2021-04-11_v2_design-candidates-GBP.csv"

def csv_row(CSV_file, row_num):

	"""
	Finds the elements given a CSV file and row
	Use for large datasets where Excel cannot
	manage the size
	"""

	with open(CSV_file, 'r') as file:
		print(next(itertools.islice(csv.reader(file), row_num, None)))

def main():
	inputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/predict-sets/random-pred-2.csv" 
	predCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/models/predictions/2021-04-13_v1_random-pred-2-PREDICTIONS.csv"
	
	all_params = ['GBP', 'avgLoss', 'bandwidth', 'delay', 'loss_at_ng0', 'ng0', 
		'p1', 'p2', 'p3', 'r0', 'r1', 'r2', 'r3', 's1', 's2', 's3']
	input_params = all_params[6:]
	# output_params = all_params[:6]
	output_params = [all_params[0]]

	# Load data
	input_df = pd.read_csv(inputCSV, index_col=0)
	pred_df = pd.read_csv(predCSV, index_col=0)
	high = 0.99999
	low = 0.001
	opt_param = 'GBP'
	outputCSV = "/Users/apple/desktop/photonic-crystals-neural-networks/models/designs/candidates/2021-04-13_candidate-set-1.csv"
	candidate_designs(input_df, pred_df, opt_param, high, low, outputCSV)

	# csv_row(inputCSV, 944038)
	# which_param(df)	

if __name__ == "__main__":
	main()
