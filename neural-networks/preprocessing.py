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

class minMax:

	"""
	Implements min-max normalisation and denormalisation
	"""

	def norm(self, df):

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

	def denorm(self, norm_df, col_max_min):

		denorm_df = norm_df.copy()
		
		for col in norm_df.columns: 
			col_max = col_max_min[col][0]
			col_min = col_max_min[col][1]
			denorm_df[col] = norm_df[col] * (col_max - col_min) + col_min

		return denorm_df

class zscaling:

	"""
	Implements z-scaling normalisation and denormalisation
	"""

	def norm(self, df):
		return norm_df

	def denorm(self, norm_df):
		return denorm_df

def visual_data(dataFrame):

	# Visualises data for exploration

	pd.plotting.scatter_matrix(dataFrame, diagonal='kde')
	plt.show()
