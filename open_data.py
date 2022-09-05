from astropy.io import fits
from astropy.table import Table
import numpy as np


def open_input_data(file_path): # Adjust to WEAVE IO
	"""
	Open spectral cube data 

	Keyword arguments:
	file_name (str) -- path to spectral cube
	
	Return:
	file (fits) -- Spectral cube blue arm
	table (table) -- information of spectral cube
	spec (array) -- all spectra from cube
	spec_std (array) -- error on the spectra 

	"""
	
	file = fits.open(file_path)
	table = Table.read(file_path)

	spec = file[1].data*file[5].data
	spec_std = file[3].data*file[5].data

	return file, table, spec, spec_std


def input_redshifts_match(target_table, input_table):
	"""
	Find input redshifts 

	Keyword arguments:
	target_table (table) -- Table with targets from spectral cube
	input_table (table) -- Table with input parameters of simulated data

	Return:
	input_redshifts (array) -- List of input redshifts for all targets 

	"""
	input_redshifts = []

	for i in range(len(target_table)):
		idx_name = np.where(input_table['TARGNAME'] == target_table['TARGNAME'][i])[0] # SHOULD IT NOT BE Z-1??
		if len(input_table['Z_2'][idx_name]) > 0:
			try:
				input_redshifts.append(input_table['Z_2'][idx_name][0])
			except:
				input_redshifts.append(input_table['Z_2'][idx_name]) 
		else:
			input_redshifts.append(np.nan)

	return np.array(input_redshifts)

