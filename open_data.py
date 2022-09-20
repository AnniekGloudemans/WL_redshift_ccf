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


def input_redshifts_match_weaveio(target_table, input_table):
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
		idx_name = np.where(input_table['TARGNAME'] == target_table['targname'][i])[0] # SHOULD IT NOT BE Z-1??
		if len(input_table['Z'][idx_name]) > 0:
			try:
				input_redshifts.append(input_table['Z'][idx_name][0])
			except:
				input_redshifts.append(input_table['Z'][idx_name]) 
		else:
			input_redshifts.append(np.nan)

	return np.array(input_redshifts)


def match_blue_red_spectra(table_blue, table_red):
	"""
	Sort blue and red spectra to same target order

	Keyword arguments:
	table_blue/red (table) -- Table with all spectra information

	Return:
	table_blue/red (table) -- Sorted/matched table with all spectra information 

	"""

    if len(table_blue) == len(table_red):
        return table_blue, table_red
    
    idx_red_list = []
    idx_blue_list = []

    for i in range(len(table_blue)):
        if table_blue['targname'][i] in table_red['targname']:
            idx_red = np.where(table_red['targname'] == table_blue['targname'][i])[0]

            idx_blue_list.append(i)
            idx_red_list.append(idx_red[0])

    return table_blue[np.array(idx_blue_list)], table_red[np.array(idx_red_list)]



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

