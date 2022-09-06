##################
# Anniek Gloudemans
# 11/08/22

"""
Determine redshift of WEAVE emission line spectra and determine the line fluxes
Developped to detect single emission lines to find high-z sources

"""

# To run without warnings: use python -W ignore run_ccf.py

##################

import temp_fitting as temp_fit
import spec_func as spec
import open_data as read_data
import emission_lines as line
import plotting as plot

import numpy as np
import os
from astropy.io import fits
import warnings
import matplotlib.pyplot as plt
import time
import pdb
from scipy.stats import kurtosis, skew
import traceback
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy.convolution import convolve, Box1DKernel

import global_params

global_params.init() # initialize the global parameters 


# File paths
red_spectra = '/Users/anniekgloudemans/Documents/AA_PhD/Projects/WL_high_z/Data/simulated_data/L1/OB/3130/stacked/stacked_1002813.fit'#/Users/anniekgloudemans/Documents/AA_PhD/Projects/WL_high_z/Data/simulated_data/L1/OB/4376/stacked/stacked_1003809.fit'
blue_spectra = '/Users/anniekgloudemans/Documents/AA_PhD/Projects/WL_high_z/Data/simulated_data/L1/OB/3130/stacked/stacked_1002814.fit'#'/Users/anniekgloudemans/Documents/AA_PhD/Projects/WL_high_z/Data/simulated_data/L1/OB/4376/stacked/stacked_1003810.fit'

# On herts:
# red_spectra = '/home/gloudemans/weave_test_data/OB/3130/stacked/stacked_1002813.fit'
# blue_spectra = '/home/gloudemans/weave_test_data/OB/3130/stacked/stacked_1002814.fit'

run_name = '3130_ccf'#'4367_ccf_normal' Change to OB name 

input_catalog = Table.read("/Users/anniekgloudemans/Documents/AA_PhD/Projects/WL_high_z/Data/simulated_data/input_files/WL_OpR3b_APS_Results_Merged.fits")


# Read data 
file_red, table_red, spec_red, spec_std_red = read_data.open_input_data(red_spectra)
file_blue, table_blue, spec_blue, spec_std_blue = read_data.open_input_data(blue_spectra)

target_bool= (table_blue['TARGUSE'] == 'T') # Targets
sky_line_bool = (table_blue['TARGUSE'] == 'S') # Sky fibers 

# # For testing, picking just 1 target:
# indices = np.arange(0,len(target_bool))
# chosen = indices[target_bool]
# chosen_idx = chosen[83] # just try 1 target
# target_bool[0:chosen_idx] = False
# target_bool[chosen_idx+1:] = False

print('Number targets = ', np.sum(target_bool))

input_redshifts = read_data.input_redshifts_match(table_red[target_bool], input_catalog)


# Run CCF
start = time.time()
redshifts_template_fitting, emission_line_wavelengths, emission_line_fluxes, emission_line_snrs = temp_fit.template_fitting(run_name, spec_blue[target_bool], spec_std_blue[target_bool], spec_red[target_bool], spec_std_red[target_bool], table_blue[target_bool], table_red[target_bool], input_redshifts, True)
end = time.time()

# print('Full run duration ', (end - start)/60., ' min')


# Check performance of ccf if there are input redshifts
try:
	if len(input_redshifts) > 0:
		ratio_redshifts_ccf = Table.read('../Output_ccf/Catalogs/catalog_'+str(run_name)+'_ccf_results_blue.fits')['z_ccf']
		single_emission_lines_ccf = Table.read('../Output_ccf/Catalogs/catalog_'+str(run_name)+'_ccf_results_blue.fits')['single_line_flag']
		#ratio_redshifts_ccf = np.load("ratio_redshifts_all_targets_4367_ccf_strict.npy")

		plot.ccf_performace(input_redshifts, ratio_redshifts_ccf, '../Output_ccf/Figures/ccf_performace_'+run_name+'.pdf', single_emission_lines_ccf)
		diff_z = abs(input_redshifts-ratio_redshifts_ccf)
		idx_nan = np.isnan(ratio_redshifts_ccf)

		print('Percentage correct redshift in total = ', (np.sum(diff_z<0.1)/len(diff_z))*100, '%')
		print('Percentage correct redshift no nan = ', (np.sum(diff_z[~idx_nan]<0.1)/len(diff_z[~idx_nan]))*100, '%')
		print('Number single line targets = ', np.sum(single_emission_lines_ccf == 1))

except Exception as e:
	print(e)
	print('No input redshifts available')

