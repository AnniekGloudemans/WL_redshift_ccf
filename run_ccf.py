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
import ccf_func as ccf

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

run_name = '3130_ccf_test_OB_no_mask'#OB_print'#'3130_ccf_test_sky_inject_lum_44'#'3130_ccf_test_sky_inject_z_fixed_57'#'4367_ccf_normal' Change to OB name # 3130_ccf_test_sky_inject_10_z_lum43_fix_dl z_test_10_01 _10_01_fix_lum_snr4
input_catalog = Table.read("/Users/anniekgloudemans/Documents/AA_PhD/Projects/WL_high_z/Data/simulated_data/input_files/WL_OpR3b_APS_Results_Merged.fits")


# Read data 
file_red, table_red, spec_red, spec_std_red = read_data.open_input_data(red_spectra)
file_blue, table_blue, spec_blue, spec_std_blue = read_data.open_input_data(blue_spectra)


target_bool= (table_blue['TARGUSE'] == 'T') # Targets
sky_line_bool = (table_blue['TARGUSE'] == 'S') # Sky fibers 

# # For testing, picking just 1 target:
# indices = np.arange(0,len(target_bool))
# chosen = indices[target_bool]
# chosen_idx = chosen[1] # just try 1 target 64
# target_bool[0:chosen_idx] = False
# target_bool[chosen_idx+1:] = False

print('Number targets = ', np.sum(target_bool))

input_redshifts = read_data.input_redshifts_match(table_red[target_bool], input_catalog)

#ccf.ccf_test(input_redshifts, spec_blue[target_bool], spec_std_blue[target_bool], spec_red[target_bool], spec_std_red[target_bool], True, '../Output_ccf/Figures/ccf_test3.pdf')

# # Preprocess
# spec.preprocess_spectra(spec_blue[target_bool], spec_std_blue[target_bool], )  # , spec_red[target_bool], spec_std_red[target_bool]

# Define mask
mask_blue = np.array([[np.nan, np.nan]])#np.full((len(spec_blue[target_bool]), 2), np.array([[np.nan, np.nan]]))#[4000, 4500])
mask_red = np.array([[8760, 8792], [6827, 6840], [7521, 7532]]) #np.full((len(spec_red[target_bool]), 2), np.array([[8760, 8780], [6827, 6840]]))


# Run CCF
start = time.time()
#sky_line_bool[40:] = False
# sky_line_bool[105:] = False # gives 10 sources to test on
# print('Num skylines = ', np.sum(sky_line_bool)) # , 
# Also try for just 1 luminosity and many redshifts! 

# # Run also this one --> change run name
# temp_fit.inject_gaussian_lines(1216., 5.7, np.array([42.5, 42.75, 43.0, 43.25, 43.5, 43.75]), np.array([2,5,8,11,14]), spec_blue[sky_line_bool], spec_std_blue[sky_line_bool], spec_red[sky_line_bool], spec_std_red[sky_line_bool], table_blue[sky_line_bool], table_red[sky_line_bool], np.array([str(i) for i in range(np.sum(sky_line_bool))]), run_name,  mask_blue, mask_red, input_redshifts, False)

#temp_fit.inject_gaussian_lines(1216., np.arange(2.0,6.5,0.1), 43.5, 10., spec_blue[sky_line_bool], spec_std_blue[sky_line_bool], spec_red[sky_line_bool], spec_std_red[sky_line_bool], table_blue[sky_line_bool], table_red[sky_line_bool], np.array([str(i) for i in range(np.sum(sky_line_bool))]), run_name,  mask_blue, mask_red, input_redshifts, False)


#temp_fit.inject_gaussian_lines(1216., np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]), np.array([42.5, 42.75, 43.0, 43.25, 43.5, 43.75]), 10., spec_blue[sky_line_bool], spec_std_blue[sky_line_bool], spec_red[sky_line_bool], spec_std_red[sky_line_bool], table_blue[sky_line_bool], table_red[sky_line_bool], table_blue['TARGNAME'][sky_line_bool], run_name,  mask_blue, mask_red, input_redshifts, False)
#temp_fit.inject_gaussian_lines(1216., np.arange(2.0,6.5,0.2), np.array([43.0]), 10., spec_blue[sky_line_bool], spec_std_blue[sky_line_bool], spec_red[sky_line_bool], spec_std_red[sky_line_bool], table_blue[sky_line_bool], table_red[sky_line_bool], table_blue['TARGNAME'][sky_line_bool], run_name,  mask_blue, mask_red, input_redshifts, False)
#temp_fit.inject_gaussian_lines(1216., np.arange(2.0,6.5,0.1), 43.0, 10., spec_blue[sky_line_bool], spec_std_blue[sky_line_bool], spec_red[sky_line_bool], spec_std_red[sky_line_bool], table_blue[sky_line_bool], table_red[sky_line_bool], np.array([str(i) for i in range(np.sum(sky_line_bool))]), run_name,  mask_blue, mask_red, input_redshifts, False)
#temp_fit.inject_gaussian_lines(1216., np.arange(2.7,2.9,0.1), 43.0, 10., spec_blue[sky_line_bool], spec_std_blue[sky_line_bool], spec_red[sky_line_bool], spec_std_red[sky_line_bool], table_blue[sky_line_bool], table_red[sky_line_bool], np.array([str(i) for i in range(np.sum(sky_line_bool))]), run_name,  mask_blue, mask_red, input_redshifts, True)

redshifts_template_fitting, emission_line_wavelengths, emission_line_fluxes, emission_line_snrs = temp_fit.template_fitting(spec_blue[target_bool], spec_std_blue[target_bool], spec_red[target_bool], spec_std_red[target_bool], table_blue[target_bool], table_red[target_bool], table_blue['TARGNAME'][target_bool], run_name,  False, False, input_redshifts, True, True, True)
# SINLGE SPEC#redshifts_template_fitting, emission_line_wavelengths, emission_line_fluxes, emission_line_snrs = temp_fit.template_fitting(spec_blue[target_bool][0], spec_std_blue[target_bool][0], spec_red[target_bool][0], spec_std_red[target_bool][0], table_blue[target_bool], table_red[target_bool], table_blue['TARGNAME'][target_bool][0], run_name, mask_blue, mask_red, input_redshifts[0], True)

end = time.time()
print('Full run duration ', (end - start)/60., ' min')

# Testing
#redshifts_template_fitting, emission_line_wavelengths, emission_line_fluxes, emission_line_snrs = temp_fit.template_fitting(spec_blue[target_bool], spec_std_blue[target_bool], spec_red[target_bool], spec_std_red[target_bool], table_blue[target_bool], table_red[target_bool], table_blue['TARGNAME'][target_bool], run_name,  False, False, False, True, True)


# # Test pix outcome
# ratio_redshifts_ccf = Table.read('../Output_ccf/Catalogs/catalog_'+'3130_ccf_test_new_v5'+'_ccf_results_blue.fits')['z_ccf']
# input_z_ccf = Table.read('../Output_ccf/Catalogs/catalog_'+'3130_ccf_test_new_v5'+'_ccf_results_blue.fits')['z_true']

# diff_z = abs(input_z_ccf-ratio_redshifts_ccf)
# print('Percentage correct redshift in total = ', (np.sum(diff_z<0.1)/len(diff_z))*100, '%')

# ratio_redshifts_ccf_pix = Table.read('../Output_ccf/Catalogs/catalog_'+'3130_ccf_test_new_v5_pix'+'_ccf_results_blue.fits')['z_ccf']
# input_z_ccf_pix = Table.read('../Output_ccf/Catalogs/catalog_'+'3130_ccf_test_new_v5_pix'+'_ccf_results_blue.fits')['z_true']

# diff_z_pix = abs(input_z_ccf_pix-ratio_redshifts_ccf_pix)
# print('Percentage correct redshift in total = ', (np.sum(diff_z_pix<0.1)/len(diff_z_pix))*100, '%')

# list_idx = []
# for i in range(len(diff_z)):
# 	if diff_z[i] < 0.1 and diff_z_pix[i] > 0.1:
# 		print(i, diff_z[i], diff_z_pix[i], input_z_ccf[i]) # any between 0.7-0.8?
# 		list_idx.append(i)

# print(list_idx)

