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
import injecting_lines as inject

import numpy as np
#import os
from astropy.io import fits
import warnings
#import matplotlib.pyplot as plt
import time
#from scipy.stats import kurtosis, skew
#import traceback
#from astropy import units as u
#from astropy.cosmology import WMAP9 as cosmo
#from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
#from astropy.convolution import convolve, Box1DKernel

# Ignore warnings unless debugging
import warnings
warnings.simplefilter("ignore")


run_name = '3130_inject_test_27sep'#'3130_OB_test_25sep_test'#

# -------------------------------------------------- Running on own data files ----------------------------------------------------------
# File paths
red_spectra = '/Users/anniekgloudemans/Documents/AA_PhD/Projects/WL_high_z/Data/simulated_data/L1/OB/3130/stacked/stacked_1002813.fit'
blue_spectra = '/Users/anniekgloudemans/Documents/AA_PhD/Projects/WL_high_z/Data/simulated_data/L1/OB/3130/stacked/stacked_1002814.fit'

# On herts:
# red_spectra = '/home/gloudemans/weave_test_data/OB/3130/stacked/stacked_1002813.fit'
# blue_spectra = '/home/gloudemans/weave_test_data/OB/3130/stacked/stacked_1002814.fit'

# input_catalog = Table.read('/beegfs/lofar/duncan/WEAVE/OpR3b/MockData/merged_cat_mags_all_OpR3.fits') 
# input_redshifts = read_data.input_redshifts_match_weaveio(table_red_sorted, input_catalog)


input_catalog = Table.read("/Users/anniekgloudemans/Documents/AA_PhD/Projects/WL_high_z/Data/simulated_data/input_files/WL_OpR3b_APS_Results_Merged.fits")

# # Read data 
# file_red, table_red, spec_array_red  = read_data.open_input_data_specutils(red_spectra, 'r')
# file_blue, table_blue, spec_array_blue = read_data.open_input_data_specutils(blue_spectra, 'b')

spec_info_red = read_data.readin_spectra(red_spectra)
spec_info_blue = read_data.readin_spectra(blue_spectra)

# target_bool = (table_blue['TARGUSE'] == 'T') # Targets
# sky_line_bool = (table_blue['TARGUSE'] == 'S') # Sky fibers 
target_bool = (spec_info_blue['table']['TARGUSE'] == 'T') # Targets
sky_line_bool = (spec_info_blue['table']['TARGUSE'] == 'S') # Sky fibers 

# Pick specific spectra from OB?
target_bool = read_data.pick_spectrum(target_bool, 500, 1)
sky_line_bool = read_data.pick_spectrum(sky_line_bool, 100, 10)#10)

input_redshifts = read_data.input_redshifts_match(spec_info_red['table'][target_bool], input_catalog)


# -------------------------------------------------- Running on weaveio ----------------------------------------------------------
# from weaveio import *

# data = Data()
# obs = data.obs

# l2s = data.l2stacks
# stacks_blue = obs.l1stack_spectra[(obs.l1stack_spectra.targuse == 'T') & (obs.l1stack_spectra.camera == 'blue') & any(obs.l1stack_spectra.fibre_target.surveys == '/WL.*/', wrt=obs.l1stack_spectra.fibre_target)] #& obs.l1stack_spectra.surveys == '/WL.*/']#& (obs.ob.id == 3191)] 
# stacks_red = obs.l1stack_spectra[(obs.l1stack_spectra.targuse == 'T') & (obs.l1stack_spectra.camera == 'red')& any(obs.l1stack_spectra.fibre_target.surveys == '/WL.*/', wrt=obs.l1stack_spectra.fibre_target)]  #& obs.l1stack_spectra.surveys == '/WL.*/']#& (obs.ob.id == 3191)] 

# table_blue = stacks_blue[['wvl', 'flux', 'ivar', 'sensfunc', 'targid', 'targname', 'targprog']]()
# table_red = stacks_red[['wvl', 'flux', 'ivar', 'sensfunc', 'targid', 'targname', 'targprog']]()

# table_blue_sorted, table_red_sorted = read_data.match_blue_red_spectra(table_blue, table_red) # Sort red and blue spectra 

# # Convert spectra into specutils object
# spec_array_blue = open_input_data_specutils_weaveio(table_blue_sorted, 'b') 
# spec_array_red = open_input_data_specutils_weaveio(table_red_sorted, 'r') 

# # Pick specific spectra?

# # Input redshifts available? 
# input_catalog = Table.read('/beegfs/lofar/duncan/WEAVE/OpR3b/MockData/merged_cat_mags_all_OpR3.fits') 
# input_redshifts = read_data.input_redshifts_match_weaveio(table_red_sorted, input_catalog)



# -------------------------------------------------- Run CCF ----------------------------------------------------------

# Define mask
mask_blue = np.array([[np.nan, np.nan]])
mask_red = np.array([[np.nan, np.nan]])


# Run CCF

# np.nan, 'Lya', np.arange(4.0,4.25,0.25), 10**43.5, 5.
# np.nan, 'OII', np.arange(1.0,1.25,0.25), 10**42.5, 1.5
# 1216., 'gaussian', np.arange(4.0,4.25,0.25), 10**43.5, 5.

# Small grid test:
# Lya
# Redshift: np.nan, 'Lya', np.arange(2.0,7.0,0.5), 10**43.0, 5. with 3 sky lines spectra? 
# Lum/std: np.nan, 'Lya', 5.7, 10**np.arange(42.0,44.0,0.5), np.array([2,8,14]) with 3 sky lines spectra? 

# OII
# Redshift: np.nan, 'OII', np.arange(0.0,2.0.0,0.25), 10**42.0, 1.5 with 3 sky lines spectra? 
# Lum/std: np.nan, 'OII', 1.0, 10**np.arange(41.0,43.0,0.5), np.array([0.5,1.5,3.0]) with 3 sky lines spectra? 

# Gaussian
# Lum/std: 1216., 'gaussian', 5.7, 10**np.arange(42.5,44.0,0.25), np.array([2,5,8,11,14]) 10 skylines # compare to Aug3!

start = time.time()


# # Example injecting a single emission line at different redshifts
# fraction_detected_lines, spurious_detected_lines, fraction_detected_redshift = inject.inject_gaussian_lines(1216., 'gaussian', np.arange(4.0,4.5,0.25), 10**43.5, 5., spec_info_blue['spectra'][sky_line_bool], spec_info_red['spectra'][sky_line_bool], spec_info_blue['table'][sky_line_bool], spec_info_red['table'][sky_line_bool], np.array([str(i) for i in range(np.sum(sky_line_bool))]), run_name,  mask_blue, mask_red, True, True, False, False)

# Example injecting a single emission line with different luminosities and widths
fraction_detected_lines, spurious_detected_lines, fraction_detected_redshift = inject.inject_gaussian_lines(1216., 'gaussian', 5.7, 10**np.arange(42.5,44.0,0.25), np.array([2,5]), spec_info_blue['spectra'][sky_line_bool], spec_info_red['spectra'][sky_line_bool], spec_info_blue['table'][sky_line_bool], spec_info_red['table'][sky_line_bool], np.array([str(i) for i in range(np.sum(sky_line_bool))]), run_name,  mask_blue, mask_red, True, False, False, False)

# Example determining redshift and emission lines for an OB or single spectrum
# redshifts_template_fitting, emission_line_wavelengths, emission_line_fluxes, emission_line_snrs = temp_fit.template_fitting(spec_info_blue['spectra'][target_bool], spec_info_red['spectra'][target_bool], spec_info_blue['table'][target_bool], spec_info_red['table'][target_bool], spec_info_blue['table']['TARGNAME'][target_bool], run_name,  mask_blue, mask_red, input_redshifts, True, True, True, True, False, False, False)#input_redshifts, True, True, True, False, False)

#redshifts_template_fitting, emission_line_wavelengths, emission_line_fluxes, emission_line_snrs = temp_fit.template_fitting(spec_array_blue[target_bool], spec_array_red[target_bool], table_blue[target_bool], table_red[target_bool], table_blue['TARGNAME'][target_bool], run_name,  mask_blue, mask_red, input_redshifts, True, False, False, False, False, False)#input_redshifts, True, True, True, False, False)


end = time.time()
print(np.round((end - start)/60.,2), ' min')

