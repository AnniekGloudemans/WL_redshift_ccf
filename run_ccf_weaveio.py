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
from weaveio import *

data = Data()
obs = data.obs

global_params.init() # initialize the global parameters 


run_name = 'mockdata_stacked_wl'

input_catalog = Table.read('/beegfs/lofar/duncan/WEAVE/OpR3b/MockData/merged_cat_mags_all_OpR3.fits')

# Get WL mock spectra

l2s = data.l2stacks
stacks_blue = obs.l1stack_spectra[(obs.l1stack_spectra.targuse == 'T') & (obs.l1stack_spectra.camera == 'blue') & any(obs.l1stack_spectra.fibre_target.surveys == '/WL.*/', wrt=obs.l1stack_spectra.fibre_target)] #& obs.l1stack_spectra.surveys == '/WL.*/']#& (obs.ob.id == 3191)] 
stacks_red = obs.l1stack_spectra[(obs.l1stack_spectra.targuse == 'T') & (obs.l1stack_spectra.camera == 'red')& any(obs.l1stack_spectra.fibre_target.surveys == '/WL.*/', wrt=obs.l1stack_spectra.fibre_target)]  #& obs.l1stack_spectra.surveys == '/WL.*/']#& (obs.ob.id == 3191)] 

table_blue = stacks_blue[['wvl', 'flux', 'ivar', 'sensfunc', 'targid', 'targname', 'targprog']]()
table_red = stacks_red[['wvl', 'flux', 'ivar', 'sensfunc', 'targid', 'targname', 'targprog']]()

table_blue_sorted, table_red_sorted = match_blue_red_spectra(table_blue, table_red)

spec_blue = table_blue_sorted['flux']*table_blue_sorted['sensfunc']
spec_std_blue = table_blue_sorted['ivar']*table_blue_sorted['sensfunc']

spec_red = table_red_sorted['flux']*table_red_sorted['sensfunc']
spec_std_red = table_red_sorted['ivar']*table_red_sorted['sensfunc']

input_redshifts = read_data.input_redshifts_match_weaveio(table_red_sorted, input_catalog)


# Run CCF
start = time.time()
redshifts_template_fitting, emission_line_wavelengths, emission_line_fluxes, emission_line_snrs = temp_fit.template_fitting(run_name, spec_blue, spec_std_blue, spec_red, spec_std_red, table_blue_sorted, table_red_sorted, table_red_sorted['targname'], input_redshifts, True)
end = time.time()

print('Full run duration ', (end - start)/60., ' min')


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

