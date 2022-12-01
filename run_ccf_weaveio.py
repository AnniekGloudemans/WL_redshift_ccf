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


# Get WL mock spectra

l2s = data.l2stacks
stacks_blue = obs.l1stack_spectra[(obs.l1stack_spectra.targuse == 'T') & (obs.l1stack_spectra.camera == 'blue') & any(obs.l1stack_spectra.fibre_target.surveys == '/WL.*/', wrt=obs.l1stack_spectra.fibre_target)] #& obs.l1stack_spectra.surveys == '/WL.*/']#& (obs.ob.id == 3191)] 
stacks_red = obs.l1stack_spectra[(obs.l1stack_spectra.targuse == 'T') & (obs.l1stack_spectra.camera == 'red')& any(obs.l1stack_spectra.fibre_target.surveys == '/WL.*/', wrt=obs.l1stack_spectra.fibre_target)]  #& obs.l1stack_spectra.surveys == '/WL.*/']#& (obs.ob.id == 3191)] 

table_blue = stacks_blue[['wvl', 'flux', 'ivar', 'sensfunc', 'targid', 'targname', 'targprog']]()
table_red = stacks_red[['wvl', 'flux', 'ivar', 'sensfunc', 'targid', 'targname', 'targprog']]()

table_blue_sorted, table_red_sorted = read_data.match_blue_red_spectra(table_blue, table_red) # Sort red and blue spectra 

spec_blue = table_blue_sorted['flux']*table_blue_sorted['sensfunc']
spec_std_blue = table_blue_sorted['ivar']*table_blue_sorted['sensfunc']

spec_red = table_red_sorted['flux']*table_red_sorted['sensfunc']
spec_std_red = table_red_sorted['ivar']*table_red_sorted['sensfunc']


# Input redshifts? 
input_catalog = Table.read('/beegfs/lofar/duncan/WEAVE/OpR3b/MockData/merged_cat_mags_all_OpR3.fits') 
input_redshifts = read_data.input_redshifts_match_weaveio(table_red_sorted, input_catalog)

# Define mask
mask_blue = np.array([[np.nan, np.nan]])#np.full((len(spec_blue[target_bool]), 2), np.array([[np.nan, np.nan]]))#[4000, 4500])
mask_red = np.array([[8760, 8792], [6827, 6840], [7521, 7532]]) #np.full((len(spec_red[target_bool]), 2), np.array([[8760, 8780], [6827, 6840]]))


# Run CCF
temp_fit.template_fitting(spec_blue, spec_std_blue, spec_red, spec_std_red, table_blue_sorted, table_red_sorted, table_red_sorted['targname'], run_name, mask_blue, mask_red, input_redshift_list, True)# mask_blue = False, mask_red = False, input_redshift_list = input_redshifts, write_to_table_bool = True)


