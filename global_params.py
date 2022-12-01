import numpy as np
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import FlatLambdaCDM

import spec_func as spec

def init():
	global cosmo
	global wavelength_b
	global wavelength_r
	global xaxis_b
	global xaxis_r
	global wav_steps
	global pix_r
	global pix_b
	global pix_set1
	global pix_set2
	global gal_zip
	global initial_redshifts_fitting
	global template_std_b
	global template_std_r
	global wav_r_min
	global wav_b_min
	global wav_r_max
	global wav_b_max
	global bin_counts_blue
	global bin_counts_red

	cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

	wavelength_b = np.arange(9649)*0.25 + 3676
	wavelength_r = np.arange(15289)*0.25 + 5772
	xaxis_b = np.arange(0, 9649, 1) # number of wavelength bins in each spectrum
	xaxis_r = np.arange(0, 15289,1)

	wav_steps = 0.25
	wav_b_min = 3676.0
	wav_b_max = 6088.25
	wav_r_min = 5772.0
	wav_r_max = 9594.25

	bin_counts_blue = 1515
	bin_counts_red = 1526

	pix_r = (np.log(9594.25) - np.log(5772))/1526 # ~ 0.000333
	pix_b = (np.log(6088.25) - np.log(3676))/1515 # ~ 0.000333
	pix_set1 = 2.144698713e-4 
	pix_set2 = 2.648662384e-4

	gal_zip = spec.prepare_template(key='AGN') # Emission line template
	
	template_std_b = np.full((len(wavelength_b), ), 1e-18) # Add noise to template spectrum - now only used in rebinning func
	template_std_r = np.full((len(wavelength_r), ), 1e-18)
	
	initial_redshifts_fitting = np.linspace(0.0,6.8,69)

