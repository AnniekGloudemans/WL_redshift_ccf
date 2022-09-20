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
from astropy.table import Table
from astropy.convolution import convolve, Box1DKernel
import global_params

np.seterr(divide='ignore', invalid='ignore') # ignore runtime warnings 

# Load the names and wavelength of expected emission lines
wavelength_table = Table.read('Input/line_wavelengths.txt', format='ascii')
wavelength_ratios_table = Table.read('Input/line_ratios_sort.txt', format='ascii')
wavelength_ratios_strong_table = Table.read('Input/line_ratios_sort_strong.txt', format='ascii')


def prepare_template(key='AGN'):
    """
    Prepare a template spectrum for AGN and single emission lines

	Keyword arguments:
	template type (str) -- 'AGN' or 'single'

	Return:
	global_params.gal_zip (list) -- fluxes of template 

    """

    if key == 'AGN':

        AGNLines = [770, 1023., 1035, 1216, 1240, 1336, 1402, 1549, 1640, 1663, 1909, 2325, 2798, 
                    3203, 3346, 3426, 3727, 3889, 3970, 3967, 4071, 4102, 4340, 4363, 4686, 
                    4861, 4959, 5007, 6548,6563, 6583]
        AGNstr = [0, 0., 0, 300, 0, 0, 0, 50, 0, 0,0, 0,  0,  
                    0, 0, 0, 364, 0, 0, 0, 0, 0, 0, 0, 0, 
                    100, 307, 866, 0, 310, 0]
        AGNLineNames = ['NeVIII', 'Lyb', 'OVI', 'Lya', 'NV', 'CII', 'SiIV', 'CIV', 'HeII', 'OIII-1663', 'CIII', 'CII-2325', 'MgII',
                        'HeII-3203', 'NeV-3346', 'NeV-3426', 'OII', 'HeI', 'Hepsilon', 'NeIII', 'FeV', 'Hdelta', 'Hgamma', 'OIII-4363', 'HeII-4686',
                        'Hb', 'OIII-4959', 'OIII-5007', 'NII-6548', 'Ha', 'NII-6583']

        galaxyStr_Norm = AGNstr / np.max(AGNstr) * 1e-17 * 2 # 1e-17*2 adjust height to match spectrum
        global_params.gal_zipper = zip(AGNLines, galaxyStr_Norm)
        global_params.gal_ziptmp = list(global_params.gal_zipper)
        global_params.gal_zip = sorted(global_params.gal_ziptmp, key=lambda x: x[1], reverse=True)

    elif key == 'single': # Template with only 1 single lines
        AGNLines = [1216]
        AGNstr = [300]
        AGNLineNames = ['Lya']

        galaxyStr_Norm = AGNstr / np.max(AGNstr) * 1e-17 * 10 
        global_params.gal_zipper = zip(AGNLines, galaxyStr_Norm)
        global_params.gal_ziptmp = list(global_params.gal_zipper)
        global_params.gal_zip = sorted(global_params.gal_ziptmp, key=lambda x: x[1], reverse=True)        

    return global_params.gal_zip


def index_data(lambda1, colour):
    """Converts wavelength to it's respective index. Colour key: 'b' for blue; 'r' for red"""
    if colour == 'b':
        i = (lambda1 - global_params.wav_b_min) / global_params.wav_steps 
    elif colour == 'r':
        i = (lambda1 - global_params.wav_r_min) / global_params.wav_steps
    return int(i)


def makeGaus(mean, x, ht, std):
    y0 = np.exp(-0.5 * (((x - mean) / std) ** 2))
    y = y0 * ht
    return y


def shift_template(template_flux, wavs,  z, colour):
	"""
	Shift the template with redshift 

	Keyword arguments:
	wavs (array) -- wavelengths of template
	template_flux (array) -- fluxes of template

	Return:
	ySum (array) -- array with gaussians around the emission lines (shifted with z+1) 
	"""
	ySum = 0

	if colour == 'r':
		for everyLine in template_flux:
			if 5772 < everyLine[0] * (z + 1) < 9594.25:
				shifted = index_data(everyLine[0] * (z + 1), 'r')
				template = makeGaus(shifted, wavs, everyLine[1], std=100) # Change std value here to fit broader or more narrow template lines
				ySum += template # an array gaussians around the emission lines (shifted with z+1)    
		return ySum

	if colour == 'b':
		for everyLine in template_flux:
			if 3676.0 < everyLine[0] * (z + 1) < 6088.25:
				shifted = index_data(everyLine[0] * (z + 1), 'b')
				template = makeGaus(shifted, wavs, everyLine[1], std=100)
				ySum += template

		return ySum


def rebin(wavs, fluxes, std_dev, arm): 
	
	"""
	Shift the template with redshift 

	Keyword arguments:
	wavs (array) -- wavelengths of spectrum
	template_flux (array) -- fluxes of spectrum
	std_dev (array) -- 
	arm (str) -- 'blue' or 'red'

	Return:
	log_wvlngth (array) -- wavelengths in log space
	rebin_val (array) -- rebinned flux values
	rebin_ivar (array) -- rebinned std values

	"""

	if arm == 'blue':
		range_begin = 3676
		range_end = 6088.25
		bin_counts = 1515

	elif arm == 'red':
		range_begin = 5772
		range_end = 9594.25
		bin_counts = 1526

	log_wvlngth = np.logspace(np.log(range_begin), np.log(range_end), bin_counts, base=np.e) 
	rebin_val = np.zeros(bin_counts) # create empty arrays for new flux values
	rebin_ivar = np.zeros(bin_counts)

	for log_indx in range(0, bin_counts-1):	
		calc_log = np.where(np.logical_and((wavs >= log_wvlngth[log_indx]),
		                                  (wavs < log_wvlngth[log_indx + 1])))
		calc_log_index = (np.asarray(calc_log)).flatten()
		frac_r = (log_wvlngth[log_indx + 1] - wavs[calc_log_index[-1]]) / 0.25 # divide by 0.25 because original wavelength has resolution of 0.25
		frac_l = (wavs[calc_log_index[0]] - log_wvlngth[log_indx]) / 0.25 		

		num_sum = 0
		den_sum = 0

		for i in calc_log_index: 
			#if fluxes[i]!= 0:
			num_sum += fluxes[i] * (1 / np.square(std_dev[i]))
			den_sum += (1 / np.square(std_dev[i]))

		if log_indx != 0:
			#if fluxes[i]!= 0:
			num_sum += frac_l * fluxes[calc_log_index[0] - 1] * (1 / np.square(std_dev[calc_log_index[0] - 1]))
			den_sum += frac_l * (1 / np.square(std_dev[calc_log_index[0] - 1]))

		if calc_log_index[-1] < (len(wavs) - 1):
			#if fluxes[i]!= 0:
			num_sum += frac_r * fluxes[calc_log_index[-1] + 1] * (1 / np.square(std_dev[calc_log_index[-1] + 1]))
			den_sum += + frac_r * (1 / np.square(std_dev[calc_log_index[-1] + 1]))

		rebin_val[log_indx] = num_sum / den_sum
		rebin_ivar[log_indx] = den_sum    

	return log_wvlngth, rebin_val, rebin_ivar



def find_lines(z, arm):
	"""
	Check for the corresponding wavelengths of emission lines in a certain region

	Keyword arguments:
	z (float) -- redshift of target
	arm (str) -- 'b' or 'r'

	Return:
	line_wavs (array) -- wavelengths of possible emission lines in wavelength region
	line_names (array) -- names of possible emission lines in wavelength region

	"""
	if arm == 'b':
		wav_min = 3800.0
		wav_max = 6000.0
	elif arm == 'r':
		wav_min = 6000.0
		wav_max = 9500.0   	

	lines_idx_min = min(enumerate(wavelength_table['wavelength(A)']), key=lambda x: abs(x[1]-(wav_min/(z+1.))))[0] 
	lines_idx_max = min(enumerate(wavelength_table['wavelength(A)']), key=lambda x: abs(x[1]-(wav_max/(z+1.))))[0]

	if wavelength_table['wavelength(A)'][lines_idx_max] > (wav_max/(z+1.)):
		lines_idx_max -= 1

	if wavelength_table['wavelength(A)'][lines_idx_min] < (wav_min/(z+1.)):
		lines_idx_min += 1

	line_wavs = wavelength_table['wavelength(A)'][lines_idx_min:lines_idx_max+1]
	line_names = wavelength_table['name'][lines_idx_min:lines_idx_max+1]

	return np.array(line_wavs), np.array(line_names)


def absorption_spectra_check(data_blue, data_red):
	"""
	Check if any of the spectra are is potentially an absorpion line spectrum 
	by fitting a 6th order polynomial

	Keyword arguments:
	data_(blue/red) -- blue or red spectrum

	Return:
	bool_array (array) -- True if likely absorption line spectrum, False if not

	"""
	bool_array = []

	for i in range(len(data_blue)):

		# Indices to fit only part of the spectrum 
		blue_min = 2000 
		blue_max = 8500
		red_min = 1000
		red_max = 14000

		# fit 6th order polynomial and subtract continuum from the data  
		z_blue = np.polyfit(global_params.wavelength_b, data_blue[i], 6)
		p_blue = np.poly1d(z_blue)

		z_red = np.polyfit(global_params.wavelength_r, data_red[i], 6)
		p_red = np.poly1d(z_red)

		std_red_abs = np.std(data_red[i][red_min:red_max]-p_red(global_params.wavelength_r[red_min:red_max]))
		std_blue_abs = np.std(data_blue[i][blue_min:blue_max]-p_blue(global_params.wavelength_b[blue_min:blue_max]))

		# Check how many measurements are below or above 2 sigma + the average and determine if spectrum is likely absorption spectrum 
		idx_red_down = np.where(((data_red[i][red_min:red_max]-p_red(global_params.wavelength_r[red_min:red_max])) < -2*std_red_abs) & (data_red[i][red_min:red_max] != 0.0))[0]
		idx_blue_down = np.where((data_blue[i][blue_min:blue_max]-p_blue(global_params.wavelength_b[blue_min:blue_max]) < -2*std_blue_abs)& (data_blue[i][blue_min:blue_max] != 0.0) )[0]
		red_down_count = len(idx_red_down)
		blue_down_count = len(idx_blue_down)

		red_up_count = len(np.where(((data_red[i][red_min:red_max]-p_red(global_params.wavelength_r[red_min:red_max])) > 2*std_red_abs) & (data_red[i][red_min:red_max] != 0.0))[0])
		blue_up_count = len(np.where((data_blue[i][blue_min:blue_max]-p_blue(global_params.wavelength_b[blue_min:blue_max]) > 2*std_blue_abs)& (data_blue[i][blue_min:blue_max] != 0.0) )[0])

		clustering_blue = np.std(global_params.wavelength_b[idx_blue_down])
		clustering_red = np.std(global_params.wavelength_r[idx_red_down])

		if red_down_count-red_up_count > 100 or blue_down_count-blue_up_count > 100 or (red_down_count-red_up_count)+(blue_down_count-blue_up_count) > 70 or (blue_down_count-blue_up_count > 25 and clustering_blue < 300) or (red_down_count-red_up_count > 25 and clustering_red < 300):
            
			if np.median(data_red[i]) > 0.1e-16 or np.median(data_blue[i]) > 0.1e-16: # check if this works on real data
				bool_array.append(True)
			else:
				bool_array.append(False)
		else: 
			bool_array.append(False)

	return bool_array
