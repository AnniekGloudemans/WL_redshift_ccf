import numpy as np
import os
from astropy.io import fits
import warnings
import matplotlib.pyplot as plt
import time
import pdb
import plotting as plot
from scipy.stats import kurtosis, skew
import traceback
from astropy import units as u
from astropy.table import Table
from astropy.convolution import convolve, Box1DKernel
from astropy.cosmology import WMAP9 as cosmo
import global_params
import spectres as spec

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

        # AGNstr = [0, 0., 0, 300, 0, 0, 0, 50, 0, 0,0, 0,  0,  
        #             0, 0, 0, 364, 0, 0, 0, 0, 0, 0, 0, 0, 
        #             100, 307, 866, 0, 310, 0]

        AGNLineNames = ['NeVIII', 'Lyb', 'OVI', 'Lya', 'NV', 'CII', 'SiIV', 'CIV', 'HeII', 'OIII-1663', 'CIII', 'CII-2325', 'MgII',
                        'HeII-3203', 'NeV-3346', 'NeV-3426', 'OII', 'HeI', 'Hepsilon', 'NeIII', 'FeV', 'Hdelta', 'Hgamma', 'OIII-4363', 'HeII-4686',
                        'Hb', 'OIII-4959', 'OIII-5007', 'NII-6548', 'Ha', 'NII-6583']

        # # ankur --> no OII line?             
        # AGNLines = [770, 1023, 1035, 1216, 1240, 1336, 1402, 1549, 1640, 1663, 1909, 2326, 2424, 2426, 2470, 2798, 3113,
        #             3203, 3346, 3426, 2727, 2869, 3889, 3970, 3967, 4072, 4102, 4340, 4363, 4686, 4861, 4959, 5007,
        #             6563, 6583]
        # AGNstr = [0, 0, 0, 3100, 154, 37, 163, 364, 318, 72, 177, 92, 41, 49, 41, 78, 11, 14, 20,
        #           69, 364, 82, 19, 21, 26, 16, 22, 24, 8, 20, 100, 307, 866, 310, 0]

        galaxyStr_Norm = AGNstr / np.max(AGNstr) * 1e-17 * 10 # 1e-17*2 adjust height to match spectrum
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


def makeGaus_no_ht(mean, x, std):
    y0 = np.exp(-0.5 * (((x - mean) / std) ** 2))
    return y0


def makeGaus_bimodal_no_ht(x, mean1, std1, mean2, std2):
    return makeGaus_no_ht(mean1,x, std1)+makeGaus_no_ht(mean2,x, std2)


def makeGaus_bimodal(x, mean1, std1, ht1, mean2, std2, ht2):
    return makeGaus(mean1,x, ht1, std1)+makeGaus(mean2,x, ht2, std2)


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
			if global_params.wav_r_min < everyLine[0] * (z + 1) < global_params.wav_r_max:
				shifted = index_data(everyLine[0] * (z + 1), 'r')
				template = makeGaus(shifted, wavs, everyLine[1], std=100) # STD value is *4, because wavelength steps are 0.25 A. Change std value here to fit broader or more narrow template lines
				ySum += template # an array gaussians around the emission lines (shifted with z+1)    
		return ySum

	if colour == 'b':
		for everyLine in template_flux:
			if global_params.wav_b_min < everyLine[0] * (z + 1) < global_params.wav_b_max:
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
		range_begin = global_params.wav_b_min
		range_end = global_params.wav_b_max
		bin_counts = global_params.bin_counts_blue

	elif arm == 'red':
		range_begin = global_params.wav_r_min
		range_end = global_params.wav_r_max
		bin_counts = global_params.bin_counts_red

	log_wvlngth = np.logspace(np.log(range_begin), np.log(range_end), bin_counts, base=np.e) 
	rebin_val = np.zeros(bin_counts) # create empty arrays for new flux values
	rebin_ivar = np.zeros(bin_counts)

	for log_indx in range(0, bin_counts-1):	
		calc_log = np.where(np.logical_and((wavs >= log_wvlngth[log_indx]),
		                                  (wavs < log_wvlngth[log_indx + 1])))
		calc_log_index = (np.asarray(calc_log)).flatten()
		frac_r = (log_wvlngth[log_indx + 1] - wavs[calc_log_index[-1]]) / global_params.wav_steps # divide by 0.25 because original wavelength has resolution of 0.25
		frac_l = (wavs[calc_log_index[0]] - log_wvlngth[log_indx]) / global_params.wav_steps		

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

	#rebin_val, rebin_ivar = spec.spectres(log_wvlngth, wavs, fluxes, std_dev, verbose=False) # Faster! 

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
		wav_min = global_params.wav_b_min 
		wav_max = global_params.wav_b_max
	elif arm == 'r':
		wav_min = global_params.wav_r_min
		wav_max = global_params.wav_r_max  	

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


def preprocess_spectra(spectrum, colour = False, plot_bool = False, savename = False):#, masking_region = np.nan):
	"""
	Preprocess the spectra by taking out masked regions 

	Keyword arguments:
	data_(blue/red) -- blue or red spectrum

	Return:
	"""

	if colour == 'blue':
		wavs = global_params.wavelength_b

	elif colour == 'red':
		wavs = global_params.wavelength_r

	# Fit 6th order polynomial and subtract continuum from the data  
	z = np.polyfit(wavs, spectrum, 6)
	p = np.poly1d(z)

	idx_zero =  np.where(spectrum == 0.0)
	subtracted_spec = spectrum - p(wavs)
	subtracted_spec[idx_zero] = 0.0

	if plot_bool == True:
		plot.plot_continuum_sub(wavs, spectrum, subtracted_spec, p, savename)

	return subtracted_spec


def masking_spectra(spectrum, std_spectrum, masks, colour): 
	"""
	Mask parts of spectrum contaminated with skylines 

	Keyword arguments:

	Return:

	"""
	spectrum_masked_list = []
	spectrum_std_masked_list = []

	if colour == 'blue':
		#idx_wavelenghts = global_params.xaxis_b
		wavelengths = global_params.wavelength_b

	elif colour == 'red':
		#idx_wavelenghts = global_params.xaxis_r
		wavelengths = global_params.wavelength_r

	for i, spec in enumerate(spectrum):

		spec_array = spec
		spec_std_array = std_spectrum[i]

		for j in range(len(masks)): # Loop over masked array

			# Determine the indices corresponding to the wavelengths
			#idx_mask = np.where((wavelengths > masks[i][j][0]) & (wavelengths < masks[i][j][1]))
			idx_mask = np.where((wavelengths > masks[j][0]) & (wavelengths < masks[j][1])) # convert wavelength to index
			
			# Set masked data to zero
			spec_array[idx_mask] = 0.0 #[masks_wav[i][0]:masks_wav[i][1]] = 0.0 #np.nan
			
			spec_std_array[idx_mask] = 0.0 #[masks_wav[i][0]:masks_wav[i][1]] = 0.0#np.nan

		spectrum_masked_list.append(spec_array)
		spectrum_std_masked_list.append(spec_std_array)

	return np.array(spectrum_masked_list), np.array(spectrum_std_masked_list)



def line_inject(luminosity, std_insert, data_blue, std_blue, data_red, std_red, z, line_wav_insert, linestyle, fix_line_flux = False, average_z = False):#, sky_line_bool, data_table, blue_table_full, red_table_full, blue_savename, red_savename):
    """
    Inject a single or double gaussian in a spectrum

	Keyword arguments:
	luminosity (float)  -- 
	std_insert (float)  -- 
	data_(blue/red) -- blue/red spectra
	std_(blue/red) -- blue/red spectra standard deviations 
	z (float) -- redshift of emission line to be injected
	line_wav_insert (float) -- rest-frame wavelength of emission line 
	linestyle (str) -- 'single' or 'double' gaussian
	fix_line_flux (boolean) -- if True average redshift is used to fix the lineflux
	average_z (float/False) -- redshift to fix lineflux to

	Return:
	data_(blue/red)_new (array) -- new spectra with injected gaussian emission line
	
    """

    # Saxena radio galaxy: z=5.7, line flux 1.6e-17, fwhm 370 km/s, lum 5.7e42 erg/s
    if fix_line_flux == True:
        dl = cosmo.luminosity_distance(average_z)
        L = 10**luminosity
        lineflux_insert = (L / (4*np.pi*(dl*(3.08567758*10**24))**2)).value
    else:
        dl = cosmo.luminosity_distance(z)
        L = 10**luminosity
        lineflux_insert = (L / (4*np.pi*(dl*(3.08567758*10**24))**2)).value

    shifted_b = index_data(line_wav_insert * (z + 1), 'b')
    shifted_r = index_data(line_wav_insert * (z + 1), 'r')

    std_insert = std_insert * (1./global_params.wav_steps) # convert the std in Angstrom to the spacing of wavelength grid, might need to change this with real data!! Depends on the resolution

    fit_step = ((max(global_params.xaxis_b)-min(global_params.xaxis_b))/len(global_params.xaxis_b))/(1./global_params.wav_steps)

    if linestyle =='single':
        height_blue = lineflux_insert/np.sum(makeGaus_no_ht(shifted_b, global_params.xaxis_b, std=std_insert)*fit_step)
        height_red = lineflux_insert/np.sum(makeGaus_no_ht(shifted_r, global_params.xaxis_r, std=std_insert)*fit_step)

        Lya_line_template_blue = makeGaus(shifted_b, global_params.xaxis_b, height_blue, std=std_insert) # height and line flux linear relation! used Aayush's source to calibrate, but need to change 2.57e-18 again if change the std
        Lya_line_template_red = makeGaus(shifted_r, global_params.xaxis_r, height_red, std=std_insert)    

        print('height_blue', height_blue)
        print('height_red', height_red)
        print('lineflux_insert', lineflux_insert)
        print('std_insert', std_insert)

    elif linestyle == 'double': # OII --> 3726.032 & 3728.815
        shifted_b_1 = index_data((line_wav_insert-2) * (z + 1), 'b')
        shifted_b_2 = index_data((line_wav_insert+2) * (z + 1), 'b')
        shifted_r_1 = index_data((line_wav_insert-2) * (z + 1), 'r')
        shifted_r_2 = index_data((line_wav_insert+2) * (z + 1), 'r')
        
        height_blue = lineflux_insert/ np.sum(makeGaus_bimodal_no_ht(global_params.xaxis_b, shifted_b_1, std_insert/2., shifted_b_2, std_insert/2.)*fit_step)
        height_red = lineflux_insert/ np.sum(makeGaus_bimodal_no_ht(global_params.xaxis_r, shifted_r_1, std_insert/2., shifted_r_2, std_insert/2.)*fit_step)

        # possibly change stds and heights of the different gaussians?
        Lya_line_template_blue = spec.makeGaus_bimodal(global_params.xaxis_b, shifted_b_1, std_insert/2., height_blue, shifted_b_2, std_insert/2., height_blue)
        Lya_line_template_red = spec.makeGaus_bimodal(global_params.xaxis_r, shifted_r_1, std_insert/2., height_red, shifted_r_2, std_insert/2., height_red)

    #lineflux_func, lineflux_err_func, SNR_func, line_wav_func, real_bool, spurious_bool = fn.fit_line_func(xaxis_r, Lya_line_template_red, z, shifted_r-60, shifted_r+60, shifted_r, str('/Users/anniekgloudemans/Documents/AA_PhD/Data/WEAVE/scripts_dst/Figures/line_fluxes_output/test/'+'K'+str(0)+'_'+'sky'+'_red'), 'sky_line', False, True, False)#, False, False, False)#True) skew_bool=False, plot_bool=True, final_bool = False
    # print('lineflux_func', lineflux_func, '+-', lineflux_err_func) # Not right line flux but shouldnt be, because not right wavelength steps

    data_blue_new = []
    data_red_new = []

    if line_wav_insert * (z+1) < 6000:
        for i in range(len(data_blue)):
            data_blue_new.append(np.array(data_blue[i]+Lya_line_template_blue))
            data_red_new.append(np.array(data_red[i]))
    if line_wav_insert * (z+1) >= 6000: 
        for i in range(len(data_red)):
            data_red_new.append(np.array(data_red[i]+Lya_line_template_red))
            data_blue_new.append(np.array(data_blue[i]))

    return np.array(data_blue_new), np.array(data_red_new)




