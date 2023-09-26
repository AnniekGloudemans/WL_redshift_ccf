import numpy as np
import os
from astropy.io import fits
import warnings
import matplotlib.pyplot as plt
import time
import plotting as plot
from scipy.stats import kurtosis, skew
import traceback
from astropy import units as u
from astropy.table import Table
from astropy.convolution import convolve, Box1DKernel
from astropy.cosmology import WMAP9 as cosmo

import spectres as spec
from astropy.nddata import StdDevUncertainty
from specutils import Spectrum1D
from specutils.fitting import fit_generic_continuum
from astropy.cosmology import FlatLambdaCDM

np.seterr(divide='ignore', invalid='ignore') # ignore runtime warnings 
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

# Load the names and wavelength of expected emission lines
wavelength_table = Table.read('Input/line_wavelengths.txt', format='ascii')
wavelength_ratios_table = Table.read('Input/line_ratios_sort.txt', format='ascii')
wavelength_ratios_strong_table = Table.read('Input/line_ratios_sort_strong.txt', format='ascii')

wav_steps = 0.25 # Don't hardcode this! 

def prepare_template(key='AGN'):
    """
    Prepare a template spectrum for AGN and single emission lines

	Keyword arguments:
	template type (str) -- 'AGN' or 'single'

	Return:
	gal_zip (list) -- fluxes of template 

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

        galaxyStr_Norm = AGNstr / np.max(AGNstr) * 1e-17 * 10 # adjust height of template to match spectrum
        gal_zipper = zip(AGNLines, galaxyStr_Norm)
        gal_ziptmp = list(gal_zipper)
        gal_zip = sorted(gal_ziptmp, key=lambda x: x[1], reverse=True)

    elif key == 'single': # Template with only 1 single lines
        AGNLines = [1216]
        AGNstr = [300]
        AGNLineNames = ['Lya']

        galaxyStr_Norm = AGNstr / np.max(AGNstr) * 1e-17 * 10 
        gal_zipper = zip(AGNLines, galaxyStr_Norm)
        gal_ziptmp = list(gal_zipper)
        gal_zip = sorted(gal_ziptmp, key=lambda x: x[1], reverse=True)        

    return gal_zip


def index_data(wavs, wav_min):
    """Converts wavelength to it's respective index. Colour key: 'b' for blue; 'r' for red"""
    i = (wavs - wav_min) / wav_steps
    return int(i)

def makeGaus(mean, x, ht, std):
    y0 = np.exp(-0.5 * (((x - mean) / std) ** 2))
    y = y0 * ht
    return y


def makeGaus_cutoff(mean, x, ht, std):
    y0 = np.exp(-0.5 * (((x - mean) / std) ** 2))
    y = y0 * ht
    idx_cutoff = (x<mean) # create half a gaussian
    y[idx_cutoff] = 0
    return y


def makeGaus_no_ht(mean, x, std):
    y0 = np.exp(-0.5 * (((x - mean) / std) ** 2))
    return y0


def makeGaus_bimodal_no_ht(x, mean1, std1, mean2, std2):
    return makeGaus_no_ht(mean1,x, std1)+makeGaus_no_ht(mean2,x, std2)


def makeGaus_bimodal(x, mean1, std1, ht1, mean2, std2, ht2):
    return makeGaus(mean1,x, ht1, std1)+makeGaus(mean2,x, ht2, std2)


def shift_template(template_flux, wavs_idx, z, colour, wav_b_min, wav_b_max, wav_r_min, wav_r_max):
	"""
	Shift the template with redshift 

	Keyword arguments:
	wavs (array) -- wavelengths of template
	template_flux (array) -- fluxes of template

	Return:
	ySum (array) -- array with gaussians around the emission lines (shifted with z+1) 
	"""
	ySum = 0

	if colour == 'b':
		for everyLine in template_flux:
			if wav_b_min < everyLine[0] * (z + 1) < wav_b_max:
				shifted = index_data(everyLine[0] * (z + 1), wav_b_min)
				template = makeGaus(shifted, wavs_idx, everyLine[1], std=100)
				ySum += template

		return ySum


	if colour == 'r':
		for everyLine in template_flux:
			if wav_r_min < everyLine[0] * (z + 1) < wav_r_max:
				shifted = index_data(everyLine[0] * (z + 1), wav_r_min)
				template = makeGaus(shifted, wavs_idx, everyLine[1], std=100) # STD value is *4, because wavelength steps are 0.25 A. Change std value here to fit broader or more narrow template lines
				ySum += template # an array gaussians around the emission lines (shifted with z+1)    
		return ySum



def find_lines(z, wav_min, wav_max):
	"""
	Check for the corresponding wavelengths of emission lines in a certain region

	Keyword arguments:
	z (float) -- redshift of target
	arm (str) -- 'b' or 'r'

	Return:
	line_wavs (array) -- wavelengths of possible emission lines in wavelength region
	line_names (array) -- names of possible emission lines in wavelength region

	"""	

	lines_idx_min = min(enumerate(wavelength_table['wavelength(A)']), key=lambda x: abs(x[1]-(wav_min/(z+1.))))[0] 
	lines_idx_max = min(enumerate(wavelength_table['wavelength(A)']), key=lambda x: abs(x[1]-(wav_max/(z+1.))))[0]

	if wavelength_table['wavelength(A)'][lines_idx_max] > (wav_max/(z+1.)):
		lines_idx_max -= 1

	if wavelength_table['wavelength(A)'][lines_idx_min] < (wav_min/(z+1.)):
		lines_idx_min += 1

	line_wavs = wavelength_table['wavelength(A)'][lines_idx_min:lines_idx_max+1]
	line_names = wavelength_table['name'][lines_idx_min:lines_idx_max+1]

	return np.array(line_wavs), np.array(line_names)




def absorption_spectra_check(spectra_blue, spectra_red):
	"""
	Check if any of the spectra are is potentially an absorpion line spectrum 
	by fitting a 6th order polynomial

	Keyword arguments:
	data_(blue/red) -- blue or red spectrum

	Return:
	bool_array (array) -- True if likely absorption line spectrum, False if not

	"""
	bool_array = []

	for i in range(len(spectra_blue)):

		# Indices to fit only part of the spectrum 
		blue_min = 2000 
		blue_max = 8500
		red_min = 1000
		red_max = 14000

		# fit 6th order polynomial and subtract continuum from the data   
		z_blue = np.polyfit(spectra_blue[i].spectral_axis.value, spectra_blue[i].flux.value, 6)
		p_blue = np.poly1d(z_blue)

		z_red = np.polyfit(spectra_red[i].spectral_axis.value, spectra_red[i].flux.value, 6)
		p_red = np.poly1d(z_red)

		std_red_abs = np.std(spectra_red[i].flux.value[red_min:red_max]-p_red(spectra_red[i].spectral_axis.value[red_min:red_max]))
		std_blue_abs = np.std(spectra_blue[i].flux.value[blue_min:blue_max]-p_blue(spectra_blue[i].spectral_axis.value[blue_min:blue_max]))

		# Check how many measurements are below or above 2 sigma + the average and determine if spectrum is likely absorption spectrum 
		idx_red_down = np.where(((spectra_red[i].flux.value[red_min:red_max]-p_red(spectra_red[i].spectral_axis.value[red_min:red_max])) < -2*std_red_abs) & (spectra_red[i].flux.value[red_min:red_max] != 0.0))[0]
		idx_blue_down = np.where((spectra_blue[i].flux.value[blue_min:blue_max]-p_blue(spectra_blue[i].spectral_axis.value[blue_min:blue_max]) < -2*std_blue_abs)& (spectra_blue[i].flux.value[blue_min:blue_max] != 0.0) )[0]
		red_down_count = len(idx_red_down)
		blue_down_count = len(idx_blue_down)

		red_up_count = len(np.where(((spectra_red[i].flux.value[red_min:red_max]-p_red(spectra_red[i].spectral_axis.value[red_min:red_max])) > 2*std_red_abs) & (spectra_red[i].flux.value[red_min:red_max] != 0.0))[0])
		blue_up_count = len(np.where((spectra_blue[i].flux.value[blue_min:blue_max]-p_blue(spectra_blue[i].spectral_axis.value[blue_min:blue_max]) > 2*std_blue_abs)& (spectra_blue[i].flux.value[blue_min:blue_max] != 0.0) )[0])

		clustering_blue = np.std(spectra_blue[i].spectral_axis.value[idx_blue_down])
		clustering_red = np.std(spectra_red[i].spectral_axis.value[idx_red_down])

		if red_down_count-red_up_count > 100 or blue_down_count-blue_up_count > 100 or (red_down_count-red_up_count)+(blue_down_count-blue_up_count) > 70 or (blue_down_count-blue_up_count > 25 and clustering_blue < 300) or (red_down_count-red_up_count > 25 and clustering_red < 300):
            
			if np.median(spectra_red[i].flux.value) > 0.1e-16 or np.median(spectra_blue[i].flux.value) > 0.1e-16: # check if this works on real data
				bool_array.append(True)
			else:
				bool_array.append(False)
		else: 
			bool_array.append(False)

	return bool_array


def preprocess_spectra(spectrum, colour = False, plot_bool = False, savename = False):
	"""
	Preprocess the spectra by taking out masked regions 

	Keyword arguments:
	spectrum -- blue or red spectrum
	colour -- blue or red
	plot_bool (boolean) -- default False
	savename (str) -- name save plot

	Return:
	subtracted_spec -- spectrum with polynomial subtracted

	"""

	# Fit 6th order polynomial and subtract continuum from the data  
	z = np.polyfit(spectrum.spectral_axis.value, spectrum.flux.value, 6)
	p = np.poly1d(z)

	idx_zero =  np.where(spectrum.flux.value == 0.0)
	subtracted_spec = spectrum.flux.value - p(spectrum.spectral_axis.value)
	subtracted_spec[idx_zero] = 0.0

	if plot_bool == True:
		plot.plot_continuum_sub(spectrum.spectral_axis.value, spectrum.flux.value, subtracted_spec, p, savename)

	return subtracted_spec


def masking_spectra(spectrum, masks): 
	"""
	Mask parts of spectrum contaminated with skylines 

	Keyword arguments:
	spectrum -- blue or red spectrum
	masks (array) -- wavelength values to mask

	Return:
	spectrum_masked_list (array) -- new spectrum with flux values masked

	"""

	spectrum_masked_list = []

	for i, spec in enumerate(spectrum): # Loop over all spectra

		for j in range(len(masks)): # Loop over masked array
			idx_mask = np.where((spec.spectral_axis.value > masks[j][0]) & (spec.spectral_axis.value < masks[j][1])) # convert wavelength to index
			spec.flux[idx_mask] = 0.0

		spectrum_masked_list.append(spec)

	return np.array(spectrum_masked_list)


def find_chipgaps(spectrum, masks, colour):
	"""
	Finding wavelength of the chipgap in spectrum 

	Keyword arguments:
	spectrum -- blue or red spectrum
	masks -- array with masked wavelength values
	colour (str) -- 'r' or 'b' for red or blue

	Return:
	gap (array) -- min and max wavelength values of the chipgap region

	"""

	mask_bool_array = np.full(len(spectrum.spectral_axis.value), True)
	
	try:
		for j in range(len(masks)): # Loop over masked array
			idx_mask = np.where((spectrum.spectral_axis.value > masks[j][0]) & (spectrum.spectral_axis.value < masks[j][1])) # convert wavelength to index
			mask_bool_array[idx_mask] = False
	except: # if no masks 
		pass


	gap_bool = (spectrum.flux.value == 0.0) * (spectrum.spectral_axis.value > min(spectrum.spectral_axis.value)+100.) * (spectrum.spectral_axis.value < max(spectrum.spectral_axis.value)-100.) * mask_bool_array
	wav_gap = spectrum.spectral_axis.value[gap_bool]

	# if colour == 'b':
	# 	gap_bool = (spectrum.flux.value == 0.0) * (global_params.wavelength_b > 4000.) * (global_params.wavelength_b < 6000.) * mask_bool_array
	# 	wav_gap = global_params.wavelength_b[gap_bool]
	# elif colour == 'r':
	# 	gap_bool = (spectrum.flux.value == 0.0) * (global_params.wavelength_r > 6000.) * (global_params.wavelength_r < 9500.) * mask_bool_array
	# 	wav_gap = global_params.wavelength_r[gap_bool]
	
	gap = np.array([min(wav_gap), max(wav_gap)])

	return gap


def add_to_spectrum(spectra_array, line_template):
	"""
	Inject line into Spectrum1D object 

	Keyword arguments:
	spectra_array (array) - blue/red spectra
	line_template (array) - spectrum with line

	Return:
	spectra_array_injected (array) - new spectrum with line added in specutils format

	"""

	uncertainty = StdDevUncertainty(0.001*1e-17*np.ones(len(spectra_array[0].spectral_axis))*u.Unit('erg cm-2 s-1 AA-1')) # change!! 

	spectra_array_injected = []

	for i in range(len(spectra_array)):
	    flux = spectra_array[i].flux + np.array(line_template) * u.Unit('erg cm-2 s-1 AA-1')
	    spec = Spectrum1D(spectral_axis=spectra_array[i].spectral_axis, flux=flux, uncertainty=uncertainty)#std_spec) SOMETHING WRONG WITH STD_SPEC
	    spectra_array_injected.append(spec)

	return spectra_array_injected


def line_inject(luminosity, std_insert, spectra_blue, spectra_red, z, line_wav_insert, line_type, fix_line_flux = False, average_z = False):#, sky_line_bool, data_table, blue_table_full, red_table_full, blue_savename, red_savename):
    """
    Inject a single or double gaussian in a spectrum

	Keyword arguments:
	luminosity (float)  -- luminosity in erg/s
	std_insert (float)  -- std of gaussian in Angstrom
	data_(blue/red) -- blue/red spectra
	std_(blue/red) -- blue/red spectra standard deviations 
	z (float) -- redshift of emission line to be injected
	line_wav_insert (float) -- rest-frame wavelength of emission line (NaN value allowed for OII and Lya)
	line_type (str) -- type of line 'Lya', 'OII', or 'gaussian'
	fix_line_flux (boolean) -- if True average redshift is used to fix the lineflux
	average_z (float/False) -- redshift to fix lineflux to

	Return:
	data_(blue/red)_new (array) -- new spectra with injected gaussian emission line
	
    """

    # Saxena radio galaxy: z=5.7, line flux 1.6e-17, fwhm 370 km/s, lum 5.7e42 erg/s

    #---------------------------------------- Prepare general line features -----------------------------------#

    if fix_line_flux == True:
        dl = cosmo.luminosity_distance(average_z)
        lineflux_insert = (luminosity / (4*np.pi*(dl*(3.08567758*10**24))**2)).value
    else:
        dl = cosmo.luminosity_distance(z)
        lineflux_insert = (luminosity / (4*np.pi*(dl*(3.08567758*10**24))**2)).value

    std_insert = std_insert * (1./wav_steps) # convert the std in Angstrom to the spacing of wavelength grid, might need to change this with real data!! Depends on the resolution
    
    xaxis_b = np.arange(0, len(spectra_blue[0].spectral_axis), 1)
    xaxis_r = np.arange(0, len(spectra_red[0].spectral_axis), 1)

    # fit_step = #((max(global_params.xaxis_b)-min(global_params.xaxis_b))/len(global_params.xaxis_b))/(1./global_params.wav_steps)
    # print('fit_step', fit_step)
    #---------------------------------------- Preparing single emission line -----------------------------------#

    if line_type =='gaussian':
    	
        shifted_b = index_data(line_wav_insert * (z + 1), np.min(spectra_blue[0].spectral_axis.value))#'b')
        shifted_r = index_data(line_wav_insert * (z + 1), np.min(spectra_red[0].spectral_axis.value))#'r')

        height_blue = lineflux_insert/np.sum(makeGaus_no_ht(shifted_b, xaxis_b, std=std_insert)*wav_steps)
        height_red = lineflux_insert/np.sum(makeGaus_no_ht(shifted_r, xaxis_r, std=std_insert)*wav_steps)

        # Could also use models.Gaussian1D(amplitude=3.*u.Jy, mean=61000*u.AA, stddev=10000.*u.AA) from specutils
        line_template_blue = makeGaus(shifted_b, xaxis_b, height_blue, std=std_insert)
        line_template_red = makeGaus(shifted_r, xaxis_r, height_red, std=std_insert)    


    #---------------------------------------- Preparing OII emission line -----------------------------------#

    elif line_type == 'OII': # Doublet at 3726.032 & 3728.815

        line_wav_insert = (3726.032 + 3728.815)/2.

        shifted_b_1 = index_data((3726.032) * (z + 1), np.min(spectra_blue[0].spectral_axis.value))#'b')
        shifted_b_2 = index_data((3728.815) * (z + 1), np.min(spectra_blue[0].spectral_axis.value))#'b')
        shifted_r_1 = index_data((3726.032) * (z + 1), np.min(spectra_red[0].spectral_axis.value))#'r')
        shifted_r_2 = index_data((3728.815) * (z + 1), np.min(spectra_red[0].spectral_axis.value))#'r')
        
        # Take ratio of peaks to be 1 for now 
        # Both lines have the inserted STD
        height_blue = lineflux_insert / np.sum(makeGaus_bimodal_no_ht(xaxis_b, shifted_b_1, std_insert, shifted_b_2, std_insert)*wav_steps)
        height_red = lineflux_insert / np.sum(makeGaus_bimodal_no_ht(xaxis_r, shifted_r_1, std_insert, shifted_r_2, std_insert)*wav_steps)

        # Possibly change stds and heights of the different gaussians
        line_template_blue = makeGaus_bimodal(xaxis_b, shifted_b_1, std_insert, height_blue, shifted_b_2, std_insert, height_blue)
        line_template_red = makeGaus_bimodal(xaxis_r, shifted_r_1, std_insert, height_red, shifted_r_2, std_insert, height_red)
 
    #---------------------------------------- Preparing Lya emission line -----------------------------------#

    elif line_type == 'Lya': # Cut off blue part of gaussian 

        line_wav_insert = 1215.67

        shifted_b = index_data(line_wav_insert * (z + 1), np.min(spectra_blue[0].spectral_axis.value))#'b')
        shifted_r = index_data(line_wav_insert * (z + 1), np.min(spectra_red[0].spectral_axis.value))#'r')

        # Make gaussian line with injected line flux
        # line flux times 2 because only half of the line is recovered
        height_blue = (lineflux_insert*2)/np.sum(makeGaus_no_ht(shifted_b, xaxis_b, std=std_insert)*wav_steps)
        height_red = (lineflux_insert*2)/np.sum(makeGaus_no_ht(shifted_r, xaxis_r, std=std_insert)*wav_steps)
        line_template_blue = makeGaus_cutoff(shifted_b, xaxis_b, height_blue, std=std_insert)
        line_template_red = makeGaus_cutoff(shifted_r, xaxis_r, height_red, std=std_insert)    

    else:
        print('Enter a valid line type: OII, Lya, or gaussian')

    #---------------------------------------- Injecting emission line -----------------------------------#

    if line_wav_insert * (z+1) < 6000:
        spectra_blue = add_to_spectrum(spectra_blue, line_template_blue)

    elif line_wav_insert * (z+1) >= 6000: 
        spectra_red = add_to_spectrum(spectra_red, line_template_red)
        
        # plt.plot(spectra_red[0].spectral_axis, spectra_red[0].flux)
        # plt.xlim(7300, 7600)
        # plt.show()



    return spectra_blue, spectra_red, line_template_blue, line_template_red




