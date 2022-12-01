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
from scipy.optimize import curve_fit
from itertools import groupby

import emission_lines as line
import spec_func as spec
import plotting as plot
import global_params


# Load the names and wavelength of expected emission lines
wavelength_table = Table.read('Input/line_wavelengths.txt', format='ascii')
global_params.wavelength_ratios_table = Table.read('Input/line_ratios_sort.txt', format='ascii')
global_params.wavelength_ratios_strong_table = Table.read('Input/line_ratios_sort_strong.txt', format='ascii')


def gauss(x, A, mu, sigma, y0):	
	return A * np.exp(-(x-mu)**2 / (2*sigma**2)) + y0

def gauss_no_offset(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)


def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2, y0):
    return gauss_no_offset(x,mu1,sigma1,A1)+gauss_no_offset(x,mu2,sigma2,A2)+y0


def spectrum_range(spectrum, center, width):
    specmax = min(enumerate(spectrum), key=lambda x: abs(x[1]-(center + width)))[0] 
    specmin = min(enumerate(spectrum), key=lambda x: abs(x[1]-(center - width)))[0]
    return specmin, specmax

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def remove_duplicates(difference=4):
    start = None
    def inner(n):
        nonlocal start
        if start is None:
            start = n
        elif abs(start-n) > difference:
            start = n
        return start
    return inner


def closest_value(input_list, input_value):
	try:
		difference = lambda input_list : abs(input_list - input_value)
		res = min(input_list, key=difference)
		return res
	except: # if list is empty
		return False


def fit_gaussian(fit_wav, fit_flux, fit_flux_err, lineloc, linename, plot_bool = False, savename_plot = False, final_run_bool = False):
    """ 
    Fitting gaussians to possible spectral line and determine if they are real or spurious

	Keyword arguments:
	fit_wav (array) -- wavelength of line
	fit_flux (array) -- flux of line
	lineloc (float) -- center wavelength of emission line (Angstrom)
	linename (str) -- name of emission line

	Return:
	lineflux (float) -- flux of fitted emission line
	SNR (float) -- SNR of fitted emission line
	cen_fit (float) -- center of the fit
	real_bool (boolean) -- True if emission line is likely to be real
	spurious_bool (boolean) -- True if emission line is likely to be spurious
    """

    try:
        popt, pcov = curve_fit(gauss, fit_wav, fit_flux, p0=[np.max(fit_flux), lineloc, 5.0, 0.0])
        amp_fit, cen_fit, width_fit, offset_fit = popt

        fitspec_step = (max(fit_wav)-min(fit_wav))/len(fit_wav)
        lineflux = np.sum(fitspec_step * gauss(fit_wav,*popt))-np.sum(offset_fit*(max(fit_wav)-min(fit_wav))) # Take area underneat curve to determine line flux

        # Get region around emission line to determine the scatter
        lowerspecmin, upperspecmax = spectrum_range(fit_wav, cen_fit, 6*abs(width_fit))
        lowerspecmax, upperspecmin = spectrum_range(fit_wav, cen_fit, 3*abs(width_fit))

        # Determine the noise of background
        if len(np.where(fit_flux[lowerspecmin:lowerspecmax] == 0.0)[0]) > len(fit_flux[lowerspecmin:lowerspecmax])/2.: # get only std of red side
            std_cont = np.std(fit_flux[upperspecmin:upperspecmax])
        elif len(np.where(fit_flux[upperspecmin:upperspecmax] == 0.0)[0]) > len(fit_flux[upperspecmin:upperspecmax])/2.: # get only std of blue side
            std_cont = np.std(fit_flux[lowerspecmin:lowerspecmax])
            
        else: # use both sides
            std_cont = (np.std(fit_flux[lowerspecmin:lowerspecmax]) + np.std(fit_flux[upperspecmin:upperspecmax]))/2.

        specmin_SNR = min(enumerate(fit_wav), key=lambda x: abs(x[1]-(cen_fit - 0.5*width_fit)))[0] # Take the 1 sigma inner region of line to determine the SNR
        specmax_SNR = min(enumerate(fit_wav), key=lambda x: abs(x[1]-(cen_fit + 0.5*width_fit)))[0] 

        # skew_param = (np.mean(fit_flux*1e17)-np.median(fit_flux*1e17))/np.std(fit_flux*1e17)
        # print('skew_param', skew_param)

        if std_cont == 0.0 and width_fit < 1: # happens when no data points in std cont region, because width of fit so small 
            return lineflux, np.nan, cen_fit, False, False
        else:
        	#SNR = np.max(gauss(fit_wav,*popt)-offset_fit) / std_cont 
            flux_fwhm = gauss(fit_wav,*popt)[specmin_SNR:specmax_SNR]-offset_fit 
            SNR = np.sqrt(sum(i**2 for i in flux_fwhm)) / std_cont # Maybe calculate SNR differently?

        # Calculate chi squared of fit to pick out spurious lines
        chisq = np.sum((fit_flux[specmin_SNR:specmax_SNR] - gauss(fit_wav,*popt)[specmin_SNR:specmax_SNR]) **2/fit_flux_err[specmin_SNR:specmax_SNR]**2)

        # Check reasonable parameters for gaussian fit
        if np.logical_or(SNR > 5.0, (SNR>4.0) * (final_run_bool == True) * (linename=='OIII-4959')) and np.isinf(SNR) == False and width_fit < 100. and np.logical_or(width_fit > 1.5, (width_fit > 1.0) * (linename == 'OIII-4959')) and amp_fit > 0.0 and lineflux > 0.0 and np.logical_or(np.logical_or(abs(cen_fit-lineloc)<35.*(width_fit>20.0), abs(cen_fit-lineloc)<45.*(width_fit>50.0)), abs(cen_fit-lineloc)<30.): 
            real_bool = True
        else:
            real_bool = False

        # Check no spurious line 
        if real_bool == True: 

            spurious_bool = False

            if cen_fit < min(global_params.wavelength_b)+100 or cen_fit > max(global_params.wavelength_r)-100: # Not on edge of blue or red spectrum
                spurious_bool = True
            elif np.logical_or(abs(cen_fit-5510) < 60, abs(cen_fit-7630) < 90): # Not in one of the gaps 
                spurious_bool = True
            elif abs(cen_fit-lineloc) > 15.0 and SNR < 10.0 and linename != 'OIII-4959': # Do not allow any low SNR lines with a large wavelength offset
                spurious_bool = True
            elif width_fit > 20.0 and SNR < 7.0: # Do not allow any low SNR lines with large widths
                spurious_bool = True
            # elif SNR < 10. and chisq < 0.01 and final_run_bool == False:# Likely spurious/sky line 
            #     spurious_bool = True

        else:
            spurious_bool = np.nan
        
        if plot_bool == True and real_bool == True and spurious_bool == False:
            #print('line, wav, final_bool chisq', linename, lineloc, final_run_bool, chisq)
            plot.plot_emission_line_fit(fit_wav, fit_flux, gauss(fit_wav,*popt), lowerspecmin, lowerspecmax, upperspecmin, upperspecmax, popt, pcov, lineloc, lineflux, SNR, spurious_bool, real_bool, savename_plot)

        return lineflux, SNR, cen_fit, real_bool, spurious_bool
    
    except Exception as e:
        print(e)

        return np.nan, np.nan, np.nan, False, np.nan



def fit_double_gauss(linewav, wavs, flux, ratio_redshift, savename_plot, plot_bool = False):
    """
    Fitting double gaussian for to emission lines

    Keyword arguments:
    flux (array) -- flux of emission line
    savename_plot (str) -- name for saving plot with emission line fit 
    plot_bool (boolean) -- True if want to save plot

    Return:
    linewav (float) -- wavelength of line
    lineflux (float) -- flux of line
    SNR (float) -- SNR of Line

    """

    specmin = min(enumerate(wavs), key=lambda x: abs(x[1]-(linewav-100)))[0]
    specmax = min(enumerate(wavs), key=lambda x: abs(x[1]-(linewav+100)))[0]

    fitspec_step = (max(wavs[specmin:specmax])-min(wavs[specmin:specmax])) / len(wavs[specmin:specmax])

    # Fit single gaussian to get estimate of initial parameters
    popt_one, pcov_one = curve_fit(gauss,wavs[specmin:specmax],flux[specmin:specmax], p0=[np.max(flux[specmin:specmax]), linewav, 5.0, 0.0])

    try:
        popt, pcov = curve_fit(bimodal,wavs[specmin:specmax],flux[specmin:specmax],p0=[popt_one[1]-(popt_one[2]/2), 5.0, np.max(flux[specmin:specmax]), popt_one[1]+(popt_one[2]/1.5), 5.0, np.max(flux[specmin:specmax]), 0.0])

        lineflux = np.sum(fitspec_step * bimodal(wavs[specmin:specmax],*popt))-np.sum(popt[6]*(max(wavs[specmin:specmax])-min(wavs[specmin:specmax])))
        linewav = popt[0] # Take first gaussian peak, could also take average: (popt[0]+popt[3])/2 

        # Determine SNR
        lowerspecmin, upperspecmax =spectrum_range(wavs[specmin:specmax], linewav, 6*abs(popt[1]+popt[4]))
        lowerspecmax, upperspecmin = spectrum_range(wavs[specmin:specmax], linewav, 3*abs(popt[1]+popt[4]))

        std_cont = (np.std(flux[specmin:specmax][lowerspecmin:lowerspecmax]) + np.std(flux[specmin:specmax][upperspecmin:upperspecmax]))/2.
        #SNR = np.max(bimodal(global_params.wavelength_r[specmin:specmax],*popt)-popt[6]) / std_cont # not really right for the double gaussian... 

        specmin_SNR = min(enumerate(wavs[specmin:specmax]), key=lambda x: abs(x[1]-(popt[0] - 0.5*popt[1])))[0] 
        specmax_SNR = min(enumerate(wavs[specmin:specmax]), key=lambda x: abs(x[1]-(popt[3] + 0.5*popt[4])))[0] 

        flux_fwhm = bimodal(wavs[specmin:specmax],*popt)[specmin_SNR:specmax_SNR]-popt[6] 
        SNR = np.sqrt(sum(i**2 for i in flux_fwhm)) / std_cont

        if plot_bool == True:
            plot.plot_emission_line_fit(wavs[specmin:specmax], flux[specmin:specmax], bimodal(wavs[specmin:specmax],*popt), lowerspecmin, lowerspecmax, upperspecmin, upperspecmax, popt, pcov, linewav, lineflux, SNR, np.nan, np.nan, savename_plot)

        return linewav, lineflux, SNR
    except:
        return np.nan, np.nan, np.nan


def single_emission_line(data_blue, data_red, linewav): # WORK IN PROGRESS
    """
    If only one emission line check if its likely OIII, OII or Lya

    Keyword arguments:

    Return:

    """

    # Try fitting double gaussian --> how determine if this is better? 

    # Check skewness of single gaussian fit 
    specmin = min(enumerate(global_params.wavelength_r), key=lambda x: abs(x[1]-(linewav-100)))[0]
    specmax = min(enumerate(global_params.wavelength_r), key=lambda x: abs(x[1]-(linewav+100)))[0]
    fit_flux = data_red[specmin:specmax]
    skew_param = (np.mean(fit_flux*1e17)-np.median(fit_flux*1e17))/np.std(fit_flux*1e17)
    print('skew_param', skew_param)


def finding_emission_lines(data_blue, data_blue_err, data_red, data_red_err, redshift_list, final_run_bool = False, plot_bool = False, savename_plot = False):
    """
    Finding real emission lines by fitting gaussian functions
	If a line is Ha then fit double gaussian function

	Keyword arguments:
	data_blue (array) -- blue spectrum
	data_red (array) -- red spectrum
	redshift_list (list) -- all possible redshifts from ccf solution
	final_run_bool (boolean) -- Default: False. True if final emission line finding loop (if z has been determined)

	Return:
	line_wav_list (list) --  wavelengths of emission lines
	line_flux_list (list) --  fluxes of emission lines
	line_snr_list (list) --  SNR of emission lines

    """

    line_wav_list = []
    line_flux_list = []
    line_snr_list = []

    fit_line_region = 100. # Wavelength region to fit around emission line. Could also use 60. 
    
    for i, z in enumerate(redshift_list): # Loop over all possible redshifts

        # Check possible lines in blue spectrum from text file
        line_wavs_blue, line_names_blue = spec.find_lines(z, 'b')

        for t in range(len(line_wavs_blue)): # Loop over all possible lines
            linewav = line_wavs_blue[t]*(z+1)

            if int(linewav) not in np.array(line_wav_list).astype(int):

                if line_names_blue[t] == 'OIII-4959' and final_run_bool == True: # use a narrow wavelength region for these two emission lines
                    fit_line_region = 30.0
                elif line_names_blue[t] == 'NeIII-3868' and final_run_bool == True:
                    fit_line_region = 50.0
                else:
                    fit_line_region = 100.
                # if line_names_blue[t] == 'CIV-1549':
                #     fit_line_region = 150.0

                # Prepare part of the spectrum to be fitted
                if line_names_blue[t] == 'OIII-4959' and z < 0.85: # Because too close to other lines
                    specmin = min(enumerate(global_params.wavelength_b), key=lambda x: abs(x[1]-(linewav-fit_line_region)))[0]
                    specmax = min(enumerate(global_params.wavelength_b), key=lambda x: abs(x[1]-(linewav+fit_line_region-20)))[0]
                else:
                    specmin = min(enumerate(global_params.wavelength_b), key=lambda x: abs(x[1]-(linewav-fit_line_region)))[0]
                    specmax = min(enumerate(global_params.wavelength_b), key=lambda x: abs(x[1]-(linewav+fit_line_region)))[0]

                # Fit gaussian and check if line is real
                lineflux, SNR, cen_fit, real_bool, spurious_bool = fit_gaussian(global_params.wavelength_b[specmin:specmax], data_blue[specmin:specmax], data_blue_err[specmin:specmax], linewav, line_names_blue[t], plot_bool, savename_plot+line_names_blue[t]+'_'+str(np.int(linewav))+'.pdf', final_run_bool)               

                if real_bool == True and spurious_bool == False and np.logical_or(len(line_wav_list) ==0, abs(cen_fit-closest_value(line_wav_list,cen_fit))>10.0): # Save lines if they seem real and not already in the list
                    line_wav_list.append(cen_fit)
                    line_flux_list.append(lineflux)
                    line_snr_list.append(SNR)

        # Check possible lines in red spectrum from text file
        line_wavs_red, line_names_red = spec.find_lines(z, 'r')

        for t in range(len(line_wavs_red)):
            linewav = line_wavs_red[t]*(z+1)

            if int(linewav) not in np.array(line_wav_list).astype(int):

                if line_names_red[t] == 'OIII-4959' and final_run_bool == True:
                    fit_line_region = 30.0
                elif line_names_red[t] == 'NeIII-3868' and final_run_bool == True:
                    fit_line_region = 50.0
                else:
                    fit_line_region = 100.

                specmin = min(enumerate(global_params.wavelength_r), key=lambda x: abs(x[1]-(linewav-fit_line_region)))[0]
                specmax = min(enumerate(global_params.wavelength_r), key=lambda x: abs(x[1]-(linewav+fit_line_region)))[0]

                 # Fit gaussian and check if line is real
                lineflux, SNR, cen_fit, real_bool, spurious_bool = fit_gaussian(global_params.wavelength_r[specmin:specmax], data_red[specmin:specmax], data_red_err[specmin:specmax], linewav, line_names_red[t], plot_bool, savename_plot+line_names_red[t]+'_'+str(np.int(linewav))+'.pdf', final_run_bool)               
                
                if real_bool == True and spurious_bool == False and np.logical_or(len(line_wav_list) == 0, abs(cen_fit-closest_value(line_wav_list,cen_fit))>10.0): # Save lines if they seem real 
                    line_wav_list.append(cen_fit)
                    line_flux_list.append(lineflux)
                    line_snr_list.append(SNR)

    return line_wav_list, line_flux_list, line_snr_list



def line_ratios(line_wavs):
    """ 
    Take ratios of all detected emission lines and determine best matching redshift 
	
	Keyword arguments:
	line_wavs (list) -- list with wavelengths of emission lines

	Return:
	ratio_redshift (float) -- most likely redshift from taking the ratios of the wavelengths of the emission lines
    """

    z_real_lines = [] # Array with all possible redshifts from emission lines ratios 

    for h in range(len(line_wavs)): # double loop over all emission line wavelengths
        for o in range(len(line_wavs)):

            ratio = line_wavs[h]/line_wavs[o]
            if ratio < 0.999 or ratio > 1.001:

                index = find_nearest(global_params.wavelength_ratios_strong_table['ratio'], ratio)

                if abs(global_params.wavelength_ratios_strong_table['ratio'][index] - ratio) < 0.006:  # First check if ratio corresponds to any strong line ratio
                    wavs_rest = np.array([global_params.wavelength_ratios_strong_table['wav_num'][index], global_params.wavelength_ratios_strong_table['wav_denom'][index]])
                    wavs_shifted = np.array([line_wavs[h], line_wavs[o]])
                    z_ratio = np.max(wavs_shifted)/np.max(wavs_rest) - 1

                    if z_ratio > 0.0:
                        z_real_lines.append(z_ratio)

                else: # If the line ratios do not correspond to any of the strong line ratios, check the more extensive list
                    
                    index = find_nearest(global_params.wavelength_ratios_table['ratio'], ratio)

                    if abs(global_params.wavelength_ratios_table['ratio'][index] - ratio) < 0.006: 
                        wavs_rest = np.array([global_params.wavelength_ratios_table['wav_num'][index], global_params.wavelength_ratios_table['wav_denom'][index]])
                        wavs_shifted = np.array([line_wavs[h], line_wavs[o]])
                        z_ratio = np.max(wavs_shifted)/np.max(wavs_rest) - 1

                        if z_ratio > 0.0:
                            z_real_lines.append(z_ratio)

    # Pick the best redshift corresponding to the line ratios
    if len(z_real_lines) > 0: 
        z_real_lines_pick = np.round(np.array(z_real_lines),2) # rounding here also affects the accuracy! 
        z_real_lines_pick = list(z_real_lines_pick)

        z_real_lines_unique = np.unique(z_real_lines_pick)
        z_real_lines_sum_list = []

        for r in range(len(z_real_lines_unique)):
            z_real_lines_sum_list.append(len(np.where(np.array(z_real_lines_pick)==z_real_lines_unique[r])[0]))

        res = max(set(z_real_lines_pick), key = z_real_lines_pick.count)

        arg_res = np.where(z_real_lines_pick == res)[0]
        ratio_redshift = np.mean(np.array(z_real_lines)[arg_res]) # final redshift is average of all the best solutions 

        diff_res = np.array(z_real_lines_sum_list) - np.max(z_real_lines_sum_list)
        idx_alternative = np.where(abs(diff_res) < 5)[0]
        idx_alternative = np.where((abs(diff_res) < 5)&(diff_res != 0))[0]

        if len(idx_alternative) > 0: # Check if the top two solutions only 4 points apart --> then pick the solution with the most strong lines if there are many strong lines
          z_real_alternative = z_real_lines_unique[idx_alternative[0]]
          if res > z_real_alternative and np.array(z_real_lines_sum_list)[idx_alternative[0]] > 150 and len(np.unique(z_real_lines))>4:
              res = z_real_alternative
              arg_res = z_real_lines_pick.index(res)
              ratio_redshift = z_real_lines[arg_res]

    else:
        ratio_redshift = np.nan # single line

    return ratio_redshift




