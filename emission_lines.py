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


def fit_gaussian(fit_wav, fit_flux, lineloc, linename, plot_bool = False, savename_plot = False):
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
        if len(np.where(fit_wav[lowerspecmin:lowerspecmax] == 0.0)[0]) > len(fit_wav[lowerspecmin:lowerspecmax])/2.:
            std_cont = np.std(fit_flux[upperspecmin:upperspecmax])
        elif len(np.where(fit_wav[upperspecmin:upperspecmax] == 0.0)[0]) > len(fit_wav[upperspecmin:upperspecmax])/2.:
            std_cont = np.std(fit_flux[lowerspecmin:lowerspecmax])
        else:
            std_cont = (np.std(fit_flux[lowerspecmin:lowerspecmax]) + np.std(fit_flux[upperspecmin:upperspecmax]))/2.


        specmin_SNR = min(enumerate(fit_wav), key=lambda x: abs(x[1]-(cen_fit - 0.5*width_fit)))[0] # Take the 1 sigma inner region of line to determine the SNR
        specmax_SNR = min(enumerate(fit_wav), key=lambda x: abs(x[1]-(cen_fit + 0.5*width_fit)))[0] 

        flux_fwhm = gauss(fit_wav,*popt)[specmin_SNR:specmax_SNR]-offset_fit  # DOES THIS WORK??

        if std_cont == 0.0 and width_fit < 1: # happens when no data points in std cont region, because width of fit so small 
            return lineflux, np.nan, cen_fit, False, False
        else:
        	#SNR = np.max(gauss(fit_wav,*popt)-offset_fit) / std_cont 
            SNR = np.sqrt(sum(i**2 for i in flux_fwhm)) / std_cont # IS THIS RIGHT?


    	# Check if real or spurious line
        if SNR > 4.0 and width_fit < 100 and np.logical_or(width_fit > 1.5, (width_fit > 1.0) * (linename == 'OIII-4959')) and amp_fit > 0.0 and np.logical_or(np.logical_or(abs(cen_fit-lineloc)<35.*(width_fit>20.0), abs(cen_fit-lineloc)<45.*(width_fit>50.0)*(linename == 'CIV-1549')), abs(cen_fit-lineloc)<30.) and cen_fit > 3800 and cen_fit < 9500: # change to 10?
            real_bool = True
        else:
            real_bool = False

        lowerspecmin_zero, upperspecmax_zero =spectrum_range(fit_wav, cen_fit, 3.5*abs(width_fit)) # specify line region 
        lowerspecmax_zero, upperspecmin_zero = spectrum_range(fit_wav, cen_fit, 1*abs(width_fit))

        if np.isnan(std_cont) == True: # to avoid runtime warnings
            flux_below_2sigma = 0
            flux_above_2sigma = 0
        else:
            flux_below_2sigma = len(np.where(fit_flux[lowerspecmax_zero:upperspecmin_zero] < np.max(gauss(fit_flux,*popt))-2*std_cont)[0])
            flux_above_2sigma = len(np.where(fit_flux[lowerspecmax_zero:upperspecmin_zero] < np.max(gauss(fit_flux,*popt))+2*std_cont)[0])


       	# Adjust this to real data! 
        if real_bool == True:
            if np.logical_or(abs(cen_fit-5510) < 50, abs(cen_fit-7630) < 90): # Or only keep this one 
                spurious_bool = True

            elif width_fit > 100 or width_fit < 1.0 or amp_fit < 0.0 or abs(cen_fit-lineloc)>35. or np.isinf(SNR) == True:
                spurious_bool = True

            elif len(np.where(fit_flux[lowerspecmin_zero:lowerspecmax_zero] == 0.0)[0]) > len(fit_flux[lowerspecmin_zero:lowerspecmax_zero])/2.1 and abs(cen_fit-lineloc) > 0.1:
                spurious_bool = True
                if SNR > 30 and np.logical_or((cen_fit < 7705)*(cen_fit > 7675), (cen_fit < 5570)*(cen_fit > 5540)):
                    spurious_bool = False

            elif len(np.where(fit_flux[upperspecmin_zero:upperspecmax_zero] == 0.0)[0]) > len(fit_flux[upperspecmin_zero:upperspecmax_zero])/2.1 and abs(cen_fit-lineloc) > 0.1: # does this still take out the false lines?
                spurious_bool = True

            elif len(np.where(fit_flux[lowerspecmax_zero:upperspecmin_zero] < offset_fit-2*std_cont)[0]) > len(fit_flux[upperspecmin_zero:upperspecmax_zero])/3.:
                spurious_bool = True

            elif flux_below_2sigma > len(fit_flux[upperspecmin_zero:upperspecmax_zero])/4. and len(fit_flux[lowerspecmin_zero:upperspecmax_zero]) < 100. and SNR < 10.0:
                spurious_bool = True

            elif flux_below_2sigma+flux_above_2sigma > len(fit_flux[upperspecmin_zero:upperspecmax_zero])/4. and len(fit_flux[lowerspecmin_zero:upperspecmax_zero]) < 150. and SNR < 10.0 and linename != 'OIII-4959' and np.logical_or(width_fit < 3.0, width_fit > 10.0): # does this not mess up other right ones?
                spurious_bool = True

            # elif width_fit > 30.0 and z < 1.0 and SNR < 10.0: 
            #     spurious_bool = True

            elif width_fit > 25.0 and SNR < 11.0 and flux_below_2sigma+flux_above_2sigma > 10: 
                spurious_bool = True

            elif flux_below_2sigma+flux_above_2sigma > len(fit_flux[lowerspecmax_zero:upperspecmin_zero])/2. and np.logical_or((SNR < 7.0)*(width_fit < 10.0),(SNR < 8.0)*(width_fit < 5.0)):
                spurious_bool = True

            elif width_fit > 20.0 and SNR < 8.0 and flux_below_2sigma+flux_above_2sigma > 10: 
                spurious_bool = True

            elif width_fit > 30.0 and cen_fit > 9300:
                spurious_bool = True

            elif width_fit > 35.0 and SNR < 10.0:
                spurious_bool = True

            elif width_fit > 30.0 and SNR < 8.0: 
                spurious_bool = True

            elif width_fit > 60.0 and SNR < 15.5:
                spurious_bool = True

            elif width_fit > 90.0 and SNR < 30.0: # Do wider lines ever exist?
                spurious_bool = True

            elif width_fit > 50.0 and SNR < 20. and np.logical_or(abs(cen_fit-5500) < 150, abs(cen_fit-7600) < 150):
                spurious_bool = True

            elif abs(cen_fit-lineloc) > 15.0 and SNR < 10.0 and width_fit < 15.0 and linename != 'OIII-4959': 
                spurious_bool = True

            elif abs(cen_fit-lineloc) > 18.0 and SNR < 10.0 and linename != 'OIII-4959': 
                spurious_bool = True

            elif abs(cen_fit-lineloc) > 20.0 and SNR < 15.0: 
                spurious_bool = True

            elif cen_fit > 9300 and SNR < 10.0:
                spurious_bool = True

            elif cen_fit > 6000 and SNR < 10. and max(fit_wav) < 7000:
                spurious_bool = True

            elif width_fit > 35. and len(np.where(fit_flux[lowerspecmax_zero:upperspecmin_zero] == 0.0)[0]) > 20.:
                spurious_bool = True

            elif len(fit_flux[lowerspecmin_zero:lowerspecmax_zero]) < 30 and SNR < 10.0:
                spurious_bool = True

            elif width_fit > 35. and len(np.where(fit_flux[lowerspecmax_zero:upperspecmin_zero] < offset_fit-2*std_cont)[0]) > 5 and SNR < 15:
                spurious_bool = True

            elif linename == 'NII-6548' and abs(cen_fit-lineloc) > 12.0:
                spurious_bool = True  

            elif linename == 'NII-6583' and abs(cen_fit-lineloc) > 12.0:
                spurious_bool = True 

            # elif linename == 'OIII-4959' and z < 0.85 and width_fit > 20.0 and SNR < 10.:
            #     spurious_bool = True  

            elif len(np.where(fit_flux[lowerspecmax_zero:upperspecmin_zero] > offset_fit+2*std_cont)[0])+len(np.where(fit_flux[lowerspecmax_zero:upperspecmin_zero] < offset_fit-2*std_cont)[0]) > len(fit_flux[upperspecmin_zero:upperspecmax_zero])/6. and SNR < 10. and linename != 'OIII-4959' and np.logical_or(width_fit < 3.0, width_fit > 10.0):
                spurious_bool = True

            elif flux_below_2sigma+flux_above_2sigma > 20 and width_fit > 80 and SNR <20:
                spurious_bool = True

            elif np.count_nonzero(fit_flux[specmin_SNR:specmax_SNR]) == 0:
                spurious_bool = True

            try:
                test = spurious_bool
            except:
                spurious_bool = False # set to false if not known yet

        else:
            spurious_bool = np.nan

        

        if plot_bool == True and real_bool == True: #and spurious_bool == True:
        	plot.plot_emission_line_fit(fit_wav, fit_flux, gauss(fit_wav,*popt), lowerspecmin, lowerspecmax, upperspecmin, upperspecmax, popt, pcov, lineloc, lineflux, SNR, spurious_bool, real_bool, savename_plot)


        return lineflux, SNR, cen_fit, real_bool, spurious_bool
    
    except Exception as e:
        print(e)

        return np.nan, np.nan, np.nan, False, np.nan



def fit_Ha_double_gauss(flux, ratio_redshift, savename_plot, plot_bool = False):
    """
    Fitting double gaussian for Ha emission lines

    Keyword arguments:
    flux (array) -- flux of emission line
    ratio_redshift (float) -- expected redshift
    savename_plot (str) -- name for saving plot with emission line fit 
    plot_bool (boolean) -- True if want to save plot

    Return:
    Ha_linewav (float) -- wavelength of Ha line
    Ha_lineflux (float) -- flux of Ha line
    SNR (float) -- SNR of Ha Line

    """
    Ha_linewav = (1+ratio_redshift)*6563.

    specmin = min(enumerate(global_params.wavelength_r), key=lambda x: abs(x[1]-(Ha_linewav-100)))[0]
    specmax = min(enumerate(global_params.wavelength_r), key=lambda x: abs(x[1]-(Ha_linewav+100)))[0]

    fitspec_step = (max(global_params.wavelength_r[specmin:specmax])-min(global_params.wavelength_r[specmin:specmax])) / len(global_params.wavelength_r[specmin:specmax])

    # Fit single gaussian to get estimate of initial parameters
    popt_one, pcov_one = curve_fit(gauss,global_params.wavelength_r[specmin:specmax],flux[specmin:specmax], p0=[np.max(flux[specmin:specmax]), Ha_linewav, 5.0, 0.0])

    try:
        popt, pcov = curve_fit(bimodal,global_params.wavelength_r[specmin:specmax],flux[specmin:specmax],p0=[popt_one[1]-(popt_one[2]/2), 5.0, np.max(flux[specmin:specmax]), popt_one[1]+(popt_one[2]/1.5), 5.0, np.max(flux[specmin:specmax]), 0.0])

        Ha_lineflux = np.sum(fitspec_step * bimodal(global_params.wavelength_r[specmin:specmax],*popt))-np.sum(popt[6]*(max(global_params.wavelength_r[specmin:specmax])-min(global_params.wavelength_r[specmin:specmax])))
        Ha_linewav = (popt[0]+popt[3])/2 # take middle of the two gaussians?

        # Determine SNR
        lowerspecmin, upperspecmax =spectrum_range(global_params.wavelength_r[specmin:specmax], Ha_linewav, 6*abs(popt[1]+popt[4]))
        lowerspecmax, upperspecmin = spectrum_range(global_params.wavelength_r[specmin:specmax], Ha_linewav, 3*abs(popt[1]+popt[4]))

        std_cont = (np.std(flux[specmin:specmax][lowerspecmin:lowerspecmax]) + np.std(flux[specmin:specmax][upperspecmin:upperspecmax]))/2.
        #SNR = np.max(bimodal(global_params.wavelength_r[specmin:specmax],*popt)-popt[6]) / std_cont # not really right for the double gaussian... 

        specmin_SNR = min(enumerate(global_params.wavelength_r[specmin:specmax]), key=lambda x: abs(x[1]-(popt[0] - 0.5*popt[1])))[0] 
        specmax_SNR = min(enumerate(global_params.wavelength_r[specmin:specmax]), key=lambda x: abs(x[1]-(popt[3] + 0.5*popt[4])))[0] 

        flux_fwhm = bimodal(global_params.wavelength_r[specmin:specmax],*popt)[specmin_SNR:specmax_SNR]-popt[6] 
        SNR = np.sqrt(sum(i**2 for i in flux_fwhm)) / std_cont

        if plot_bool == True:
            plot.plot_emission_line_fit(global_params.wavelength_r[specmin:specmax], flux[specmin:specmax], bimodal(global_params.wavelength_r[specmin:specmax],*popt), lowerspecmin, lowerspecmax, upperspecmin, upperspecmax, popt, pcov, Ha_linewav, Ha_lineflux, SNR, np.nan, np.nan, savename_plot)

        return Ha_linewav, Ha_lineflux, SNR
    except:
        return np.nan, np.nan, np.nan


def finding_emission_lines(data_blue, data_red, redshift_list, final_run_bool = False, plot_bool = False, savename_plot = False):
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
                    fit_line_region = 50.0
                if line_names_blue[t] == 'NeIII-3868' and final_run_bool == True:
                    fit_line_region = 50.0
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
                lineflux, SNR, cen_fit, real_bool, spurious_bool = fit_gaussian(global_params.wavelength_b[specmin:specmax], data_blue[specmin:specmax], linewav, line_names_blue[t], plot_bool, savename_plot+line_names_blue[t]+'_'+str(np.int(linewav))+'.pdf')               
                #print('lineflux, SNR, cen_fit, real_bool, spurious_bool', lineflux, SNR, cen_fit, real_bool, spurious_bool)
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
                    fit_line_region = 50.0
                if line_names_red[t] == 'NeIII-3868' and final_run_bool == True:
                    fit_line_region = 50.0

                specmin = min(enumerate(global_params.wavelength_r), key=lambda x: abs(x[1]-(linewav-fit_line_region)))[0]
                specmax = min(enumerate(global_params.wavelength_r), key=lambda x: abs(x[1]-(linewav+fit_line_region)))[0]

                 # Fit gaussian and check if line is real
                lineflux, SNR, cen_fit, real_bool, spurious_bool = fit_gaussian(global_params.wavelength_r[specmin:specmax], data_red[specmin:specmax], linewav, line_names_red[t], plot_bool, savename_plot+line_names_red[t]+'_'+str(np.int(linewav))+'.pdf')               

                if real_bool == True and spurious_bool == False and np.logical_or(len(line_wav_list) ==0, abs(cen_fit-closest_value(line_wav_list,cen_fit))>10.0): # Save lines if they seem real 
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

    z_real_lines = []

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

    # Pick the best redshift corresponding to the found line ratios
    if len(z_real_lines) > 0: 
        z_real_lines_pick = np.round(np.array(z_real_lines),2) 
        z_real_lines_pick = list(z_real_lines_pick)

        z_real_lines_unique = np.unique(z_real_lines_pick)
        z_real_lines_sum_list = []

        for r in range(len(z_real_lines_unique)):
          z_real_lines_sum_list.append(len(np.where(np.array(z_real_lines_pick)==z_real_lines_unique[r])[0]))

        if len(z_real_lines_sum_list) == 2 and z_real_lines_sum_list[0] == z_real_lines_sum_list[1]:
            res = np.max(z_real_lines_pick) # pick the highest redshift if only two lines and same number of instances?
            arg_res = np.argmax(z_real_lines_pick)
        else:
            res = max(set(z_real_lines_pick), key = z_real_lines_pick.count)
            arg_res = z_real_lines_pick.index(res)

        diff_res = np.array(z_real_lines_sum_list) - np.max(z_real_lines_sum_list)
        idx_alternative = np.where(abs(diff_res) < 5)[0]
        idx_alternative = np.where((abs(diff_res) < 5)&(diff_res != 0))[0]

        if len(idx_alternative) > 0: # check if the top two solutions only 4 points apart --> then pick the solution with the most strong lines if there are many strong lines
          z_real_alternative = z_real_lines_unique[idx_alternative[0]]
          if res > z_real_alternative and np.array(z_real_lines_sum_list)[idx_alternative[0]] > 150 and len(np.unique(z_real_lines))>4: # CHECK!! z_real_lines here used to be real_line_wav_list
              res = z_real_alternative
              arg_res = z_real_lines_pick.index(res)
        
        ratio_redshift = z_real_lines[arg_res]

    else:
        ratio_redshift = np.nan # single line

    return ratio_redshift



