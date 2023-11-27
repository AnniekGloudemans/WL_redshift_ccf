import numpy as np
import matplotlib.pyplot as plt
import time
from astropy import units as u

import spec_func as spec
trunc = lambda f: f - f % 0.01
import traceback
from specutils.analysis import template_logwl_resample 
from specutils.analysis import correlation
from specutils import Spectrum1D
import plotting as plot
import pdb

initial_redshifts_fitting = np.linspace(0.0,6.8,69)#6.8,69)
gal_zip = spec.prepare_template(key='AGN')


def ccf_algorithm(z, spec_b, spec_r):
    """
    Determine cross correlation values between data and template

    Keyword arguments:
    z (float) -- redshift to test
    spec_(b/r) -- rebinned blue/red spectrum

    Return:
    z_lag_(r/b) (float) -- ccf offset
    r_(blue/red) (float) -- strenght of correlation

    """

    tem_r = spec.shift_template(gal_zip, np.arange(0, len(spec_r.spectral_axis.value), 1), z, 'r', np.min(spec_b.spectral_axis.value), np.max(spec_b.spectral_axis.value), np.min(spec_r.spectral_axis.value), np.max(spec_r.spectral_axis.value))
    tem_b = spec.shift_template(gal_zip, np.arange(0, len(spec_b.spectral_axis.value), 1), z, 'b', np.min(spec_b.spectral_axis.value), np.max(spec_b.spectral_axis.value), np.min(spec_r.spectral_axis.value), np.max(spec_r.spectral_axis.value))


    if z < 0.0: # Don't try to correlate if redshift is negative
        return np.nan, np.nan, np.nan, np.nan

    spec_tem_r = Spectrum1D(spectral_axis=spec_r.spectral_axis, flux=tem_r * u.Unit('erg cm-2 s-1 AA-1')) #
    spec_tem_b = Spectrum1D(spectral_axis=spec_b.spectral_axis, flux=tem_b * u.Unit('erg cm-2 s-1 AA-1')) #

    # Rebin spectrum and template onto common log-scale
    rebin_val_b, rebin_val_temb = template_logwl_resample(spec_b, spec_tem_b)
    rebin_val_r, rebin_val_temr = template_logwl_resample(spec_r, spec_tem_r)

    # # Cross-correlate with template
    # ccf_b, lag = correlation.template_correlate(rebin_val_b, rebin_val_temb) 
    # ccf_r, lag = correlation.template_correlate(rebin_val_r, rebin_val_temr)

    # lag_r = (np.argmax(ccf_r) - ((len(ccf_r) + 0) / 2)) # Change this? 
    # lag_b = (np.argmax(ccf_b) - ((len(ccf_b) + 0) / 2))
    # print('ccf_b, lag_b', ccf_b, lag_b)
    # print('ccf_r, lag_r', ccf_r, lag_r)

    # Numpy faster than correlation.template_correlate: 
    ccf_b = np.correlate(rebin_val_b.flux.value, rebin_val_temb.flux.value, 'full') 
    ccf_r = np.correlate(rebin_val_r.flux.value, rebin_val_temr.flux.value, 'full')

    lag_r = (np.argmax(ccf_r) - ((len(ccf_r) + 0) / 2))
    lag_b = (np.argmax(ccf_b) - ((len(ccf_b) + 0) / 2))

    r_red = ((np.ndarray.max(ccf_r) * np.ndarray.mean(ccf_r)) / np.std(ccf_r)) * 1e32
    r_blue = ((np.ndarray.max(ccf_b) * np.ndarray.mean(ccf_b)) / np.std(ccf_b)) * 1e32

    pix_width_r = (np.log(max(rebin_val_r.spectral_axis.value)) - np.log(min(rebin_val_r.spectral_axis.value)))/len(rebin_val_r.spectral_axis)
    pix_width_b = (np.log(max(rebin_val_b.spectral_axis.value)) - np.log(min(rebin_val_b.spectral_axis.value)))/len(rebin_val_b.spectral_axis)


    z_lag_r = round((10 ** (pix_width_r * lag_r) * (1 + (30 / 3e5)) - 1), 5)
    z_lag_b = round((10 ** (pix_width_b * lag_b) * (1 + (30 / 3e5)) - 1), 5)

    return z_lag_r, r_blue, r_red, z_lag_b


def ccf_function(spec_blue, spec_red):
    """
    Run CCF algorithm to find spectroscopic redshift

	Keyword arguments:
	spec_(blue/red) -- blue and red spectrum

	Return:
	final_redshifts (list) -- best fitting redshifts 
	z_lag_list (list) -- ccf values for each redshift (smaller corresponds to better correlation)

    """

    final_redshifts = []
    z_lag_list = []
    r_list = []
    z_guess_list1 = []
    z_guess_list2 = []
    z_guess_list3 = []
    z_guess_list4 = []
    z_guess_list1_blue = []
    z_guess_list2_blue = []
    z_guess_list3_blue = []
    z_guess_list4_blue = [] 

    print_bool = False 

    ########### Round 1 ##############
    # In round 1 the standard initial redshifts are used

    for i, z in enumerate(initial_redshifts_fitting): 

        z_lag_r, r_blue, r_red, z_lag_b = ccf_algorithm(z, spec_blue, spec_red)

        if(z_lag_r >= -0.1 and z_lag_r<=0.1):
            z_guess_list1.append((z, z_lag_r))

        if (z_lag_b >= -0.1 and z_lag_b<=0.1):
            z_guess_list1_blue.append((z, z_lag_b))

        if print_bool == True:
            print('round 1', z, z_lag_r, r_red, z_lag_b, r_blue)


    ########### Round 2 ##############

    count = 0
    for trial_z, trialz_lag_r in z_guess_list1: 
        z = trial_z + trialz_lag_r
           
        if z <= 0:
            continue
        z_lag_r, r_blue, r_red , z_lag_b = ccf_algorithm(z, spec_blue, spec_red)

        if (z_lag_r > -0.05 and z_lag_r <= 0.05): # -0.01 - 0.09 or +-0.03?
            z_guess_list2.append((z, z_lag_r))

        if print_bool == True:
            print('round 2 z, zlag r_red', z, z_lag_r, r_red, z_lag_b, r_blue)

        count += 1

    count = 0
    for trial_z, trialz_lag_b in z_guess_list1_blue: 
        z = trial_z + trialz_lag_b
         
        if z <= 0:
            continue
        z_lag_r, r_blue, r_red , z_lag_b = ccf_algorithm(z, spec_blue, spec_red)

        if (z_lag_b > -0.05 and z_lag_b <= 0.05): 
            z_guess_list2_blue.append((z, z_lag_b))

        if print_bool == True:
            print('round 2  BLUE z, zlag r_red, r_blue', z, z_lag_r, r_red, z_lag_b, r_blue)

        count += 1


    ########### Round 3 ##############

    for trial_z, trialz_lag_r in z_guess_list2:
        z = trial_z + trialz_lag_r

        if z <= 0:
            continue
        z_lag_r, r_blue, r_red , z_lag_b = ccf_algorithm(z, spec_blue, spec_red)
        if print_bool == True:
            print('round 3 z, zlag r_red', z, z_lag_r, r_red, z_lag_b, r_blue)
        if (z_lag_r > -0.015 and z_lag_r <= 0.015):  # 0.012 or -/+0.005?
            z_guess_list3.append((z, z_lag_r))

    for trial_z, trialz_lag_b in z_guess_list2_blue:
        z = trial_z + trialz_lag_b

        if z <= 0:
            continue
        z_lag_r, r_blue, r_red , z_lag_b = ccf_algorithm(z, spec_blue, spec_red)
        if print_bool == True:
            print('round 3 BLUE z, zlag r_red', z, z_lag_r, r_red, z_lag_b, r_blue)
        if (z_lag_b > -0.015 and z_lag_b <= 0.015): #-0.01, #0.012): 
            z_guess_list3_blue.append((z, z_lag_b))


    ########### Round 4 ##############

    for trial_z, trialz_lag_r in z_guess_list3:
        z = trial_z + trialz_lag_r

        if z <= 0:
            continue
        z_lag_r = 999
        count = 0

        while (z_lag_r > 0.001 or z_lag_r < -0.001 and count < 8): # count was 8
            count += 1
            z_lag_r, r_blue, r_red, z_lag_b = ccf_algorithm(z, spec_blue, spec_red)
            z = z+z_lag_r
            if print_bool == True:
                print('round 4 z, zlag, count', z, z_lag_r, z_lag_b, count)
        if count < 8: 
            est_redshift = z - z_lag_r
            final_redshifts.append((round(est_redshift, 5)))
            z_lag_list.append(z_lag_r)
            r_list.append(r_red)

    for trial_z, trialz_lag_b in z_guess_list3_blue:
        z = trial_z + trialz_lag_b

        if z <= 0:
            continue
        z_lag_b = 999
        count = 0

        while (z_lag_b > 0.001 or z_lag_b < -0.001 and count <8): 
            count += 1
            z_lag_r, r_blue, r_red, z_lag_b = ccf_algorithm(z, spec_blue, spec_red)
            z = z + z_lag_b
            if print_bool == True:
                print('round 4 BLUE z, zlag, count', z, z_lag_r, z_lag_b, count)
        if count < 8: 
            est_redshift = z - z_lag_b
            final_redshifts.append((round(est_redshift, 5)))
            z_lag_list.append(z_lag_b)
            r_list.append(r_blue)

    # Remove any possible nan values 
    final_redshifts = np.array(final_redshifts)[~np.isnan(z_lag_list)]
    z_lag_list = np.array(z_lag_list)[~np.isnan(z_lag_list)]

    return final_redshifts, z_lag_list


def ccf_test(true_z, data_blue_test, data_red_test, save_fig, save_name):
    """
    Test ccf algorithm with real redshift and plot results

    """

    # # Rebin the spectra
    # log_wvlngth_b, rebin_val_b, rebin_ivar_b = spec.rebin(global_params.wavelength_b, data_blue_test[0], std_blue_test[0], 'blue')
    # log_wvlngth_r, rebin_val_r, rebin_ivar_r = spec.rebin(global_params.wavelength_r, data_red_test[0], std_red_test[0], 'red') 

    ccf_r_solutions = []
    ccf_b_solutions = []
    ccf_r_all = []
    ccf_b_all = []

    for i, z in enumerate(initial_redshifts_fitting): 

        # already rebinned the data in advance
        z_lag_r, r_blue, r_red, z_lag_b = ccf_algorithm(z, data_blue_test[0], data_red_test[0])#ccf_algorithm(z, rebin_val_b, rebin_val_r)
        ccf_r_all.append([z, z_lag_r, r_red])
        ccf_b_all.append([z, z_lag_b, r_blue])

        if(z_lag_r >= 0.0 and z_lag_r<=0.2):
            ccf_r_solutions.append((z, z_lag_r, r_red))

        if(z_lag_r >= -0.07 and z_lag_r<=0.2):
            ccf_r_solutions.append((z, z_lag_r, r_red))

        if (z_lag_b >= 0.0 and z_lag_b<=0.2):
            ccf_b_solutions.append((z, z_lag_b, r_blue))

        if(z_lag_r >= -0.005 and z_lag_r<=0.05):
            ccf_r_solutions.append((z, z_lag_r, r_red)) 

        if(z_lag_b >= -0.005 and z_lag_b<=0.05):
            ccf_b_solutions.append((z, z_lag_b, r_blue)) 

        # if print_bool == True:
        #     print('round 1', z, z_lag_r, r_red, z_lag_b, r_blue)

    z_lag_r, r_blue, r_red, z_lag_b = ccf_algorithm(true_z, data_blue_test[0], data_red_test[0]) # Test ccf for true redshift
    print("CCF algo result true z", z_lag_r, r_blue, r_red, z_lag_b)

    ccf_r_all = np.array(ccf_r_all)
    ccf_b_all = np.array(ccf_b_all)

    # How shifted template looks like at true redshift
    tem_r = spec.shift_template(gal_zip, np.arange(0, len(data_red_test.spectral_axis.value), 1), true_z, 'r', np.min(data_blue_test.spectral_axis.value), np.max(data_blue_test.spectral_axis.value), np.min(data_red_test.spectral_axis.value), np.max(data_red_test.spectral_axis.value)) 
    tem_b = spec.shift_template(gal_zip, np.arange(0, len(data_blue_test.spectral_axis.value), 1), true_z, 'b', np.min(data_blue_test.spectral_axis.value), np.max(data_blue_test.spectral_axis.value), np.min(data_red_test.spectral_axis.value), np.max(data_red_test.spectral_axis.value))

    if save_fig == True:
        plot.ccf_test_plot(data_blue_test[0], data_red_test[0], tem_r, tem_b, ccf_b_all, ccf_r_all, r_blue, r_red, z_lag_b, z_lag_r, true_z, save_name)



