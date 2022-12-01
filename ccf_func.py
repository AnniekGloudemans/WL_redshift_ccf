
import numpy as np
import matplotlib.pyplot as plt
import time
import pdb
from astropy import units as u
import global_params
import spec_func as spec
trunc = lambda f: f - f % 0.01
import traceback

def ccf_algorithm(z, rebin_val_b, rebin_val_r, rebin_val_temb = False, rebin_val_temr = False):
    """
    Determine cross correlation values between data and template

	Keyword arguments:
	z (float) -- redshift to test
	rebin_val_(b/r) -- rebinned spectrum
	rebin_val_tem(b/r) -- rebinned template if provided 

	Return:
	z_lag_(r/b) (float) -- ccf value ()
	r_(blue/red) (float) -- strenght of correlation (?)

    """
    try:
        if rebin_val_temb == False: # Only shift and rebin a template if one doesn't already exist

		    # Shifting template
            tem_r = spec.shift_template(global_params.gal_zip, global_params.xaxis_r, z, 'r') # Should you even do this also for the first already rebinned ones?? 
            tem_b = spec.shift_template(global_params.gal_zip, global_params.xaxis_b, z, 'b')

			# Rebin template to log bins
            log_wvlngth_temb, rebin_val_temb, rebin_ivar_temb = spec.rebin(global_params.wavelength_b, tem_b, global_params.template_std_b, 'blue') 
            log_wvlngth_temr, rebin_val_temr, rebin_ivar_temr = spec.rebin(global_params.wavelength_r, tem_r, global_params.template_std_r, 'red')

    except Exception as e:
    	pass

    if z<0.0: # Don't try to correlate if redshift is negative
        return np.nan, np.nan, np.nan, np.nan

    # remove any nan values
    nan_val_r = np.isnan(rebin_val_r)
    rebin_val_r[nan_val_r] = 0.0
    nan_val_temr = np.isnan(rebin_val_temr)
    rebin_val_temr[nan_val_temr] = 0.0

    nan_val_b = np.isnan(rebin_val_b)
    rebin_val_b[nan_val_b] = 0.0
    nan_val_temb = np.isnan(rebin_val_temb)
    rebin_val_temb[nan_val_temb] = 0.0

    xaxis_corr_b = np.arange(-1515, 1514, 1) # blue has 1515 bins
    ccf_b = np.correlate(rebin_val_temb[200:-141], rebin_val_b[200:-141], 'full')
    xaxis_corr_r = np.arange(-1526, 1525, 1) # red has 1526 bins
    ccf_r = np.correlate(rebin_val_temr, rebin_val_r, 'full') 

    if np.std(ccf_b) == 0.0:
        r_blue = np.nan
    else:
        r_blue = ((np.ndarray.max(ccf_b) * np.ndarray.mean(ccf_b)) / np.std(ccf_b)) * 1e32 

    if np.std(ccf_r) == 0.0:
        r_red = np.nan
    else:    
        r_red = ((np.ndarray.max(ccf_r) * np.ndarray.mean(ccf_r)) / np.std(ccf_r)) * 1e32 

    lag_r = (np.argmax(ccf_r) - ((len(ccf_r) + 0) / 2))
    lag_b = (np.argmax(ccf_b) - ((len(ccf_b) + 0) / 2)) 

    if z <= 0.7: # Need different pixel width to correct for offset 
        pix_width_r = global_params.pix_set1
        pix_width_b = global_params.pix_set1
    elif z > 0.7 and z <= 0.8:
        pix_width_r = global_params.pix_set2
        pix_width_b = global_params.pix_set2
    else:
        pix_width_r = global_params.pix_r
        pix_width_b = global_params.pix_b

    # pix_width_b = global_params.pix_b
    # pix_width_r = global_params.pix_r

    z_lag_r = round((10 ** (pix_width_r * lag_r) * (1 + (30 / 3e5)) - 1), 5)
    z_lag_b = round((10 ** (pix_width_b * lag_b) * (1 + (30 / 3e5)) - 1), 5)

    return z_lag_r, r_blue, r_red, z_lag_b



def ccf_function(data_blue, std_blue, data_red, std_red, rebin_val_b, rebin_val_r, log_wvlngth_temb_z_blue_list, rebin_val_temb_z_blue_list, log_wvlngth_temr_z_red_list, rebin_val_temr_z_red_list):
    """
    Run CCF algorithm to find spectroscopic redshift

	Keyword arguments:
	data_(blue/red) -- spectrum
	std_(blue/red) -- spectrum std
	rebin_val_(b/r) -- rebinned wavelength in logspace
	log_wvlngth_tem(b/r)_z_(blue/red)_list -- rebinned wavelengths of template 
	rebin_val_tem(b/r)_z_(blue_/red)_list -- rebinned flux values of template

	Return:
	final_redshifts (list) -- best fitting redshifts 
	z_lag_list (list) -- ccf values for each redshift (smaller corresponds to better correlation)

    """

    first_round_z = []
    first_round_z_lag = []
    first_round_low_list = []
    first_round_low_list_blue = []

    final_redshifts = []
    z_lag_list = []
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
    # start = time.time()
    for i, z in enumerate(global_params.initial_redshifts_fitting): 

        # Already rebinned the data in advance
        z_lag_r, r_blue, r_red, z_lag_b = ccf_algorithm(z, rebin_val_b, rebin_val_r, rebin_val_temb_z_blue_list[i], rebin_val_temr_z_red_list[i])

        # if(z_lag_r >= 0.0 and z_lag_r<=0.2):
        #     z_guess_list1.append((z, z_lag_r))
        #     first_round_low_list.append(False)
        # if(z_lag_r >= -0.07 and z_lag_r<=0.2):
        #     z_guess_list1.append((z, z_lag_r))
        #     first_round_low_list.append(True)        

        # if (z_lag_b >= 0.0 and z_lag_b<=0.2):
        #     z_guess_list1_blue.append((z, z_lag_b))
        #     first_round_low_list_blue.append(False)
        # if(z_lag_r >= -0.005 and z_lag_r<=0.05):
        #     z_guess_list1.append((z, z_lag_r)) 
        #     first_round_low_list.append(True)
        # if(z_lag_b >= -0.005 and z_lag_b<=0.05):
        #     z_guess_list1_blue.append((z, z_lag_b)) 
        #     first_round_low_list_blue.append(True)          
        if(z_lag_r >= 0.0 and z_lag_r<=0.1):
            z_guess_list1.append((z, z_lag_r))
            first_round_low_list.append(False)
        if(z_lag_r >= -0.07 and z_lag_r<=0.1):
            z_guess_list1.append((z, z_lag_r))
            first_round_low_list.append(True)        

        if (z_lag_b >= 0.0 and z_lag_b<=0.1):
            z_guess_list1_blue.append((z, z_lag_b))
            first_round_low_list_blue.append(False)
        if(z_lag_r >= -0.005 and z_lag_r<=0.05):
            z_guess_list1.append((z, z_lag_r)) 
            first_round_low_list.append(True)
        if(z_lag_b >= -0.005 and z_lag_b<=0.05):
            z_guess_list1_blue.append((z, z_lag_b)) 
            first_round_low_list_blue.append(True)     

        if print_bool == True:
            print('round 1', z, z_lag_r, r_red, z_lag_b, r_blue)

    # end = time.time()
    # print('Full run duration CCF round 1', (end - start)/60., ' min')
    # start = time.time()
    ########### Round 2 ##############
    count = 0
    for trial_z, trialz_lag_r in z_guess_list1: 
        if first_round_low_list[count] == False:    
            z = trial_z - trialz_lag_r + 0.03
        if first_round_low_list[count] == True: 
            z = trial_z - trialz_lag_r #+ 0.03              
        if z <= 0:
            continue
        z_lag_r, r_blue, r_red , z_lag_b = ccf_algorithm(z, rebin_val_b, rebin_val_r)

        if (z_lag_r > -0.01 and z_lag_r <= 0.09): 
            z_guess_list2.append((z, z_lag_r))
        if (z_lag_r > -0.005 and z_lag_r <= 0.001): 
            z_guess_list2.append((z, z_lag_r))

        if print_bool == True:
            print('round 2 z, zlag r_red', z, z_lag_r, r_red, z_lag_b, r_blue)

        count += 1

    count = 0
    for trial_z, trialz_lag_b in z_guess_list1_blue: 
        if first_round_low_list_blue[count] == False:   
            z = trial_z - trialz_lag_b + 0.03
        if first_round_low_list_blue[count] == True:    
            z = trial_z - trialz_lag_b #+ 0.03              
        if z <= 0:
            continue
        z_lag_r, r_blue, r_red , z_lag_b = ccf_algorithm(z, rebin_val_b, rebin_val_r)

        if (z_lag_b > -0.01 and z_lag_b <= 0.09 ): 
            z_guess_list2_blue.append((z, z_lag_b))
        if (z_lag_b > -0.005 and z_lag_b <= 0.001): 
            z_guess_list2_blue.append((z, z_lag_b))
        if print_bool == True:
            print('round 2  BLUE z, zlag r_red, r_blue', z, z_lag_r, r_red, z_lag_b, r_blue)

        count += 1

    ########### Round 3 ##############
    for trial_z, trialz_lag_r in z_guess_list2:
        z = trial_z - trialz_lag_r
        if z <= 0:
            continue
        z_lag_r, r_blue, r_red , z_lag_b = ccf_algorithm(z, rebin_val_b, rebin_val_r)
        if print_bool == True:
            print('round 3 z, zlag r_red', z, z_lag_r, r_red, z_lag_b, r_blue)
        if (z_lag_r > -0.01 and z_lag_r <= 0.009): 
            z_guess_list3.append((z, z_lag_r))
        if (z_lag_r > -0.0005 and z_lag_r <= 0.0005): 
            z_guess_list3.append((z, z_lag_r))

    for trial_z, trialz_lag_b in z_guess_list2_blue:
        z = trial_z - trialz_lag_b
        if z <= 0:
            continue
        z_lag_r, r_blue, r_red , z_lag_b = ccf_algorithm(z, rebin_val_b, rebin_val_r)
        if print_bool == True:
            print('round 3 BLUE z, zlag r_red', z, z_lag_r, r_red, z_lag_b, r_blue)
        if (z_lag_b > -0.01 and z_lag_b <= 0.012): 
            z_guess_list3_blue.append((z, z_lag_b))
        if (z_lag_b > -0.0005 and z_lag_b <= 0.0005 ):
            z_guess_list3_blue.append((z, z_lag_b))

    ########### Round 4 ##############
    for trial_z, trialz_lag_r in z_guess_list3:
        z = trial_z - abs(trialz_lag_r) 
        if z <= 0:
            continue
        z_lag_r = 999
        count = 0

        while (z_lag_r > 0.001 or z_lag_r < -0.001 and count <6): 
            count += 1
            z_lag_r, r_blue, r_red, z_lag_b = ccf_algorithm(z, rebin_val_b, rebin_val_r)
            z = z - z_lag_r
            if print_bool == True:
                print('round 4 z, zlag', z, z_lag_r, z_lag_b)
        if count < 6: 
            est_redshift = z - z_lag_r
            final_redshifts.append((round(est_redshift, 5)))
            z_lag_list.append(z_lag_r)

    for trial_z, trialz_lag_b in z_guess_list3_blue:
        z = trial_z - abs(trialz_lag_b) 
        if z <= 0:
            continue
        z_lag_b = 999
        count = 0

        while (z_lag_b > 0.001 or z_lag_b < -0.001 and count <6): 
            count += 1
            z_lag_r, r_blue, r_red, z_lag_b = ccf_algorithm(z, rebin_val_b, rebin_val_r)
            z = z - z_lag_b
            if print_bool == True:
                print('round 4 BLUE z, zlag', z, z_lag_r, z_lag_b)
        if count < 6: 
            est_redshift = z - z_lag_b
            final_redshifts.append((round(est_redshift, 5)))
            z_lag_list.append(z_lag_b)

    # end = time.time()
    # print('Full run duration CCF other rounds', (end - start)/60., ' min')

    # Remove any possible nan values 
    final_redshifts = np.array(final_redshifts)[~np.isnan(z_lag_list)]
    z_lag_list = np.array(z_lag_list)[~np.isnan(z_lag_list)]

    return final_redshifts, z_lag_list



def ccf_test(true_z, data_blue_test, std_blue_test, data_red_test, std_red_test, save_fig, save_name):
    """
    Test ccf algorithm with real redshift and plot results

    """


    # Rebin the spectra
    log_wvlngth_b, rebin_val_b, rebin_ivar_b = spec.rebin(global_params.wavelength_b, data_blue_test[0], std_blue_test[0], 'blue')
    log_wvlngth_r, rebin_val_r, rebin_ivar_r = spec.rebin(global_params.wavelength_r, data_red_test[0], std_red_test[0], 'red') 

    ccf_r_solutions = []
    ccf_b_solutions = []
    ccf_r_all = []
    ccf_b_all = []

    for i, z in enumerate(global_params.initial_redshifts_fitting): 

        # already rebinned the data in advance
        z_lag_r, r_blue, r_red, z_lag_b = ccf_algorithm(z, rebin_val_b, rebin_val_r)
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

    z_lag_r, r_blue, r_red, z_lag_b = ccf_algorithm(true_z, rebin_val_b, rebin_val_r) # Test ccf for true redshift
    print("CCF algo result true z", z_lag_r, r_blue, r_red, z_lag_b)

    ccf_r_all = np.array(ccf_r_all)
    ccf_b_all = np.array(ccf_b_all)

    # How shifted template looks like at true redshift
    tem_r = spec.shift_template(global_params.gal_zip, global_params.xaxis_r, true_z, 'r') 
    tem_b = spec.shift_template(global_params.gal_zip, global_params.xaxis_b, true_z, 'b')


    Fig, (Ax1, Ax2, Ax3, Ax4) = plt.subplots(4,1, figsize=(10,8))
    Ax1.plot(global_params.wavelength_b, data_blue_test[0])
    Ax1.plot(global_params.wavelength_r, data_red_test[0])
    Ax1.set_ylim(-0.1e-17, 1e-16)
    #Ax1.set_xlim(6800,7000)
    Ax1.set_title('Spectra')

    Ax2.plot(global_params.wavelength_b, tem_b)
    Ax2.plot(global_params.wavelength_r, tem_r)
    Ax2.set_ylim(-0.1e-17, 1e-16)
    #Ax2.set_xlim(6800,7000)
    Ax2.set_title("Templates")

    Ax3.scatter(ccf_b_all[:,0], ccf_b_all[:,1])
    Ax3.scatter(ccf_r_all[:,0], ccf_r_all[:,1])
    Ax3.scatter(true_z, z_lag_b, color='red')
    Ax3.scatter(true_z, z_lag_r, color='red')
    Ax3.set_title("CCF")
    Ax3.axvline(true_z, color='red')
    Ax3.set_ylabel('z_lag')
    Ax3.axhline(0.0, color='black', linestyle='--')

    Ax4.scatter(ccf_b_all[:,0], ccf_b_all[:,2])
    Ax4.scatter(ccf_r_all[:,0], ccf_r_all[:,2])
    Ax4.scatter(true_z, r_blue, color='red')
    Ax4.scatter(true_z, r_red, color='red')
    Ax4.axvline(true_z, color='red')
    Ax4.set_ylabel('r')
    Ax4.axhline(0.0, color='black', linestyle='--')
    plt.tight_layout()

    if save_fig == True:
        plt.savefig(save_name)
    else:
        plt.show()


