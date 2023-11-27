import numpy as np
import os
from astropy.io import fits
import warnings
import matplotlib.pyplot as plt
import time
from scipy.stats import kurtosis, skew
import traceback
from astropy import units as u
from astropy.table import Table
from astropy.convolution import convolve, Box1DKernel
from itertools import groupby 

import plotting as plot
import emission_lines as line
import open_data as read_data
import spec_func as spec
import ccf_func as ccf

from specutils.analysis import template_logwl_resample 
from specutils import Spectrum1D
import pdb


def template_fitting(spectra_blue, spectra_red, data_table_blue, data_table_red, target_names, run_name, mask_blue = False, mask_red = False, input_redshift_list = False, save_plot_bool = False, write_to_table_bool = True, print_bool = False, diagnostic_plot_bool = False, show_result_bool = False, fix_redshift = False, inject_bool = False):
	"""
	Calculate redshifts and identify emission lines in the spectra
	
	Keyword Arguments:
	path_save_plots (str) -- path for saving spectra
	run_name (str) -- name for spectra
	data_(blue/red) -- blue/red spectra
	std_(blue/red) -- blue/red spectra standard deviations
	data_table_(blue/red) (table) -- input data tables
	target_names (array) -- names of targets 
	input_redshift_list (boolean) -- default = False
	save_plot_bool (boolean) -- default = False

	Return:
	ratio_redshifts_all_targets (array) -- resulting redshifts from ccf algo
	line_wavs_all_targets (array) -- wavelengths of detected emission lines
	line_fluxes_all_targets (array) -- line fluxes of detected emission lines
	line_snr_all_targets (array) -- SNR values of detected emission lines
	
	"""

	#-------------------------------------------------- Create folders ---------------------------------------------------------
	try:
		os.makedirs('../Output_ccf/Catalogs/')
		os.makedirs('../Output_ccf/Figures/')
		os.makedirs('../Output_ccf/Figures/ccf_performance')
	except Exception as e: # If folder already exists 
		pass

	if save_plot_bool == True: # Make directories to save plots
		try:
			os.makedirs('../Output_ccf/Figures/spectra_output/spectra_'+run_name)
		except Exception as e: # If folder already exists 
			pass
	if diagnostic_plot_bool == True:
		try:
			os.makedirs('../Output_ccf/Figures/emission_lines_output/emission_lines_'+run_name)
		except Exception as e: # If folder already exists 
			pass

	try:
		num_spec = len(spectra_blue)
	except:
		# Make single spectrum into an array
		spectra_blue, spectra_red, target_names, input_redshift_list = np.array([spectra_blue]), np.array([spectra_red]), np.array([target_names]), np.array([input_redshift_list])


	# Mask specific regions of spectrum
	try:
		spectra_blue = spec.masking_spectra(spectra_blue, mask_blue) 
	except:
		pass
	try:
		spectra_red = spec.masking_spectra(spectra_red, mask_red)
	except:
		pass

	# Check for any absorption line spectra by fitting polynomial 
	absorption_spec_bool = spec.absorption_spectra_check(spectra_blue, spectra_red)

	# Prepare binned templates for the standard redshift range
	log_wvlngth_temb_z_blue_list = []
	rebin_val_temb_z_blue_list = []

	log_wvlngth_temr_z_red_list = []
	rebin_val_temr_z_red_list = []


	#-------------------------------------------------- Run cross-correlation --------------------------------------------------

	if fix_redshift == True:
		print("Fixed redshift - skipping CCF for {0} targets".format(len(spectra_blue)))
	else:
		print("Running CCF for {0} targets...".format(len(spectra_blue)))

	ratio_redshifts_all_targets = []
	line_wavs_all_targets = []
	line_fluxes_all_targets = []
	line_snr_all_targets = []
	single_line_bool_all_targets = []

	# # Determine location of chipgap
	# chipgap_blue = spec.find_chipgaps(spectra_blue[0], mask_blue, 'b')
	# chipgap_red = spec.find_chipgaps(spectra_red[0], mask_red, 'r')

	for i in range(len(spectra_blue)):


		if fix_redshift == True: # If the redshift is fixed skip the CCF
			line_wav_list, line_flux_list, line_snr_list = line.finding_emission_lines(spectra_blue[i], spectra_red[i], [input_redshift_list[i]], True, diagnostic_plot_bool, '../Output_ccf/Figures/emission_lines_output/emission_lines_'+run_name+'/emission_line_') #  True, False,
			
			line_wavs_all_targets.append(line_wav_list)
			line_fluxes_all_targets.append(line_flux_list)
			line_snr_all_targets.append(line_snr_list)
			ratio_redshifts_all_targets.append(np.nan)

			if len(line_wav_list) == 1:
				single_line_bool_all_targets.append(True)
			else:
				single_line_bool_all_targets.append(False)

			if save_plot_bool == True:
				plot.plot_spectrum_blue_red_lines(spectra_blue[i], spectra_red[i], target_names[i], line_wav_list, np.nan, '../Output_ccf/Figures/spectra_output/spectra_'+run_name+'/spectrum_'+str(target_names[i])+'.pdf', save_plot_bool, input_redshift_list[i], show_result_bool) 

			continue

		if absorption_spec_bool[i] == True: # Move to next target if absorption line spectrum
			ratio_redshift = np.nan
			line_wav_list = np.array([])
			ratio_redshifts_all_targets.append(ratio_redshift)
			line_wavs_all_targets.append(np.array([]))
			line_fluxes_all_targets.append(np.array([]))
			line_snr_all_targets.append(np.array([]))
			single_line_bool_all_targets.append(False)#np.nan)

			if save_plot_bool == True:
				try: 
					plot.plot_spectrum_blue_red_lines(spectra_blue[i], spectra_red[i], target_names[i], [], np.nan, '../Output_ccf/Figures/spectra_output/spectra_'+run_name+'/spectrum_'+str(target_names[i])+'_abs.pdf', save_plot_bool, input_redshift_list[i], show_result_bool)
				except:
					plot.plot_spectrum_blue_red_lines(spectra_blue[i], spectra_red[i], target_names[i], [], np.nan, '../Output_ccf/Figures/spectra_output/spectra_'+run_name+'/spectrum_'+str(target_names[i])+'_abs.pdf', save_plot_bool, False, show_result_bool)
			continue 

		if print_bool == True:
			print('Target', target_names[i])
			try:
				print("Input redshift = ", input_redshift_list[i])
			except:
				print("No target redshift known")

		# # Preprocess the spectrum
		# data_blue[i] = spec.preprocess_spectra(data_blue[i], 'blue')#, True, '../Output_ccf/Figures/spectra_output/spectra_'+run_name+'/cont_sub_spectrum_blue_'+str(target_names[i])+'.pdf')
		# data_red[i] = spec.preprocess_spectra(data_red[i], 'red')#, True, '../Output_ccf/Figures/spectra_output/spectra_'+run_name+'/cont_sub_spectrum_red_'+str(target_names[i])+'.pdf')

		# Run the cross-correlation
		ccf_redshifts, z_lag_list = ccf.ccf_function(spectra_blue[i], spectra_red[i])

		# Remove duplicates from estimated redshift list
		ccf_redshifts_round = np.round(ccf_redshifts,3) # Faster if round of to 2
		ccf_redshifts_clean = []
		for z_est in ccf_redshifts_round:
			if z_est not in ccf_redshifts_clean:
				ccf_redshifts_clean.append(z_est)

		if print_bool == True:
			print("CCF redshift solutions..", ccf_redshifts_clean)


	#-------------------------------------------------- Determine redshift and emission lines --------------------------------------------------

		# If CCF solution found then check if emission lines are real and determine the redshift by calculating the wavelength ratios
		if len(z_lag_list) > 0:

			line_wav_list, line_flux_list, line_snr_list = line.finding_emission_lines(spectra_blue[i], spectra_red[i], ccf_redshifts_clean, False, diagnostic_plot_bool, '../Output_ccf/Figures/emission_lines_output/emission_lines_'+run_name+'/emission_line_')
			
			if print_bool == True:
				print('Lines detected before final = ', line_wav_list)

			# Determine the best redshift from the ratios of the real emisson line wavelengths            
			if len(line_wav_list) > 1:
				ratio_redshift = line.line_ratios(line_wav_list)
			elif len(line_wav_list) == 1:
				ratio_redshift = float((line_wav_list[0]/3727) - 1) # If only 1 line set to OII for now
			else:
				ratio_redshift = np.nan

			if print_bool == True:
				print('Ratio redshift', ratio_redshift)

			if np.isnan(ratio_redshift) == False: # If found a redshift from ratio's then check if there are any additional emission lines that were missed
				line_wav_list_extra, line_flux_list_extra, line_snr_list_extra = line.finding_emission_lines(spectra_blue[i], spectra_red[i], np.array([ratio_redshift]), True, diagnostic_plot_bool, '../Output_ccf/Figures/emission_lines_output/emission_lines_'+run_name+'/emission_line_') #  True, False,
				
				# Merge the two lists 
				for p, line_wav in enumerate(line_wav_list_extra):
					if np.logical_or(len(line_wav_list)==0, abs(line_wav-line.closest_value(line_wav_list,line_wav))>10.0): #  abs(line_wav - (6563.*(1+ratio_redshift)))<10.0
						line_wav_list.append(line_wav)
						line_flux_list.append(line_flux_list_extra[p])
						line_snr_list.append(line_snr_list_extra[p])

			line_wav_list = np.array(line_wav_list)
			line_flux_list = np.array(line_flux_list)
			line_snr_list = np.array(line_snr_list)

			if print_bool == True:
				print('Lines detected = ', line_wav_list)

			# Check if Ha line in there, if yes, then fit double gaussian and replace line fluxes
			Ha_line_idx = (ratio_redshift<0.46)*(np.array(line_wav_list)-25 <(1.+ratio_redshift)*6563)*(np.array(line_wav_list)+25 >(1.+ratio_redshift)*6563)

			if np.sum(Ha_line_idx) > 0:
				line_wav_Ha, line_flux_Ha, line_snr_Ha, sep_peaks_Ha, reduced_chisq_Ha = line.fit_double_gauss(spectra_red[i], (1+ratio_redshift)*6563., diagnostic_plot_bool, '../Output_ccf/Figures/emission_lines_output/emission_lines_'+run_name+'/emission_line_Ha_double_gauss.pdf', False)

				# insert new Ha line wavelength and flux
				if np.isnan(line_wav_Ha) == True:
					pass
				else:
					line_wav_list[Ha_line_idx] = line_wav_Ha
					line_flux_list[Ha_line_idx] = line_flux_Ha
					line_snr_list[Ha_line_idx] = line_snr_Ha

			# Classify the single emission lines
			if len(line_wav_list) == 1:

				# Check if OII, OIII, Lya or CIV
				linename, ratio_redshift, line_wav_list, line_flux_list, line_snr_list = line.single_emission_line(spectra_blue[i], spectra_red[i], line_wav_list[0], diagnostic_plot_bool, '../Output_ccf/Figures/emission_lines_output/emission_lines_'+run_name+'/emission_line_')
				
				if np.isnan(line_wav_list[0]) == False: # Check if single line was real
					single_line_bool_all_targets.append(True)
				else:
					single_line_bool_all_targets.append(False)

			else:
				single_line_bool_all_targets.append(False)

			# Save line results to lists
			ratio_redshifts_all_targets.append(ratio_redshift)
			line_wavs_all_targets.append(line_wav_list)
			line_fluxes_all_targets.append(line_flux_list)
			line_snr_all_targets.append(line_snr_list)
			
		else:
			ratio_redshift = np.nan
			ratio_redshifts_all_targets.append(ratio_redshift)
			single_line_bool_all_targets.append(False)

			# if fix_redshift == False:
			line_wavs_all_targets.append(np.array([]))
			line_fluxes_all_targets.append(np.array([]))
			line_snr_all_targets.append(np.array([]))
			line_wav_list = np.array([])

	#-------------------------------------------------- Save results --------------------------------------------------

		if save_plot_bool == True or show_result_bool == True:
			try:
				plot.plot_spectrum_blue_red_lines(spectra_blue[i], spectra_red[i], target_names[i], line_wav_list, ratio_redshift, '../Output_ccf/Figures/spectra_output/spectra_'+run_name+'/spectrum_'+str(target_names[i])+'.pdf', save_plot_bool, input_redshift_list[i], show_result_bool) 
			except: # if input_redshift = False
				plot.plot_spectrum_blue_red_lines(spectra_blue[i], spectra_red[i], target_names[i], line_wav_list, ratio_redshift, '../Output_ccf/Figures/spectra_output/spectra_'+run_name+'/spectrum_'+str(target_names[i])+'.pdf', save_plot_bool, False, show_result_bool)

	# Write to table
	if write_to_table_bool == True:
		try:
			read_data.write_to_table(data_table_blue, data_table_red, ratio_redshifts_all_targets, line_wavs_all_targets, line_fluxes_all_targets, line_snr_all_targets, single_line_bool_all_targets, absorption_spec_bool, '../Output_ccf/Catalogs/catalog_'+str(run_name), input_redshift_list)
		except Exception as e:
			print(e)
			#traceback.print_exc()
			print('Cannot write to file')

	if save_plot_bool == True and inject_bool == False:

		try:
			plot.ccf_redshift_output(ratio_redshifts_all_targets, '../Output_ccf/Figures/ccf_performance/ccf_redshifts_'+run_name+'.pdf', single_line_bool_all_targets)
			
			if np.sum(single_line_bool_all_targets) > 0:
				plot.single_line_spectra(np.array(spectra_blue)[single_line_bool_all_targets], np.array(spectra_red)[single_line_bool_all_targets], target_names[single_line_bool_all_targets], np.array(ratio_redshifts_all_targets)[single_line_bool_all_targets], np.array(line_wavs_all_targets)[single_line_bool_all_targets], np.array(line_fluxes_all_targets)[single_line_bool_all_targets], np.array(line_snr_all_targets)[single_line_bool_all_targets], '../Output_ccf/Figures/spectra_output/spectra_'+run_name+'/single_line_spectra'+'.pdf')

			try: # only if input_redshifts available and if redshift not fixed

				if fix_redshift == False:
					diff_z = abs(input_redshift_list-ratio_redshifts_all_targets)
					idx_nan = np.isnan(ratio_redshifts_all_targets)

					plot.ccf_performace(input_redshift_list, ratio_redshifts_all_targets, '../Output_ccf/Figures/ccf_performance/ccf_performace_'+run_name+'.pdf', single_line_bool_all_targets)

					print('Percentage correct redshift in total = ', np.int((np.sum(diff_z<0.1)/len(diff_z))*100), '%')
					print('Percentage correct redshift no nan = ', np.int((np.sum(diff_z[~idx_nan]<0.1)/len(diff_z[~idx_nan]))*100), '%')
					print('Number single line targets found = ', np.nansum(single_line_bool_all_targets))

			except:
				print('No input redshift or no lines found, so cannot determine percentage correct')

		except Exception as e:
			print('Error in plotting')
			print(e)
	else:
		print('Number single line targets found = ', np.nansum(single_line_bool_all_targets))

	print('Done!')
	#pdb.set_trace()
	return np.array(ratio_redshifts_all_targets), line_wavs_all_targets, line_fluxes_all_targets, line_snr_all_targets



