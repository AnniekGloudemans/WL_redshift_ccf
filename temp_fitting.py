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
from itertools import groupby 

import plotting as plot
import emission_lines as line
import spec_func as spec
import ccf_func as ccf
import global_params
import pdb

def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(v)
    return out


def write_to_table(table_blue, table_red, z_ccfs, line_wavs, line_fluxes, line_snrs, single_line_bools, absorption_spec_bools, savename, input_redshift_list = False):
	"""
	Write resulting redshift and emission lines to orginial table 

	"""
	table_blue_write = table_blue
	table_red_write = table_red

	try:
		max_num_lines = max([len(i) for i in line_wavs])


	except:
		max_num_lines = 1

	z_ccf = Table.Column(name=str('z_ccf'), data=z_ccfs)
	
	try:
		table_blue_write.add_column(z_ccf)
		table_red_write.add_column(z_ccf)
	except Exception as e:
		# print('why no add', e)
		table_blue_write.replace_column('z_ccf', z_ccf)
		table_red_write.replace_column('z_ccf', z_ccf)

	try:
		input_z_column = Table.Column(name=str('z_true'), data=input_redshift_list)
		try:
			table_blue_write.add_column(input_z_column)
			table_red_write.add_column(input_z_column)
		except:			
			table_blue_write.replace_column('z_true', input_z_column)
			table_red_write.replace_column('z_true', input_z_column)
	except:
		pass

	absorption_spec_bool_col = Table.Column(name=str('absorption_line_flag'), data=absorption_spec_bools)
	single_line_bool_col = Table.Column(name=str('single_line_flag'), data=single_line_bools)

	table_blue_write.add_column(single_line_bool_col)
	table_red_write.add_column(single_line_bool_col)
	table_blue_write.add_column(absorption_spec_bool_col)
	table_red_write.add_column(absorption_spec_bool_col)

	line_wavs_write = boolean_indexing(line_wavs)
	line_fluxes_write = boolean_indexing(line_fluxes)
	line_snrs_write = boolean_indexing(line_snrs)	

	for t in range(max_num_lines): # loop over the max amount of emission lines
		line_wav_column = Table.Column(name='line_wav_'+str(t+1), data=np.array(line_wavs_write)[:,t])
		line_flux_column = Table.Column(name='line_flux_'+str(t+1), data=np.array(line_fluxes_write)[:,t])
		line_snr_column = Table.Column(name='line_snr_'+str(t+1), data=np.array(line_snrs_write)[:,t])

		table_blue_write.add_column(line_wav_column)
		table_red_write.add_column(line_wav_column)
		table_blue_write.add_column(line_flux_column)
		table_red_write.add_column(line_flux_column)
		table_blue_write.add_column(line_snr_column)
		table_red_write.add_column(line_snr_column)

	table_blue_write.write(savename+'_ccf_results_blue.fits', overwrite=True)
	table_red_write.write(savename+'_ccf_results_red.fits', overwrite=True)



def template_fitting(run_name, data_blue, std_blue, data_red, std_red, data_table_blue, data_table_red, input_redshift_list = False, save_plot_bool = False):
	"""
	Calculate redshifts and identify emission lines in the spectra
	
	Keyword Arguments:
	path_save_plots (str) -- path for saving spectra
	run_name (str) -- name for spectra
	data_(blue/red) -- blue/red spectra
	std_(blue/red) -- blue/red spectra standard deviations
	data_table_(blue/red) (table) -- input data tables
	input_redshift_list (boolean) -- default = False
	save_plot_bool (boolean) -- default = False

	Return:
	ratio_redshifts_all_targets (array) -- resulting redshifts from ccf algo
	line_wavs_all_targets (array) -- wavelengths of detected emission lines
	line_fluxes_all_targets (array) -- line fluxes of detected emission lines
	line_snr_all_targets (array) -- SNR values of detected emission lines
	
	"""

	try:
		os.makedirs('../Output_ccf/Catalogs/')
	except Exception as e: # If folder already exists 
		pass

	if save_plot_bool == True: # Make directories to save plots
		try:
			os.makedirs('../Output_ccf/Figures/emission_lines_output/emission_lines_'+run_name)
			os.makedirs('../Output_ccf/Figures/spectra_output/spectra_'+run_name)

		except Exception as e: # If folder already exists 
			pass
	
	# Check for any absorption line spectra by fitting polynomial 
	absorption_spec_bool = spec.absorption_spectra_check(data_blue, data_red)

	# Prepare binned templates for the standard redshift range
	log_wvlngth_temb_z_blue_list = []
	rebin_val_temb_z_blue_list = []

	log_wvlngth_temr_z_red_list = []
	rebin_val_temr_z_red_list = []

	print("Preparing templates..")
	for i, z in enumerate(global_params.initial_redshifts_fitting): 

		tem_r = spec.shift_template(global_params.gal_zip, global_params.xaxis_r, z, 'r') # shifting the template with redshift
		tem_b = spec.shift_template(global_params.gal_zip, global_params.xaxis_b, z, 'b')

		log_wvlngth_temb_z_blue, rebin_val_temb_z_blue, rebin_ivar_temb_z_blue = spec.rebin(global_params.wavelength_b, tem_b, global_params.template_std_b, 'blue') 
		log_wvlngth_temr_z_red, rebin_val_temr_z_red, rebin_ivar_temr_z_red = spec.rebin(global_params.wavelength_r, tem_r, global_params.template_std_r, 'red')

		log_wvlngth_temb_z_blue_list.append(log_wvlngth_temb_z_blue)
		rebin_val_temb_z_blue_list.append(rebin_val_temb_z_blue)

		log_wvlngth_temr_z_red_list.append(log_wvlngth_temr_z_red)
		rebin_val_temr_z_red_list.append(rebin_val_temr_z_red)

	print("Starting CCF for {0} targets".format(len(data_blue)))

	ratio_redshifts_all_targets = []
	line_wavs_all_targets = []
	line_fluxes_all_targets = []
	line_snr_all_targets = []
	single_line_bool_all_targets = []

	for i in range(len(data_blue)):

		if absorption_spec_bool[i] == True: # Move to next target if absorption line spectrum
			ratio_redshift = np.nan
			line_wav_list = np.array([])
			ratio_redshifts_all_targets.append(ratio_redshift)
			line_wavs_all_targets.append(np.array([]))
			line_fluxes_all_targets.append(np.array([]))
			line_snr_all_targets.append(np.array([]))
			single_line_bool_all_targets.append(np.nan)

			if save_plot_bool == True:
				plot.plot_spectrum_blue_red_lines(data_blue[i], data_red[i], data_table_blue['TARGNAME'][i], [np.nan], np.nan, '../Output_ccf/Figures/spectra_output/spectra_'+run_name+'/spectrum_'+str(data_table_blue['TARGNAME'][i])+'_abs.pdf', input_redshift_list[i])
			continue 

		print('Target', data_table_blue['TARGNAME'][i])
		try:
			print("Input redshift = ", input_redshift_list[i])
		except:
			print("No target redshift known")

		# Rebin the spectra
		log_wvlngth_b, rebin_val_b, rebin_ivar_b = spec.rebin(global_params.wavelength_b, data_blue[i], std_blue[i], 'blue')
		log_wvlngth_r, rebin_val_r, rebin_ivar_r = spec.rebin(global_params.wavelength_r, data_red[i], std_red[i], 'red') 

		# Run the cross-correlation
		ccf_redshifts, z_lag_list = ccf.ccf_function(data_blue[i], std_blue[i], data_red[i], std_red[i], rebin_val_b, rebin_val_r, log_wvlngth_temb_z_blue_list, rebin_val_temb_z_blue_list, log_wvlngth_temr_z_red_list, rebin_val_temr_z_red_list)

		# Remove duplicates from estimated redshift list
		ccf_redshifts_round = np.round(ccf_redshifts,3)
		ccf_redshifts_clean = []
		for z_est in ccf_redshifts_round:
			if z_est not in ccf_redshifts_clean:
				ccf_redshifts_clean.append(z_est)
		print("CCF redshift solutions..", ccf_redshifts_clean)

		# If CCF solution found then check if emission lines are real and determine the redshift by calculating the wavelength ratios
		if len(z_lag_list) > 0:
			line_wav_list, line_flux_list, line_snr_list = line.finding_emission_lines(data_blue[i], data_red[i], ccf_redshifts_clean, False, False, '../Output_ccf/Figures/emission_lines_output/emission_lines_'+run_name+'/emission_line_')

			# Determine the best redshift from the ratios of the real emisson line wavelengths            
			if len(line_wav_list) > 1:
				ratio_redshift = line.line_ratios(line_wav_list)
			elif len(line_wav_list) == 1:
				ratio_redshift = float((line_wav_list[0]/3727) - 1) # If only 1 line set to OII for now
			else:
				ratio_redshift = np.nan
			print('Ratio redshift', ratio_redshift)

			if np.isnan(ratio_redshift) == False: # If found a redshift from ratio's then check if there are any additional emission lines that were missed
				line_wav_list_extra, line_flux_list_extra, line_snr_list_extra = line.finding_emission_lines(data_blue[i], data_red[i], np.array([ratio_redshift]), True, False, 'Output/Figures/emission_lines_output/emission_lines_'+run_name+'/emission_line_') 
				
				# Merge the two lists
				for p, line_wav in enumerate(line_wav_list_extra):
					if np.logical_or(len(line_wav_list)==0, abs(line_wav-line.closest_value(line_wav_list,line_wav))>10.0):
					#if line_wav not in line_wav_list:
						line_wav_list.append(line_wav)
						line_flux_list.append(line_flux_list_extra[p])
						line_snr_list.append(line_snr_list_extra[p])

			line_wav_list = np.array(line_wav_list)
			line_flux_list = np.array(line_flux_list)
			line_snr_list = np.array(line_snr_list)

			print('Lines detected = ', line_wav_list)

			# Check if Ha line in there, if yes, then fit double gaussian and replace line fluxes
			Ha_line_idx = (ratio_redshift<0.46)*(np.array(line_wav_list)-25 <(1.+ratio_redshift)*6563)*(np.array(line_wav_list)+25 >(1.+ratio_redshift)*6563)
			if np.sum(Ha_line_idx) > 0:
				line_wav_Ha, line_flux_Ha, line_snr_Ha = line.fit_Ha_double_gauss(data_red[i], ratio_redshift, '../Output_ccf/Figures/emission_lines_output/emission_lines_'+run_name+'/emission_line_Ha.pdf')
				# insert new Ha line wavelength and flux
				if np.isnan(line_wav_Ha) == True:
					pass
				else:
					line_wav_list[Ha_line_idx] = line_wav_Ha
					line_flux_list[Ha_line_idx] = line_flux_Ha
					line_snr_list[Ha_line_idx] = line_snr_Ha

			# Save line results to lists
			ratio_redshifts_all_targets.append(ratio_redshift)
			line_wavs_all_targets.append(line_wav_list)
			line_fluxes_all_targets.append(line_flux_list)
			line_snr_all_targets.append(line_snr_list)
		
		else:
			ratio_redshift = np.nan
			ratio_redshifts_all_targets.append(ratio_redshift)
			line_wavs_all_targets.append(np.array([]))
			line_fluxes_all_targets.append(np.array([]))
			line_snr_all_targets.append(np.array([]))
			line_wav_list = np.array([])


		# Check if any single emission lines in there
		try:
			if len(line_wav_list) == 1:
				single_line_bool = True
			else:
				single_line_bool = False
		except: # e.g. if there is no line_wav_list
			single_line_bool = False

		single_line_bool_all_targets.append(single_line_bool)

		# If single emission line: include check if OII or Lya

		# Plot spectrum with identified emission lines
		if save_plot_bool == True:
			plot.plot_spectrum_blue_red_lines(data_blue[i], data_red[i], data_table_blue['TARGNAME'][i], line_wav_list, ratio_redshift, '../Output_ccf/Figures/spectra_output/spectra_'+run_name+'/spectrum_'+str(data_table_blue['TARGNAME'][i])+'.pdf', input_redshift_list[i])


	# Write to table
	try:
		write_to_table(data_table_blue, data_table_red, ratio_redshifts_all_targets, line_wavs_all_targets, line_fluxes_all_targets, line_snr_all_targets, single_line_bool_all_targets, absorption_spec_bool, '../Output_ccf/Catalogs/catalog_'+str(run_name), input_redshift_list)
	except Exception as e:
		print(e)
		traceback.print_exc()
		print('Cannot write to file')
		pdb.set_trace()

	return np.array(ratio_redshifts_all_targets), np.array(line_wavs_all_targets), np.array(line_fluxes_all_targets), np.array(line_snr_all_targets)

