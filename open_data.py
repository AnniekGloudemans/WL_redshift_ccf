from astropy.io import fits
from astropy.table import Table
import numpy as np
import global_params
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from specutils import Spectrum1D
import warnings

def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(v)
    return out


def pick_spectrum(target_bool_array, start_num, num_spec):
	idx = np.where(target_bool_array == True)[0]
	chosen_idx = idx[start_num]
	target_bool_array[0:chosen_idx] = False

	idx_targets = np.where(target_bool_array[chosen_idx:] == True)[0]
	idx_max = idx_targets[num_spec]
	target_bool_array[chosen_idx+idx_max:] = False
	return target_bool_array


def open_input_data_specutils(file_path, arm): 
	
	file = fits.open(file_path)
	table = Table.read(file_path)

	spec_flux = (file[1].data*file[5].data)* u.Unit('erg cm-2 s-1 AA-1') 
	spec_std = (file[3].data*file[5].data)* u.Unit('erg cm-2 s-1 AA-1') 

	if arm == 'r':
		lamb = global_params.wavelength_r*u.AA
	elif arm == 'b':
		lamb = global_params.wavelength_b*u.AA

	uncertainty = StdDevUncertainty(0.001*1e-17*np.ones(len(lamb))*u.Unit('erg cm-2 s-1 AA-1')) # CHANGE! 

	spectra_array = []
	for i in range(len(spec_flux)):
		spec = Spectrum1D(spectral_axis=lamb, flux=spec_flux[i], uncertainty=uncertainty)#spec_std)
		spectra_array.append(spec)

	return file, table, np.array(spectra_array)


def open_input_data_specutils_weaveio(data_table, arm): 

	spec_flux = (data_table['flux']*data_table['sensfunc'])* u.Unit('erg cm-2 s-1 AA-1') 
	spec_std = (data_table['ivar']*data_table['sensfunc'])* u.Unit('erg cm-2 s-1 AA-1') 

	if arm == 'r':
		lamb = global_params.wavelength_r*u.AA
	elif arm == 'b':
		lamb = global_params.wavelength_b*u.AA

	uncertainty = StdDevUncertainty(0.001*1e-17*np.ones(len(lamb))*u.Unit('erg cm-2 s-1 AA-1')) # CHANGE! 

	spectra_array = []
	for i in range(len(spec_flux)):
		spec = Spectrum1D(spectral_axis=lamb, flux=spec_flux[i], uncertainty=uncertainty)#spec_std)
		spectra_array.append(spec)

	return np.array(spectra_array)


def input_redshifts_match_weaveio(target_table, input_table):
	"""
	Find input redshifts 

	Keyword arguments:
	target_table (table) -- Table with targets from spectral cube
	input_table (table) -- Table with input parameters of simulated data

	Return:
	input_redshifts (array) -- List of input redshifts for all targets 

	"""
	input_redshifts = []

	for i in range(len(target_table)):
		idx_name = np.where(input_table['TARGNAME'] == target_table['targname'][i])[0] # SHOULD IT NOT BE Z-1??
		if len(input_table['Z'][idx_name]) > 0:
			try:
				input_redshifts.append(input_table['Z'][idx_name][0])
			except:
				input_redshifts.append(input_table['Z'][idx_name]) 
		else:
			input_redshifts.append(np.nan)

	return np.array(input_redshifts)


def match_blue_red_spectra(table_blue, table_red):
	"""
	Sort blue and red spectra to same target order

	Keyword arguments:
	table_blue/red (table) -- Table with all spectra information

	Return:
	table_blue/red (table) -- Sorted/matched table with all spectra information 

	"""

	if len(table_blue) == len(table_red):
		return table_blue, table_red

	idx_red_list = []
	idx_blue_list = []

	for i in range(len(table_blue)):
		if table_blue['targname'][i] in table_red['targname']:
			idx_red = np.where(table_red['targname'] == table_blue['targname'][i])[0]

			idx_blue_list.append(i)
			idx_red_list.append(idx_red[0])

	return table_blue[np.array(idx_blue_list)], table_red[np.array(idx_red_list)]



def input_redshifts_match(target_table, input_table):
	"""
	Find input redshifts 

	Keyword arguments:
	target_table (table) -- Table with targets from spectral cube
	input_table (table) -- Table with input parameters of simulated data

	Return:
	input_redshifts (array) -- List of input redshifts for all targets 

	"""
	input_redshifts = []

	for i in range(len(target_table)):
		idx_name = np.where(input_table['TARGNAME'] == target_table['TARGNAME'][i])[0] # SHOULD IT NOT BE Z-1??
		if len(input_table['Z_2'][idx_name]) > 0:
			try:
				input_redshifts.append(input_table['Z_2'][idx_name][0])
			except:
				input_redshifts.append(input_table['Z_2'][idx_name]) 
		else:
			input_redshifts.append(np.nan)

	return np.array(input_redshifts)



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


	with warnings.catch_warnings():  # Ignore warnings
		warnings.simplefilter('ignore')

		absorption_spec_bool_col = Table.Column(name=str('absorption_line_flag'), data=absorption_spec_bools)
		single_line_bool_col = Table.Column(name=str('single_line_flag'), data=single_line_bools)

		table_blue_write.add_column(single_line_bool_col)
		table_red_write.add_column(single_line_bool_col)
		table_blue_write.add_column(absorption_spec_bool_col)
		table_red_write.add_column(absorption_spec_bool_col)

		line_wavs_write = boolean_indexing(line_wavs)
		line_fluxes_write = boolean_indexing(line_fluxes)
		line_snrs_write = boolean_indexing(line_snrs)	

		full_line_wav_colum = Table.Column(name='line_wavs', data=np.array(line_wavs_write), unit=u.AA)
		full_line_fluxes_colum = Table.Column(name='line_fluxes', data=np.array(line_fluxes_write))
		full_line_snrs_colum = Table.Column(name='line_snrs', data=np.array(line_snrs_write))
		table_blue_write.add_column(full_line_wav_colum)
		table_red_write.add_column(full_line_wav_colum)
		table_blue_write.add_column(full_line_fluxes_colum)
		table_red_write.add_column(full_line_fluxes_colum)
		table_blue_write.add_column(full_line_snrs_colum)
		table_red_write.add_column(full_line_snrs_colum)

		for t in range(max_num_lines): # loop over the max amount of emission lines
			line_wav_column = Table.Column(name='line_wav_'+str(t+1), data=np.array(line_wavs_write)[:,t], unit=u.AA)
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


