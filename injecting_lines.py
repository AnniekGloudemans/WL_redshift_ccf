import plotting as plot
import temp_fitting as temp
import spec_func as spec
import numpy as np


def line_detected_check(line_wavs, line_wav_rest_inject, redshift_inject):
	"""
	Check if the injected emission line is detected within 1% error

	Keyword arguments:
	line_wavs (array) -- wavelengths of the detected emission lines 
	line_wav_rest_inject (float) -- wavelength of the injected line
	redshift_inject (float) -- redshift of injected line

	Return:
	line_detected_bool (array) -- boolean array with True if injected line is detected
	spurious_lines (array) -- boolean array with True if more than 1 line is detected

	"""

	line_detected_bool = np.zeros(len(line_wavs))
	spurious_lines = np.zeros(len(line_wavs))

	for i, wavs in enumerate(line_wavs):
		line_detected = (wavs > 0.99*line_wav_rest_inject*(redshift_inject+1)) * (wavs < 1.01*line_wav_rest_inject*(redshift_inject+1))
		if np.sum(line_detected) > 0:
			line_detected_bool[i] = True
		else:
			line_detected_bool[i] = False
		if len(wavs) > 1:
			spurious_lines[i] = True
		else:
			spurious_lines[i] = False	

	return line_detected_bool, spurious_lines


def inject_gaussian_lines(linewav_rest_inject, redshift_inject, lum_inject, std_inject, data_blue, data_red, data_table_blue, data_table_red, target_names, run_name, mask_blue = False, mask_red = False, save_plot_bool = False, print_bool = False, show_result_bool = False):
	"""
	Inject gaussian emission lines in multiple spectra and determine success rate of detection by the algorithm

	Keyword arguments:
	linewav_rest_inject (float) - rest-frame wavelength of line to inject  
	redshift_inject (float/array) - redshifts of injected lines
	lum_inject (float/array) - luminosities of injected lines
	std_inject (float/array) - standard deviations/width of injected lines  
	data_(blue/red) -- blue/red spectra
	std_(blue/red) -- blue/red spectra standard deviations 
	data_table_(blue/red) (table) -- input data tables
	target_names (array) -- names of targets 
	mask_(blue/red) (array) -- masked wavelengths of spectra 
	save_plot_bool (boolean) -- default = False

	"""
	
	try:
		os.makedirs('../Output_ccf/Files/')
	except:
		pass

	try:
		num_spec = len(data_blue)
	except:
		# Make single spectrum into an array
		data_blue = np.array([data_blue])
		data_red = np.array([data_red])
		target_names = np.array([target_names])

	chipgap_blue = spec.find_chipgaps(data_blue[0], mask_blue, 'b')
	chipgap_red = spec.find_chipgaps(data_red[0], mask_red, 'r')

	# ---------------------------------- Fixed redshift -----------------------------------
	if type(redshift_inject) == float:
		line_detected_fraction_matrix = []
		spurious_lines_matrix = []	
		redshifts = np.full(len(data_blue), redshift_inject)

		for lum in lum_inject:
			line_detected_fraction = []
			spurious_lines = []	

			for std in std_inject:

				if print_bool == True:
					print('Running lum =', lum, ' and std = ', std)

				data_blue_inject, data_red_inject = spec.line_inject(lum, std, data_blue, data_red, redshift_inject, linewav_rest_inject, 'single')
				redshifts_template_fitting, emission_line_wavelengths, emission_line_fluxes, emission_line_snrs = temp.template_fitting(data_blue_inject, data_red_inject, data_table_blue, data_table_red, target_names, run_name,  mask_blue, mask_red, redshifts, save_plot_bool, False, print_bool, False, show_result_bool, False)

				line_detected_bool_array, spurious_lines_array = line_detected_check(emission_line_wavelengths, linewav_rest_inject, redshift_inject)

				line_detected_fraction.append(np.sum(line_detected_bool_array)/len(data_blue))
				spurious_lines.append(np.sum(spurious_lines_array)/len(data_blue))

			line_detected_fraction_matrix.append(line_detected_fraction)
			spurious_lines_matrix.append(spurious_lines)

		# np.save('../Output_ccf/Files/inject_lines_detected_fraction_'+run_name+'.npy', line_detected_fraction_matrix)
		# np.save('../Output_ccf/Files/spurious_lines_'+run_name+'.npy', spurious_lines_matrix)
		
		return line_detected_fraction_matrix, spurious_lines
		
		if save_plot_bool == True:
			plot.plot_2D_hist(std_inject, np.log10(lum_inject), line_detected_fraction_matrix, r'$\sigma_{inject}$' , 'Lum$_{inject}$', '../Output_ccf/Figures/ccf_performance/inject_lines_2D_hist_'+str(run_name)+'.pdf')


	# ---------------------------------- Fixed luminosity -----------------------------------
	elif type(lum_inject) == float and type(std_inject) == float: 

		line_detected_fraction = []
		spurious_lines = []	

		for redshift in redshift_inject:

			if print_bool == True:
				print('Running z =', redshift)

			redshifts = np.full(len(data_blue), redshift)
			data_blue_inject, data_red_inject = spec.line_inject(lum_inject, std_inject, data_blue, data_red, redshift, linewav_rest_inject, 'single', True, np.mean(redshift_inject)) # False, False)
			redshifts_template_fitting, emission_line_wavelengths, emission_line_fluxes, emission_line_snrs = temp.template_fitting(data_blue_inject, data_red_inject, data_table_blue, data_table_red, np.array([num + '_z'+ str(redshift) for num in target_names]), run_name,  mask_blue, mask_red, redshifts, save_plot_bool, False, print_bool, False, show_result_bool, False)

			# Check if injected line is detected
			line_detected_bool_array, spurious_lines_array = line_detected_check(emission_line_wavelengths, linewav_rest_inject, redshift)

			line_detected_fraction.append(np.sum(line_detected_bool_array)/len(data_blue))
			spurious_lines.append(np.sum(spurious_lines_array)/len(data_blue))
		
		
		# np.save('../Output_ccf/Files/inject_lines_detected_fraction_'+run_name+'.npy', line_detected_fraction)
		# np.save('../Output_ccf/Files/spurious_lines_'+run_name+'.npy', spurious_lines)

		# line_detected_fraction = np.load('../Output_ccf/inject_lines_detected_fraction_'+run_name+'.npy')
		# spurious_lines = np.load('../Output_ccf/spurious_lines_'+run_name+'.npy')
		
		return line_detected_fraction, spurious_lines

		if save_plot_bool == True:
			plot.plot_fraction_detected_vs_redshift(np.log10(lum_inject), redshift_inject, line_detected_fraction, spurious_lines, '$z_{inject}$', 'Fraction detected', chipgap_blue, chipgap_red, '../Output_ccf/Figures/ccf_performance/inject_lines_detected_curves_'+str(run_name)+'.pdf')

	else:
		print('Insert valid redshift, luminosity, and standard deviation arrays and floats')

	

