import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import time
import numpy as np
import global_params


# Set colours figures 
c_blue_spec = 'C0'
c_red_spec = 'C1'
cmap = matplotlib.cm.get_cmap('inferno')
c_fit = cmap(0.8)
c_general = cmap(0.2)
c_single_line = cmap(0.4)


def plot_spectrum_blue_red_lines(data_blue, data_red, target_name, line_wavelengths, z_ratio, savename, save_fig = False, input_redshift = False, show_plot = False):
	"""
	Plot blue and red spectrum and indicate emission lines found

	Keyword arguments:
	data_(blue/red) (array) -- blue and red spectrum
	target_name (str) -- Name of target
	line_wavelengths (array) -- wavelengths of emission lines found in ccf algo
	z_ratio (float) -- redshift estimated by ccf algo
	savename (str) -- filename for saving plot
	input_redshift (float) -- input redshift of target (if available)

	"""

	# Sort linewavelengths for plotting
	line_wavelengths = np.sort(line_wavelengths)

	if len(line_wavelengths) == 0:
		fig = plt.figure(tight_layout=True, figsize=(10,7))
		gs = gridspec.GridSpec(1,1)

	elif len(line_wavelengths) == 1:
		fig = plt.figure(tight_layout=True, figsize=(10,7))
		gs = gridspec.GridSpec(2,3)

	elif len(line_wavelengths) > 6 and len(line_wavelengths) <= 12:
		fig = plt.figure(tight_layout=True, figsize=(10,10))
		gs = gridspec.GridSpec(3,6)

	elif len(line_wavelengths) > 12:
		fig = plt.figure(tight_layout=True, figsize=(10,13))
		gs = gridspec.GridSpec(4,6)
	
	else:
		fig = plt.figure(tight_layout=True, figsize=(10,7))
		gs = gridspec.GridSpec(2, len(line_wavelengths))

	ax = fig.add_subplot(gs[0, :])
	ax.plot(global_params.wavelength_b, data_blue, c=c_blue_spec, alpha=0.7)
	ax.plot(global_params.wavelength_r, data_red, c=c_red_spec, alpha=0.7)

	for l in range(len(line_wavelengths)):
		ax.axvline(line_wavelengths[l], color='grey', linestyle='--')

	ax.set_ylabel(r'Flux (erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$)', fontsize=14) # per angstrom or not??
	ax.set_xlabel(r'Observed wavelength ($\AA$)', fontsize=14)
	ax.set_ylim(np.min(data_red), np.max(data_red)+1e-17)

	if input_redshift == False:
		ax.set_title(target_name + r'    z$_{CCF}$ ='+str(np.round(z_ratio,3)), fontsize=14)
	else:
		ax.set_title(target_name + r'    z$_{CCF}$ ='+str(np.round(z_ratio,3))+'   z$_{input}$ ='+str(np.round(input_redshift,3)), fontsize=14)

	for i in range(len(line_wavelengths)):

		if i < 6:
			ax = fig.add_subplot(gs[1, i])
		elif i >= 6 and i<12:
			ax = fig.add_subplot(gs[2, i-6])
		elif i >= 12:
			ax = fig.add_subplot(gs[3, i-12])

		if line_wavelengths[i] < global_params.wav_r_min:
			idx_plot_line = np.where((global_params.wavelength_b > line_wavelengths[i]-100) & (global_params.wavelength_b < line_wavelengths[i]+100))
			ax.plot(global_params.wavelength_b[idx_plot_line], data_blue[idx_plot_line], c=c_blue_spec, alpha=0.7)
			ax.set_ylim(np.min(data_blue[idx_plot_line])-1e-17, np.max(data_blue[idx_plot_line])+1e-17)

		else:
			idx_plot_line = np.where((global_params.wavelength_r > line_wavelengths[i]-100) & (global_params.wavelength_r < line_wavelengths[i]+100))
			ax.plot(global_params.wavelength_r[idx_plot_line], data_red[idx_plot_line], c=c_red_spec, alpha=0.7)
			ax.set_ylim(np.min(data_red[idx_plot_line])-1e-17, np.max(data_red[idx_plot_line])+1e-17)

		ax.axvline(line_wavelengths[i], color='grey', linestyle='--')
		ax.tick_params(labelsize=12)
		#ax.set_ylabel('flux %d' % i)
		#ax.set_xlabel('wvl %d' % i)
		#ax.set_xlim(line_wavelengths[i]-100, line_wavelengths[i]+100)

	fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()

	if save_fig == True:
		plt.savefig(savename, dpi=100, format='pdf', bbox_inches='tight')
	
	if show_plot == True:
		plt.show()	
			
	plt.close()


def single_line_spectra(specs_blue, specs_red, names, redshifts, line_wavs, line_fluxes, line_snrs, savename):
	"""
	Plot all single line spectra in pdf file for overview

	"""

	f = plt.figure()
	gs1 = gridspec.GridSpec(len(specs_blue),1)
	gs1.update(wspace=0.01, hspace=0.02) 

	for i in range(len(specs_blue)):
		ax = f.add_subplot(gs1[i])
		ax.plot(global_params.wavelength_b, specs_blue[i].flux.value, c=c_blue_spec, alpha=0.7)
		ax.plot(global_params.wavelength_r, specs_red[i].flux.value, c=c_red_spec, alpha=0.7)
		ax.axvline(line_wavs[i][0], color='grey', linestyle='--') # should only be 1 line

		#ax.set_ylabel(r'Flux (erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$)', fontsize=12) 
		
		ax.set_ylim(np.min(specs_red[i].flux.value), np.max(specs_red[i].flux.value)+1e-17)

		ax.set_title(str(names[i]) + '   z='+str(np.round(redshifts[i],2)), fontsize=10)
		
		if i == len(specs_blue)-1:
			ax.set_xlabel(r'Observed wavelength ($\AA$)', fontsize=12)

	plt.savefig(savename, dpi=100, format='pdf')#, bbox_inches='tight')
	plt.close()


def plot_emission_line_fit(fit_wav, fit_flux, gauss_fit, idx_bkg_left_min, idx_bkg_left_max, idx_bkg_right_min, idx_bkg_right_max, popt_fit, pcov_fit, lineloc_fit, lineflux_fit, SNR_fit, spurious_bool, real_bool, savename):
	"""
	Plot emission lines fitted with Gaussian

	Keyword arguments:
	fit_wav (array) -- wavelength array of emission line
	fit_flux (array) -- flux array of emission line
	gauss_fit (array) -- gaussian fit result
	idx_bkg_left_min, idx_bkg_left_max, idx_bkg_right_min, idx_bkg_right_max (int) -- indices of lower and upper bounds to determine background
	popt_fit (array) -- fitting result gaussian
	pcov_fit (array) -- errors fitting result gaussian
	lineloc_fit (float) -- wavelength of emission line
	lineflux_fit (float) -- line flux of emission line
	SNR_fit (float) -- SNR of emission line
	spurious_bool (boolean) -- True if emission line is likely spurious
	real_bool (boolean) -- True if emission line is likely to be real
	"""

	textstr = '\n'.join(['Amp = '+str("{:.2e}".format(popt_fit[0])) + " +/- " + str("{:.2e}".format(np.sqrt(pcov_fit[0,0]))), 
						'y0 = '+str("{:.2e}".format(popt_fit[3])) + " +/- " + str("{:.2e}".format(np.sqrt(pcov_fit[3,3]))),
						'Center = '+str(round(popt_fit[1],2)) + " +/- " + str(round(np.sqrt(pcov_fit[1,1]),2)), 
						'Width = '+str(round(popt_fit[2],2)) + " +/- " + str(round(np.sqrt(pcov_fit[2,2]),2)),
						'Line flux ='+str("{:.2e}".format(lineflux_fit)),
						'SNR = '+str(round(SNR_fit,2)),
						'Spurious bool = '+str(spurious_bool),
						'Real bool = '+str(real_bool)])

	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

	Fig, (Ax1) = plt.subplots(1,1, figsize=(6.4,4.8))
	Ax1.plot(fit_wav, fit_flux, color='C0', alpha=0.8)
	Ax1.set_xlim(fit_wav[idx_bkg_left_min]-10, fit_wav[idx_bkg_right_max]+10)
	#Ax1.set_ylim(np.min(fit_flux[idx_bkg_left_min:idx_bkg_right_max]), np.max(fit_flux[idx_bkg_left_min:idx_bkg_right_max])+1e-17)
	
	Ax1.axvline(fit_wav[idx_bkg_left_min], color='grey', linestyle='--')
	Ax1.axvline(fit_wav[idx_bkg_left_max], color='grey', linestyle='--')
	Ax1.axvline(fit_wav[idx_bkg_right_min], color='grey', linestyle='--')
	Ax1.axvline(fit_wav[idx_bkg_right_max], color='grey', linestyle='--')
	plt.plot(fit_wav, gauss_fit, color=c_fit,label='fit')

	Ax1.set_xlabel(r'Observed wavelength ($\AA$)', fontsize=15)
	Ax1.set_ylabel(r'Flux (erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$)', fontsize=15)
	Ax1.text(0.52, 0.62, textstr, transform=Ax1.transAxes, fontsize=11, verticalalignment='bottom', bbox=props)

	Ax1.tick_params(axis='both', which='major', labelsize=12)
	#plt.legend(loc='upper right', prop={'size': 12})
	#plt.title(str(lineloc_fit), fontsize=20)
	plt.tight_layout()
	plt.savefig(savename, dpi=100, format='pdf', bbox_inches='tight')
	plt.close()


def ccf_performace(true_zs, ccf_zs, savename, single_line_bool = False):
	"""
	If there are input redshifts plot the performance of ccf against true redshifts

	Keyword arguments:
	true_zs (array) -- real redshifts of targets
	ccf_zs (array) -- redshifts determined by ccf algo

	"""
	Fig, Ax1 = plt.subplots(1,1, figsize=(6.4,4.8))	
	
	count_single = 0
	count_multiple = 0

	try:
		for i in range(len(single_line_bool)):
			if single_line_bool[i] == True:
				if count_single == 0:
					Ax1.errorbar(ccf_zs[i], true_zs[i], fmt='o', ms=8, color='black', mfc=c_single_line, mew=0.8, alpha=0.8, label='Single line')
				else:
					Ax1.errorbar(ccf_zs[i], true_zs[i], fmt='o', ms=8, color='black', mfc=c_single_line,  alpha=0.8, mew=0.8)
				count_single += 1
			else:
				if count_multiple == 0:
					Ax1.errorbar(ccf_zs[i], true_zs[i], fmt='o', ms=8, color='black', mfc=c_general, alpha=0.8, mew=0.8, label='Multiple lines')
				else:
					Ax1.errorbar(ccf_zs[i], true_zs[i], fmt='o', ms=8, color='black', mfc=c_general, alpha=0.8, mew=0.8)
				count_multiple += 1

	except: # If there is only 1 
		Ax1.errorbar(ccf_zs, true_zs, fmt='o', ms=8, color='black', mfc=c_general, mew=0.8)
	
	Ax1.plot(np.arange(0,6,0.1), np.arange(0,6,0.1), color='grey', linestyle='--')	
	Ax1.set_xlabel(r'$z_{ccf}$', fontsize=20)
	Ax1.set_ylabel(r'$z_{input}$', fontsize=20)
	#Ax1.set_title('Redshift', fontsize=20)
	Ax1.tick_params(axis='both', which='major', labelsize=12)
	plt.legend(loc='upper right', prop={'size': 12})
	plt.tight_layout()
	plt.savefig(savename, dpi=100, format='pdf', bbox_inches='tight')
	plt.close()


def ccf_redshift_output(ccf_zs, savename, single_line_bool = False):
	"""
	Plot histogram of the CCF redshifts 

	Keyword arguments:
	ccf_zs (array) -- redshifts determined by ccf algo

	"""
	Fig, Ax1 = plt.subplots(1,1, figsize=(6.4,4.8))	
	
	Ax1.hist(ccf_zs, bins=np.arange(0,6,0.1), color=c_general, label='All')
	Ax1.hist(np.array(ccf_zs)[single_line_bool], bins=np.arange(0,6,0.1), color=c_single_line, label='Single line')

	Ax1.set_xlabel(r'$z_{ccf}$', fontsize=14)
	Ax1.set_ylabel('Count', fontsize=14)
	#Ax1.set_title('Redshift', fontsize=20)
	Ax1.tick_params(axis='both', which='major', labelsize=12)
	plt.legend(loc='upper right', prop={'size': 12})
	plt.tight_layout()
	plt.savefig(savename, dpi=100, format='pdf', bbox_inches='tight')
	plt.close()


def plot_continuum_sub(wavs, spectrum, subtracted_spectrum, p, savename):

	Fig, (Ax1) = plt.subplots(1,1, figsize=(8,6))	
	
	plt.plot(wavs, spectrum, zorder=1)
	plt.plot(wavs, subtracted_spectrum)
	plt.plot(wavs, p(wavs))
		
	plt.xlabel('Wavelength', fontsize=20)
	plt.ylabel('Flux', fontsize=20)
	plt.title('Continuum subtraction', fontsize=20)
	plt.tight_layout()
	plt.savefig(savename, dpi=100, format='pdf', bbox_inches='tight')
	plt.close()


def plot_2D_hist(x, y, z, xlabel, ylabel, savename):

	xticks_array = np.arange(-1+(1/len(x)), 1.0, 2/len(x))
	yticks_array = np.arange(-1+(1/len(y)), 1.0, 2/len(y))

	y = np.flip(np.array(y))

	fig, ax = plt.subplots(1,1)
	img = ax.imshow(np.array(z), extent=[-1,1, -1,1], cmap='inferno', vmin=0.0, vmax=1.0)
	
	ax.set_xticks(xticks_array)
	ax.set_xticklabels(x)
	ax.set_yticks(yticks_array)
	ax.set_yticklabels(y)
	ax.set_xlabel(xlabel, fontsize=14)
	ax.set_ylabel(ylabel, fontsize=14)

	cbar = plt.colorbar(img)
	cbar.ax.tick_params(labelsize=10)
	cbar.ax.set_ylabel('Fraction detected', fontsize=10)

	plt.savefig(savename, dpi=100, format='pdf', bbox_inches='tight')


def plot_fraction_detected_vs_redshift(variable_list, redshift, fraction_matrix, spurious_matrix, xlabel, ylabel, gap_blue, gap_red, savename):

	fig, ax1 = plt.subplots(1,1)

	if type(variable_list) == float:
		ax1.plot(redshift, fraction_matrix, label='Completeness')
		ax1.plot(redshift, spurious_matrix, color='k', linestyle='--', label='Spurious lines')

	else:
		for i in range(len(variable_list)):
			ax1.plot(redshift, fraction_matrix[:,i], label=str(variable_list[i]))

	ax1.set_xlabel(xlabel, fontsize=14)
	ax1.set_ylabel(ylabel, fontsize=14)
	ax1.tick_params(labelsize=12)

	secax = ax1.secondary_xaxis('top', functions=(lambda x: 1216.*(1+x), lambda x: (x/1216.) -1 ))
	secax.set_xlabel('Wavelength ($\AA$)', fontsize=12)
	secax.tick_params(labelsize=12)

	ax1.fill_between((gap_blue/1216.)-1., -0.1, 1.1, alpha=0.2, color='grey') # chipgaps
	ax1.fill_between((gap_red/1216.)-1., -0.1, 1.1, alpha=0.2, color='grey') # chipgaps

	ax1.set_ylim(-0.05,1.05)
	ax1.legend()
	plt.savefig(savename, dpi=100, format='pdf', bbox_inches='tight')


def plot_detected_curves(std_list, lum_list, fraction_matrix, savename):

	fig, ax = plt.subplots(1,1)

	for i in range(len(std_list)):
		ax.plot(lum_list, fraction_matrix[:,i], label=str(std_list[i]))

	ax.set_xlabel('Luminosity')
	ax.set_ylabel('Fraction detected')
	plt.legend()
	plt.savefig(savename, dpi=100, format='pdf', bbox_inches='tight')


def ccf_test_plot(data_blue, data_red, template_r, template_b, ccf_b_sol, ccf_r_sol, r_blue, r_red, z_lag_b, z_lag_r, z_true, savename):


    Fig, (Ax1, Ax2, Ax3, Ax4) = plt.subplots(4,1, figsize=(10,8))

    Ax1.plot(global_params.wavelength_b, data_blue.flux.value, c=c_blue_spec, alpha=0.8)
    Ax1.plot(global_params.wavelength_r, data_red.flux.value, c=c_red_spec, alpha=0.8)
    Ax1.set_ylim(np.min(data_red.flux.value), np.max(data_red.flux.value)+1e-17)
    #Ax1.text(0.1, 0.8, "Spectrum", transform=Ax1.transAxes, fontsize=16, verticalalignment='bottom')
    Ax1.set_title('Spectrum', fontsize=16)
    #Ax1.set_xlabel(r'Observed wavelength ($\AA$)', fontsize=12)

    Ax2.plot(global_params.wavelength_b, template_b, c=c_blue_spec, alpha=0.8)
    Ax2.plot(global_params.wavelength_r, template_r, c=c_red_spec, alpha=0.8)
    Ax2.set_ylim(np.min(data_red.flux.value), np.max(data_red.flux.value)+1e-17)
    Ax2.set_ylabel(r'                                     Flux (erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$)', fontsize=12)
    Ax2.set_title("Template", fontsize=16)
    #Ax2.text(0.1, 0.8, "Template", transform=Ax2.transAxes, fontsize=16, verticalalignment='bottom')
    Ax2.set_xlabel(r'Observed wavelength ($\AA$)', fontsize=12)

    Ax3.scatter(ccf_b_sol[:,0], ccf_b_sol[:,1], c=c_blue_spec, alpha=0.8)
    Ax3.scatter(ccf_r_sol[:,0], ccf_r_sol[:,1], c=c_red_spec, alpha=0.8)
    Ax3.scatter(z_true, z_lag_b, color='red')
    Ax3.scatter(z_true, z_lag_r, color='red')

   # Ax3.set_title("CCF", fontsize=16)
    Ax3.axvline(z_true, color='red')
    Ax3.set_ylabel(r'z$_{lag}$', fontsize=12)
    Ax3.axhline(0.0, color='black', linestyle='--')
    #Ax3.set_xlabel('Redshift', fontsize=12)

    #Ax4.set_title("CCF", fontsize=16)
    Ax4.scatter(ccf_b_sol[:,0], ccf_b_sol[:,2], c=c_blue_spec, alpha=0.8)
    Ax4.scatter(ccf_r_sol[:,0], ccf_r_sol[:,2], c=c_red_spec, alpha=0.8)

    Ax4.scatter(z_true, r_blue, color='red')
    Ax4.scatter(z_true, r_red, color='red')
    Ax4.axvline(z_true, color='red')
    Ax4.set_xlabel('Redshift', fontsize=12)

    Ax4.set_ylabel('r', fontsize=12)
    Ax4.axhline(0.0, color='black', linestyle='--')

    #plt.tight_layout()
    plt.savefig(savename, dpi=100, format='pdf', bbox_inches='tight')
    plt.close()



