import matplotlib.pyplot as plt
import time
import pdb
import numpy as np
import global_params


def plot_spectrum_blue_red_lines(data_blue, data_red, target_name, line_wavelengths, z_ratio, savename, real_redshift = False):
	"""
	Plot blue and red spectrum and indicate emission lines found

	Keyword arguments:
	data_(blue/red) (array) -- blue and red spectrum
	target_name (str) -- Name of target
	line_wavelengths (array) -- wavelengths of emission lines found in ccf algo
	z_ratio (float) -- redshift estimated by ccf algo
	savename (str) -- filename for saving plot
	real_redshift (float) -- real redshift of target (if available)

	"""

	Fig, (Ax1, Ax2) = plt.subplots(2,1, figsize=(20,14))
	if real_redshift != False:
		Fig.suptitle(target_name+' , z_real='+ str(real_redshift)+', z_ccf='+str(z_ratio), fontsize=20)
	else:
		Fig.suptitle(target_name+', z_ccf='+str(z_ratio), fontsize=20)

	Ax1.plot(global_params.wavelength_b, data_blue)
	Ax1.set_ylim(np.min(data_blue), np.max(data_blue)+1e-17)
	Ax1.set_xlabel('Wavelength (A)', fontsize=15)
	Ax1.set_ylabel('erg/cm^2/s/A', fontsize=15)

	for l in range(len(line_wavelengths)):
		if line_wavelengths[l] < 5772.+10.:
			Ax1.axvline(line_wavelengths[l], color='red', linestyle='--')
		elif line_wavelengths[l] > 5772.+10. and line_wavelengths[l] < 6088.-10.:
			Ax1.axvline(line_wavelengths[l], color='red', linestyle='--')
			Ax2.axvline(line_wavelengths[l], color='red', linestyle='--')			
		else:
			Ax2.axvline(line_wavelengths[l], color='red', linestyle='--')

	Ax2.plot(global_params.wavelength_r, data_red)
	Ax2.set_ylim(np.min(data_red), np.max(data_red)+1e-17)
	Ax2.set_xlabel('Wavelength (A)', fontsize=15)
	Ax2.set_ylabel('erg/cm^2/s/A', fontsize=15)
	plt.tight_layout()
	plt.savefig(savename, dpi=100, format='pdf', bbox_inches='tight')
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
						'spurious bool = '+str(spurious_bool),
						'real bool = '+str(real_bool)])

	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

	Fig, (Ax1) = plt.subplots(1,1, figsize=(8,6))
	Ax1.scatter(fit_wav, fit_flux)
	Ax1.set_xlim(fit_wav[idx_bkg_left_min]-10, fit_wav[idx_bkg_right_max]+10)
	Ax1.set_ylim(np.min(fit_flux[idx_bkg_left_min:idx_bkg_right_max]), np.max(fit_flux[idx_bkg_left_min:idx_bkg_right_max])+1e-17)
	Ax1.axvline(fit_wav[idx_bkg_left_min], color='red', linestyle='--')
	Ax1.axvline(fit_wav[idx_bkg_left_max], color='red', linestyle='--')
	Ax1.axvline(fit_wav[idx_bkg_right_min], color='red', linestyle='--')
	Ax1.axvline(fit_wav[idx_bkg_right_max], color='red', linestyle='--')
	plt.plot(fit_wav, gauss_fit, color='red',label='fit')
	Ax1.set_xlabel('Wavelength (A)', fontsize=15)
	Ax1.set_ylabel('erg/cm^2/s/A', fontsize=15)
	Ax1.text(0.05, 0.5, textstr, transform=Ax1.transAxes, fontsize=14, verticalalignment='bottom', bbox=props)
	plt.legend()
	plt.title(str(lineloc_fit), fontsize=20)
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
	Fig, (Ax1) = plt.subplots(1,1, figsize=(8,6))	
	
	try:
		for i in range(len(single_line_bool)):
			if single_line_bool[i] == True:
				plt.scatter(ccf_zs[i], true_zs[i], color='red')
			else:
				plt.scatter(ccf_zs[i], true_zs[i], color='C0')
	except:
		plt.scatter(ccf_zs, true_zs)
		
	plt.xlabel('z_ccf', fontsize=20)
	plt.ylabel('z_true', fontsize=20)
	plt.title('Redshift', fontsize=20)
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

	xticks_array = np.arange(-1+(1/len(x)), 1.0, 2/len(x))#[-0.75,-0.25,0.25,0.75]
	yticks_array = np.arange(-1+(1/len(y)), 1.0, 2/len(y))

	y = np.flip(np.array(y))

	fig, ax = plt.subplots(1,1)
	img = ax.imshow(np.array(z), extent=[-1,1, -1,1])#extent=[min(x), max(x), max(y), min(y)])
	ax.set_xticks(xticks_array)
	ax.set_xticklabels(x)
	ax.set_yticks(yticks_array)
	ax.set_yticklabels(y)
	ax.set_xlabel(xlabel, fontsize=14)
	ax.set_ylabel(ylabel, fontsize=14)
	fig.colorbar(img)
	plt.savefig(savename, dpi=100, format='pdf', bbox_inches='tight')


def plot_fraction_detected_vs_redshift(variable_list, x_data, fraction_matrix, spurious_matrix, xlabel, ylabel, savename):

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax2 = ax1.twiny()

	if type(variable_list) == float:
		ax1.plot(x_data, fraction_matrix, label='Completeness')
		ax1.plot(x_data, spurious_matrix, color='k', linestyle='--', label='Spurious lines')

	else:
		for i in range(len(variable_list)):
			ax1.plot(x_data, fraction_matrix[:,i], label=str(variable_list[i]))

	ax1.set_xlabel(xlabel, fontsize=14)
	ax1.set_ylabel(ylabel, fontsize=14)

	z_ticks = np.arange(min(x_data), max(x_data), 1.0)
	wav_ticks = (1+z_ticks)*1216.

	ax2.set_xlim(ax1.get_xlim())
	ax2.set_xticks(z_ticks)
	ax2.set_xticklabels(wav_ticks)
	ax2.set_xlabel("Wavelength", fontsize=14)

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


