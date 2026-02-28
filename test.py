import numpy as np
import os
from astropy.io import fits
import warnings
import matplotlib.pyplot as plt
from specutils.analysis import template_logwl_resample 
from specutils import Spectrum1D
from specutils.analysis import template_redshift
from astropy import units as u
from specutils.analysis import correlation
from astropy.modeling import models
from astropy.nddata import StdDevUncertainty


def prepare_template(key='AGN'):
    """
    Prepare a template spectrum for AGN and single emission lines

	Keyword arguments:
	template type (str) -- 'AGN' or 'single'

	Return:
	global_params.gal_zip (list) -- fluxes of template 

    """

    if key == 'AGN':

        AGNLines = [770, 1023., 1035, 1216, 1240, 1336, 1402, 1549, 1640, 1663, 1909, 2325, 2798, 
                    3203, 3346, 3426, 3727, 3889, 3970, 3967, 4071, 4102, 4340, 4363, 4686, 
                    4861, 4959, 5007, 6548,6563, 6583]
        AGNstr = [0, 0., 0, 300, 0, 0, 0, 50, 0, 0,0, 0,  0,  
            0, 0, 0, 364, 0, 0, 0, 0, 0, 0, 0, 0, 
            100, 307, 866, 0, 310, 0]

        # AGNstr = [0, 0., 0, 300, 0, 0, 0, 50, 0, 0,0, 0,  0,  
        #             0, 0, 0, 364, 0, 0, 0, 0, 0, 0, 0, 0, 
        #             100, 307, 866, 0, 310, 0]

        AGNLineNames = ['NeVIII', 'Lyb', 'OVI', 'Lya', 'NV', 'CII', 'SiIV', 'CIV', 'HeII', 'OIII-1663', 'CIII', 'CII-2325', 'MgII',
                        'HeII-3203', 'NeV-3346', 'NeV-3426', 'OII', 'HeI', 'Hepsilon', 'NeIII', 'FeV', 'Hdelta', 'Hgamma', 'OIII-4363', 'HeII-4686',
                        'Hb', 'OIII-4959', 'OIII-5007', 'NII-6548', 'Ha', 'NII-6583']

        # # ankur --> no OII line?             
        # AGNLines = [770, 1023, 1035, 1216, 1240, 1336, 1402, 1549, 1640, 1663, 1909, 2326, 2424, 2426, 2470, 2798, 3113,
        #             3203, 3346, 3426, 2727, 2869, 3889, 3970, 3967, 4072, 4102, 4340, 4363, 4686, 4861, 4959, 5007,
        #             6563, 6583]
        # AGNstr = [0, 0, 0, 3100, 154, 37, 163, 364, 318, 72, 177, 92, 41, 49, 41, 78, 11, 14, 20,
        #           69, 364, 82, 19, 21, 26, 16, 22, 24, 8, 20, 100, 307, 866, 310, 0]

        galaxyStr_Norm = AGNstr / np.max(AGNstr) * 1e-17 * 10 # 1e-17*2 adjust height to match spectrum
        gal_zipper = zip(AGNLines, galaxyStr_Norm)
        gal_ziptmp = list(gal_zipper)
        gal_zip = sorted(gal_ziptmp, key=lambda x: x[1], reverse=True)

    elif key == 'single': # Template with only 1 single lines
        AGNLines = [1216]
        AGNstr = [300]
        AGNLineNames = ['Lya']

        galaxyStr_Norm = AGNstr / np.max(AGNstr) * 1e-17 * 10 
        gal_zipper = zip(AGNLines, galaxyStr_Norm)
        gal_ziptmp = list(gal_zipper)
        gal_zip = sorted(gal_ziptmp, key=lambda x: x[1], reverse=True)        

    return gal_zip

def index_data(lambda1, colour):
    """Converts wavelength to it's respective index. Colour key: 'b' for blue; 'r' for red"""
    if colour == 'b':
        i = (lambda1 - wav_b_min) / wav_steps 
    elif colour == 'r':
        i = (lambda1 - wav_r_min) / wav_steps
    return int(i)

def makeGaus(mean, x, ht, std):
    y0 = np.exp(-0.5 * (((x - mean) / std) ** 2))
    y = y0 * ht
    return y

def shift_template(template_flux, wavs,  z, colour):
	"""
	Shift the template with redshift 

	Keyword arguments:
	wavs (array) -- wavelengths of template
	template_flux (array) -- fluxes of template

	Return:
	ySum (array) -- array with gaussians around the emission lines (shifted with z+1) 
	"""
	ySum = 0

	if colour == 'r':
		for everyLine in template_flux:
			if wav_r_min < everyLine[0] * (z + 1) < wav_r_max:
				shifted = index_data(everyLine[0] * (z + 1), 'r')
				template = makeGaus(shifted, wavs, everyLine[1], std=100) # STD value is *4, because wavelength steps are 0.25 A. Change std value here to fit broader or more narrow template lines
				ySum += template # an array gaussians around the emission lines (shifted with z+1)    
		return ySum

	if colour == 'b':
		for everyLine in template_flux:
			if wav_b_min < everyLine[0] * (z + 1) < wav_b_max:
				shifted = index_data(everyLine[0] * (z + 1), 'b')
				template = makeGaus(shifted, wavs, everyLine[1], std=100)
				ySum += template

		return ySum




red_spectra = '/Users/anniekgloudemans/Documents/AA_PhD/Projects/WL_high_z/Data/simulated_data/L1/OB/3130/stacked/stacked_1002813.fit'#/Users/anniekgloudemans/Documents/AA_PhD/Projects/WL_high_z/Data/simulated_data/L1/OB/4376/stacked/stacked_1003809.fit'
f = fits.open(red_spectra)  

# Test case
flux_spec = (f[1].data*f[5].data)[1] * u.Unit('erg cm-2 s-1 AA-1') 
std_spec = StdDevUncertainty((f[3].data*f[5].data)[1]*u.Unit('erg cm-2 s-1 AA-1'))
f.close() 

wavelength_r = np.arange(15289)*0.25 + 5772

lamb = wavelength_r * u.AA
uncertainty = StdDevUncertainty(0.001*1e-17*np.ones(len(lamb))*u.Unit('erg cm-2 s-1 AA-1'))
spec = Spectrum1D(spectral_axis=lamb, flux=flux_spec, uncertainty=uncertainty)#std_spec) SOMETHING WRONG WITH STD_SPEC


# Make template
wav_steps = 0.25
wav_b_min = 3676.0
wav_b_max = 6088.25
wav_r_min = 5772.0
wav_r_max = 9594.25
gal_zip = prepare_template(key='AGN') 
xaxis_r = np.arange(0, 15289,1)
tem_r = shift_template(gal_zip, xaxis_r, 0.8421052631578947, 'r')
tem_spec = Spectrum1D(spectral_axis=wavelength_r*u.AA, flux=tem_r * u.Unit('erg cm-2 s-1 AA-1'))#, uncertainty=uncertainty)


rebin_val_test, rebin_val_temr_z_test = template_logwl_resample(spec, tem_spec)
ospec = rebin_val_test
tspec = rebin_val_temr_z_test

corr, lag = correlation.template_correlate(ospec, tspec)

# #print(corr, lag)
# print(np.argmax(corr), np.argmin(abs(lag)))
# #print(ospec.spectral_axis[np.argmax(corr)])
# print(len(ospec.spectral_axis), len(corr), len(lag))

# plt.plot(ospec.spectral_axis, ospec.flux)
# plt.plot(tspec.spectral_axis, tspec.flux)
# plt.show()

r = ((np.ndarray.max(corr) * np.ndarray.mean(corr)) / np.std(corr)) * 1e32
print('r', r)
pix_width = (np.log(max(ospec.spectral_axis.value)) - np.log(min(ospec.spectral_axis.value)))/len(ospec.spectral_axis)
lag_r = (np.argmax(corr) - ((len(corr) + 0) / 2))
z_lag = round((10 ** (pix_width * lag_r) * (1 + (30 / 3e5)) - 1), 5)
print(z_lag)



# # Make full template
# wav_steps = 0.25
# wav_b_min = 0.0#3676.0
# wav_b_max = 6088.25
# wav_r_min = 0.0#5772.0
# wav_r_max = 9594.25
# gal_zip = prepare_template(key='AGN') 
# xaxis_r = np.arange(0, 15289*2,1)

# tem_r_rest = shift_template(gal_zip, xaxis_r, 0.0, 'r')
# wavelength_r_rest = np.arange(15289*2)*0.25
# tem_spec_rest = Spectrum1D(spectral_axis=wavelength_r_rest*u.AA, flux=tem_r_rest * u.Unit('erg cm-2 s-1 AA-1'))#, uncertainty=unde)


# plt.plot(spec.spectral_axis, spec.flux)
# plt.plot(tem_spec_rest.spectral_axis, tem_spec_rest.flux)
# plt.show()

# Works but takes very long
# test = template_redshift(spec, tem_spec_rest, np.arange(0.73,1.4,0.1))
# print(test)

# Test new rebinning and correlation



