import numpy as np

def line_ratio_txt_file(savename, wav_file): 
    """
    Create a text file with line ratios from text file of line wavelengths

    Keyword arguments:
    savename (str) - name for saving emission line ratios 
    wav_file (str) - path to .txt file with all emission line wavelengths

    """

    line_names = np.genfromtxt(wav_file, skip_header=1, usecols=(1), dtype='str')
    line_wav = np.genfromtxt(wav_file, skip_header=1, usecols=(2))
    line_strength = np.genfromtxt(wav_file, skip_header=1, usecols=(3))

    line_ratios = []
    line_ratio_names = []
    line_ratio_index = []
    line_wav_num = []
    line_wav_denom = []
    line_strength_list = []

    ratio_file = open(savename, 'w')
    ratio_file.write('# num ratio names wav_num wav_denom line_strength'+'\n')
    for k in range(len(line_wav)): 
      for j in range(len(line_wav)):
          line_ratios.append(line_wav[k]/line_wav[j])
          line_ratio_names.append(line_names[k]+'/'+line_names[j])
          line_ratio_index.append([k,j])
          line_wav_num.append(line_wav[k])
          line_wav_denom.append(line_wav[j])
          line_strength_list.append(np.max([line_strength[k], line_strength[j]]))

    sort_ratios = np.argsort(line_ratios)

    for p in range(len(line_ratios)):
      ratio_file.write(str(p+1)+' '+str(np.array(line_ratio_names)[sort_ratios][p])+' '+str(np.array(line_ratios)[sort_ratios][p])+' '+str(np.array(line_wav_num)[sort_ratios][p])+' '+str(np.array(line_wav_denom)[sort_ratios][p])+' '+ str(np.array(line_strength_list)[sort_ratios][p])+'\n')

    ratio_file.close()


line_ratio_txt_file('Input/line_ratios_sort.txt', 'Input/line_wavelengths.txt') 
line_ratio_txt_file('Input/line_ratios_sort_strong.txt', 'Input/line_wavelengths_strong.txt')

# Could include extra lines in wavelength file
# Put extra weight on ratios between Lya, OII, OIII-5007, and Ha