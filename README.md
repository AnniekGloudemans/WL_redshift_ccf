# Identify redshift and emission lines in WL spectra using CCF

** Testing phase **

This code estimates the redshift and emission line fluxes for WEAVE-LOFAR spectra using a cross-correlation method. It has been developed with the main aim to identify single (faint) emission line sources.   

An example on how to run this is given in run_ccf_example.ipynb (runs on Python 3.7.4). 

Note that this code is still in the testing phase and there will be many improvements/adjustments to come. Please let me know if you run into any issue or if you would like me to add a certain feature. 

It is not yet optimised and therefore runs quite slow (about 1 spectrum per min). Running through an OB (with ~900 spectra) might take a full night. 
