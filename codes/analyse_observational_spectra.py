#!/usr/bin/env python

import os                                                               
import sys

import pickle
import numpy as np
import pandas as pd   

import tardis.tardistools.compute_features as cp

class Analyse_Observational(object):
    """Takes observed spectra data (usually from WISEREP) and process it
    using the 'compute_features' package under tardistools.
    
    Parameters
    ----------
    ind_dict : ~dict
        Dictionary containing the following keys:
        'filenames', 'host_redshift', 'phase', 't_exp', 'L_bol' and
        'extinction'
    """
    def __init__(self, filename, redshift):                            
        self.filename = filename
        self.redshift = redshift
        self.analyse_observation_spectra()

    def analyse_observation_spectra(self):
        
        path_data = './../INPUT_FILES/observational_spectra/'
        
        wavelength, flux = [], []
        with open(path_data + self.filename, 'r') as inp:
            for line in inp:
                
                #Spectra files are usually tabulated using '\n'. But in
                #some cases '\t' is employed. If condition addresses this.
                column = line.rstrip('\n').split(' ')
                if len(column) == 1:
                    column = line.rstrip('\n').split('\t')
                
                #Store wavelength and flux in variable within the loop
                column = filter(None, column)
                w, f = float(column[0].strip()), float(column[1].strip())
                wavelength.append(w)                    
                flux.append(f)                      

        #Call routines to compute features and uncertainties.
        D = cp.Analyse_Spectra(
          wavelength=np.asarray(wavelength),
          flux=np.asarray(flux),
          redshift=self.redshift, extinction=0.,
          smoothing_window=51, deredshift_and_normalize=True).run_analysis()            

        D = cp.Compute_Uncertainty(
          D=D, smoothing_window=51, N_MC_runs=3000).run_uncertainties() 

        #cp.Plot_Spectra(D, show_fig=False, save_fig=False)                                

        #Create .pkl containg the spectrum and derived qquantities.
        outfile = path_data + self.filename.split('.dat')[0] + '.pkl'
        with open(outfile, 'w') as out_pkl:
            pickle.dump(D, out_pkl, protocol=pickle.HIGHEST_PROTOCOL)
                   
files_11fe = ['2011fe/2011_08_25.dat', '2011fe/2011_08_28.dat',
              '2011fe/2011_08_31.dat', '2011fe/2011_09_03.dat',
              '2011fe/2011_09_07.dat', '2011fe/2011_09_10.dat',
              '2011fe/2011_09_13.dat', '2011fe/2011_09_19.dat']


files_05bl = ['2005bl/2005_04_16.dat', '2005bl/2005_04_17.dat',
              '2005bl/2005_04_19.dat', '2005bl/2005_04_26.dat',
              '2005bl/2005_05_04.dat']

redshift_11fe = 0.
redshift_05bl = 0.02406

#Note, the observed spectra is corrected for redshift, while the synthetic
#spectra is reddened to match the observed extinction.
for inp_file in files_11fe:
   Analyse_Observational(filename=inp_file, redshift=redshift_11fe)

for inp_file in files_05bl:                                                         
   Analyse_Observational(filename=inp_file, redshift=redshift_05bl)



