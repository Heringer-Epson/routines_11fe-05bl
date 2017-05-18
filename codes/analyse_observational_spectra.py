#!/usr/bin/env python

import os                                                               
import sys

import cPickle
import numpy as np

from tardis.tardistools.compute_features import Analyse_Spectra
from tardis.tardistools.compute_features import Compute_Uncertainty

class Analyse_Observational(object):

    def __init__(self, inp_dict):
                                
        self.inp_dict = inp_dict
        self.analyse_observation_spectra()

    def analyse_observation_spectra(self):
        for i, filename in enumerate(self.inp_dict['filenames']):
            D = {}            
            inp = open(path_data+filename, 'r')
            wavelength, flux = [], []
            for line in inp:
                column = line.rstrip('\n').split(' ')
                if len(column) == 1:
                    column = line.rstrip('\n').split('\t')
                column = filter(None, column)
                x, y = float(column[0].strip()), float(column[1].strip())
                wavelength.append(x)                    
                flux.append(y)              
            D = pd.DataFrame({'wavelength_raw': [wavelength], 'flux_raw': [flux]})        
            for key in self.inp_dict.keys():
                D[key] = [self.inp_dict[key][i]]           
            D = analyse(D, smoothing_mode='savgol',verbose=True).run_analysis()       
            D = uncertainty_routine(D, N_MC_runs=3000).run_uncertainties()                
        outfile = filename[:-4]+'.pkl'
        D.to_pickle(path_data+outfile)
        return None

files_11fe = ['observational_spectra/2011fe/2011_08_25.dat', 'observational_spectra/2011fe/2011_08_28.dat', 'observational_spectra/2011fe/2011_08_31.dat', 'observational_spectra/2011fe/2011_09_03.dat', 'observational_spectra/2011fe/2011_09_07.dat', 'observational_spectra/2011fe/2011_09_10.dat', 'observational_spectra/2011fe/2011_09_13.dat', 'observational_spectra/2011fe/2011_09_19.dat']
files_05bl = ['observational_spectra/2005bl/2005_04_16.dat', 'observational_spectra/2005bl/2005_04_17.dat', 'observational_spectra/2005bl/2005_04_19.dat', 'observational_spectra/2005bl/2005_04_26.dat', 'observational_spectra/2005bl/2005_05_04.dat']
redshift_11fe = 0.
redshift_05bl = 0.02406 #For 11fe use redshift = 0 and for 05bl use redshift = 0.02406, for 2007hj use 0.014

for inp_file in files_11fe:
    observational_dict_input = {'filenames': [inp_file], 'host_redshift': [redshift_11fe], 'phase': [0.], 't_exp': [0.], 'L_bol': [0.], 'extinction': [np.nan]}                                                          
    run_observational_analysis = Analyse_Observational(inp_dict=observational_dict_input)

#for inp_file in files_05bl:
#   observational_dict_input = {'filenames': [inp_file], 'host_redshift': [redshift_05bl], 'phase': [0.], 't_exp': [0.], 'L_bol': [0.], 'extinction': [np.nan]}                                                          
#   run_observational_analysis = Analyse_Observational(inp_dict=observational_dict_input)

#For other SNe use, e.g.:
#observational_dict_input = {'filenames': ['observational_spectra/2007hj/2007_09_04.dat'], 'host_redshift': [0.014], 'phase': [0.], 't_exp': [0.], 'L_bol': [0.], 'extinction': [np.nan]}                                                         
#run_observational_analysis = Analyse_Observational(inp_dict=observational_dict_input)

