#!/usr/bin/env python

import os                                                               
import sys
import time

path_tardis_output = os.environ['path_tardis_output']

import matplotlib
import cPickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
from matplotlib import colors
from itertools import cycle

import colormaps as cmaps
                                                
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

def normalize_spectrum(w, f):
    aux_flux = np.asarray(f).astype(np.float)                    
    window_condition = ((w >= 4000.) & (w <= 9000.))             
    flux_window = aux_flux[window_condition]
    normalization_factor = np.mean(flux_window)    
    return f / normalization_factor

class Analyse_Vphoto(object):
    """TBW"""
    
    def __init__(self, L_folder, show_fig=True, save_fig=False):

        self.show_fig = show_fig
        self.save_fig = save_fig
        self.L_folder = L_folder
        self.list_pkl = []
        self.pkl_05bl = None 

        self.FIG = plt.figure(figsize=(16,10))
        self.ax1 = plt.subplot(211) 
        self.ax2 = plt.subplot(212)

        self.fs_label = 26
        self.fs_ticks = 26
        self.fs_legend = 20
        self.fs_text = 22
        self.fs_as = 24
        self.fs_feature = 14
    
        self.run_vphoto()
        
    def set_fig_frame(self):
        """Define the configuration of the figure axes."""
        
        top_x_label = r'$\mathrm{rest \ wavelength} \ \mathrm{[\AA}]}$'
        top_y_label = r'$\mathrm{Relative \ f}_{\lambda}$'
     
        bot_x_label = r'$\mathrm{v} \ \mathrm{[10^3 \, km \, s^{-1}}]}$'
        bot_y_label = r'$\mathrm{T \, [K]}$'
        
        self.ax1.set_xlabel(top_x_label, fontsize=self.fs_label)
        self.ax1.set_ylabel(top_y_label, fontsize=self.fs_label)
        self.ax1.set_xlim(1500.,10000.)
        self.ax1.set_ylim(0.,3.)
        self.ax1.tick_params(axis='y', which='major', labelsize=self.fs_ticks, pad=8)       
        self.ax1.tick_params(axis='x', which='major', labelsize=self.fs_ticks, pad=8)
        self.ax1.minorticks_on()
        self.ax1.tick_params('both', length=8, width=1, which='major')
        self.ax1.tick_params('both', length=4, width=1, which='minor')
        self.ax1.xaxis.set_minor_locator(MultipleLocator(500.))
        self.ax1.xaxis.set_major_locator(MultipleLocator(1000.))
        self.ax1.yaxis.set_minor_locator(MultipleLocator(0.5))
        self.ax1.yaxis.set_major_locator(MultipleLocator(1.))       

        self.ax2.set_xlabel(bot_x_label, fontsize=self.fs_label)
        self.ax2.set_ylabel(bot_y_label, fontsize=self.fs_label)
        self.ax2.set_xlim(3.,20.)
        self.ax2.set_ylim(1000.,20000.)      
        self.ax2.tick_params(axis='y', which='major', labelsize=self.fs_ticks, pad=8)       
        self.ax2.tick_params(axis='x', which='major', labelsize=self.fs_ticks, pad=8)
        self.ax2.minorticks_on()
        self.ax2.tick_params('both', length=8, width=1, which='major')
        self.ax2.tick_params('both', length=4, width=1, which='minor')
        self.ax2.xaxis.set_minor_locator(MultipleLocator(1.))
        self.ax2.xaxis.set_major_locator(MultipleLocator(5.))
        self.ax2.yaxis.set_minor_locator(MultipleLocator(1000.))
        self.ax2.yaxis.set_major_locator(MultipleLocator(5000.))       
    
    def load_spectra(self):
        """Load data files."""
        
        #Sequence in vphoto.
        path_data = path_tardis_output + '11fe_vphoto_' + self.L_folder + '/'
        for filename in os.listdir(path_data):
            if filename.split('.')[-1] == 'pkl':
                with open(path_data + filename, 'r') as inp:
                    self.list_pkl.append(cPickle.load(inp))
                    
        #05bl spectrum that is matched with the scaled temperature.
        path_data = path_tardis_output + '05bl_standard_downbranch/'
        L_suffix = self.L_folder.split('_')[0]
        if L_suffix == 'premax':
            filename = 'velocity_start-8100_loglum-8.63_time_explosion-12.0.pkl' 
        elif L_suffix == 'max':
            filename = 'velocity_start-6600_loglum-8.89_time_explosion-21.8.pkl'      
        elif L_suffix == 'postmax':
            filename = 'velocity_start-3300_loglum-8.59_time_explosion-29.9.pkl'        
                
        with open(path_data + filename, 'r') as inp: 
            self.pkl_05bl = cPickle.load(inp)                

    def plotting(self):
        
        #Plot sequence in vphoto.
        color = cycle(['r', 'b', 'g', 'm', 'c', 'y'])
       
        for i, pkl in enumerate(self.list_pkl):                           
            
            #Get values for plotting
            wavelength = (np.asarray(pkl['wavelength_raw'].tolist()[0])
                          .astype(np.float))
            
            if 'flux_normalized' not in pkl.keys():
                flux_raw = (np.asarray(pkl['flux_raw'].tolist()[0])
                             .astype(np.float))
                flux_norm = normalize_spectrum(wavelength, flux_raw)
            else:
                flux_norm = (np.asarray(pkl['flux_normalized'].tolist()[0])
                             .astype(np.float))                        
            
            
            T_inner = (np.asarray(pkl['t_inner'].tolist()[0])
                          .astype(np.float))
            T_prof = (np.asarray(pkl['t_rad'].tolist()[0])
                          .astype(np.float))
            v_prof = (np.asarray(pkl['v_inner'].tolist()[0])
                          .astype(np.float)) / 1.e8                        
            
            #Draw
            c = next(color)
            self.ax1.plot(wavelength, flux_norm, color=c)
            self.ax2.plot(v_prof, T_prof, color=c)
            self.ax2.plot(v_prof[0], T_inner, marker='*', markersize=14., color=c)
        
            #print 'v', v_prof[0], 'dilution factor', pkl['w'].tolist()[0][0]
        
        #Plot the 05bl spectrum whose scaled 11fe spectrum matches.
                
        #Get values for plotting
        wavelength = (np.asarray(self.pkl_05bl['wavelength_raw'].tolist()[0])
                      .astype(np.float))
        flux_norm = (np.asarray(self.pkl_05bl['flux_normalized'].tolist()[0])
                     .astype(np.float))                

        T_inner = (np.asarray(self.pkl_05bl['t_inner'].tolist()[0])
                      .astype(np.float))
        T_prof = (np.asarray(self.pkl_05bl['t_rad'].tolist()[0])
                      .astype(np.float))
        v_prof = (np.asarray(self.pkl_05bl['v_inner'].tolist()[0])
                      .astype(np.float)) / 1.e8             

        #Draw
        c = 'k'
        self.ax1.plot(wavelength, flux_norm, color=c, label=self.L_folder
                      + '$_{\mathrm{11fe}}$' + ' SN 2005bl')
        self.ax2.plot(v_prof, T_prof, color=c)
        self.ax2.plot(v_prof[0], T_inner, marker='*', markersize=14., color=c)
       
        self.ax1.legend(frameon=False, fontsize=self.fs_legend, numpoints=1,
                        labelspacing=0.05, loc=1)         
                
        plt.tight_layout()
            
    def save_figure(self, ext='pdf', dpi=360):
        directory = './../OUTPUT_FILES/FIGURES/'
        if self.save_fig:
            filename = directory + 'Fig_vphoto_' + self.L_folder + '.' + ext
            plt.savefig(filename, format=ext, dpi=dpi)
        
    def show_figure(self):
        if self.show_fig:
            plt.show()
                
    def run_vphoto(self):
        self.set_fig_frame()
        self.load_spectra()
        self.plotting()
        self.save_figure()
        self.show_figure()  

compare_spectra_object = Analyse_Vphoto(L_folder='postmax_1L', show_fig=True, 
                                        save_fig=True)
                                        
                                        
