#!/usr/bin/env python

import os                                                               
import sys

path_tardis_output = os.environ['path_tardis_output']

import matplotlib
import cPickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
import itertools
from matplotlib.ticker import MultipleLocator
                                                
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

class Macroatom_Comparison(object):
    
    def __init__(self, show_fig=True, save_fig=False):
        """Creates a figure where spectra computed using the 'downbranch' and
        'macroatom' modes are compared for both 11fe and 05bl.
        """

        self.show_fig = show_fig
        self.save_fig = save_fig 
       
        self.FIG = plt.figure(figsize=(10,10))
        self.ax = plt.subplot(111)  
        self.pkl_11fe_macroatom, self.pkl_11fe_downbranch = None, None  
        self.pkl_05bl_macroatom, self.pkl_05bl_downbranch = None, None  
        self.pkl_11fe_obs, self.pkl_11fe_obs = None, None  

        self.fs_label = 26
        self.fs_ticks = 26
        self.fs_legend = 20
        self.fs_text = 22
        self.fs_as = 24
        self.fs_feature = 14    
        self.run_comparison()
        
    def set_fig_frame(self):
        
        x_label = r'$\mathrm{rest \ wavelength} \ \mathrm{[\AA}]}$'
        y_label = r'$\mathrm{Absolute \ f}_{\lambda}$'
        
        self.ax.set_xlabel(x_label,fontsize=self.fs_label)
        self.ax.set_ylabel(y_label,fontsize=self.fs_label)
        self.ax.set_xlim(1500.,10000.)
        self.ax.set_ylim(-1.5,3.5)      
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs_ticks, pad=8)      
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs_ticks, pad=8)
        self.ax.minorticks_on()
        self.ax.tick_params('both', length=8, width=1, which='major')
        self.ax.tick_params('both', length=4, width=1, which='minor')
        self.ax.xaxis.set_minor_locator(MultipleLocator(500.))
        self.ax.xaxis.set_major_locator(MultipleLocator(1000.))
        self.ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        self.ax.yaxis.set_major_locator(MultipleLocator(2.))        
        self.ax.tick_params(labelleft='off')                
    
    def load_spectra(self):

        def get_path(event, v, L, lm, texp): 
            case_folder = path_tardis_output + event + '_default_L-scaled/'
            filename = ('velocity_start-' + v + '_loglum-' + L + '_line_'
                    + 'interaction-' + lm + '_time_explosion-' + texp)
            path_sufix = filename + '/' + filename + '.pkl'
            return case_folder + path_sufix
            
        """11fe"""

        fname = get_path('11fe', '10700', '9.362', 'downbranch', '12.1')
        with open(fname, 'r') as inp:
            self.pkl_11fe_downbranch = cPickle.load(inp)        
        
        fname = get_path('11fe', '10700', '9.362', 'macroatom', '12.1')
        with open(fname, 'r') as inp:
            self.pkl_11fe_macroatom = cPickle.load(inp)

        """05bl"""  

        fname = get_path('05bl', '8100', '8.617', 'downbranch', '12.0')
        with open(fname, 'r') as inp:
            self.pkl_05bl_downbranch = cPickle.load(inp)
        
        fname = get_path('05bl', '8100', '8.617', 'macroatom', '12.0')
        with open(fname, 'r') as inp:
            self.pkl_05bl_macroatom = cPickle.load(inp)
        
        """Observational"""

        path_data = './../INPUT_FILES/observational_spectra/'
        with open(path_data + '2011fe/2011_09_03.pkl', 'r') as inp:
            self.pkl_11fe_obs = cPickle.load(inp)                       

        with open(path_data + '2005bl/2005_04_17.pkl', 'r') as inp:
            self.pkl_05bl_obs = cPickle.load(inp)                   

    def plotting(self):

        offset = -1.5

        """11fe"""
        #Observational
        wavelength_obs = self.pkl_11fe_obs['wavelength_corr']
        flux_obs = self.pkl_11fe_obs['flux_normalized']
        
        self.ax.plot(
          wavelength_obs, flux_obs, color='k', linewidth=1., label='SN 2011fe')

        #Downbranch
        wavelength_11fe_downbranch = self.pkl_11fe_downbranch['wavelength_corr']
        flux_11fe_downbranch = self.pkl_11fe_downbranch['flux_normalized']
          
        self.ax.plot(wavelength_11fe_downbranch, flux_11fe_downbranch,
          color='b', linewidth=1., label=r'${\rm 2011fe} \, - \, '
          + r'{\tt Downbranch}$')         

        #Macroatom
        wavelength_11fe_macroatom = self.pkl_11fe_macroatom['wavelength_corr']
        flux_11fe_macroatom = self.pkl_11fe_macroatom['flux_normalized']
       
        self.ax.plot(wavelength_11fe_macroatom, flux_11fe_macroatom,
          color='b', ls=':', linewidth=1., label=r'${\rm 2011fe} \, - \, '
          + r'{\tt Macroatom}$')

        """05bl"""
        #Observational
        wavelength_obs = self.pkl_05bl_obs['wavelength_corr']
        flux_obs = self.pkl_05bl_obs['flux_normalized']
        
        self.ax.plot(
          wavelength_obs, flux_obs + offset, color='r', linewidth=1.,
          label='SN 2005bl')

        #Downbranch
        wavelength_05bl_downbranch = self.pkl_05bl_downbranch['wavelength_corr']
        flux_05bl_downbranch = self.pkl_05bl_downbranch['flux_normalized']
          
        self.ax.plot(wavelength_05bl_downbranch, flux_05bl_downbranch + offset,
          color='g', linewidth=1., label=r'${\rm 2005bl} \, - \, '
          + r'{\tt Downbranch}$')         

        #Macroatom
        wavelength_05bl_macroatom = self.pkl_05bl_macroatom['wavelength_corr']
        flux_05bl_macroatom = self.pkl_05bl_macroatom['flux_normalized']
       
        self.ax.plot(wavelength_05bl_macroatom, flux_05bl_macroatom + offset,
          color='g', ls=':', linewidth=1., label=r'${\rm 2005bl} \, - \, '
          + r'{\tt Macroatom}$')

        self.ax.legend(frameon=False, fontsize=20., numpoints=1, ncol=1,
                       labelspacing=0.05, loc=1)        

    def save_figure(self, extension='pdf', dpi=360):        
        directory = './../OUTPUT_FILES/FIGURES/'
        if self.save_fig:
            plt.savefig(directory + 'Fig_line_mode' + '.' + extension,
                        format=extension, dpi=dpi)
        
    def show_figure(self):
        if self.show_fig:
            plt.show()
                
    def run_comparison(self):
        self.set_fig_frame()
        self.load_spectra()
        plt.tight_layout()
        self.plotting()
        self.save_figure()
        self.show_figure()  

compare_spectra_object = Macroatom_Comparison(show_fig=True, save_fig=True)
