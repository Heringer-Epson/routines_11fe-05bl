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
        return None
    
    def load_spectra(self):

        """11fe"""
        
        path_data = path_tardis_output 
        
        with open(
          path_data + '11fe_standard_downbranch/velocity_start-10700_loglum-'
          + '9.36_time_explosion-12.1.pkl', 'r') as inp:
            self.pkl_11fe_downbranch = cPickle.load(inp)        
        
        with open(
          path_data + '11fe_standard_macroatom/velocity_start-10700_loglum-'
          + '9.36_time_explosion-12.1.pkl', 'r') as inp:
            self.pkl_11fe_macroatom = cPickle.load(inp)

        """05bl"""  

        with open(
          path_data + '05bl_standard_downbranch/velocity_start-8100_loglum-'
          + '8.63_time_explosion-12.0.pkl', 'r') as inp:
            self.pkl_05bl_downbranch = cPickle.load(inp)
        
        with open(
          path_data + '05bl_standard_macroatom/velocity_start-8100_loglum-'
          + '8.63_time_explosion-12.0.pkl', 'r') as inp:
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
        wavelength_obs = np.asarray(
          self.pkl_11fe_obs['wavelength_raw'].tolist()[0]).astype(np.float)
        
        flux_obs = np.asarray(
          self.pkl_11fe_obs['flux_normalized'].tolist()[0]).astype(np.float)
        
        self.ax.plot(
          wavelength_obs, flux_obs, color='k', linewidth=1., label='SN 2011fe')

        #Downbranch
        wavelength_11fe_downbranch = (np.asarray(
          self.pkl_11fe_downbranch['wavelength_raw'].tolist()[0])
          .astype(np.float))
       
        flux_11fe_downbranch = (np.asarray(
          self.pkl_11fe_downbranch['flux_normalized'].tolist()[0])
          .astype(np.float))
          
        self.ax.plot(wavelength_11fe_downbranch, flux_11fe_downbranch,
          color='b', linewidth=1., label=r'${\rm 2011fe} \, - \, '
          + r'{\tt Downbranch}$')         

        #Macroatom
        wavelength_11fe_macroatom = (np.asarray(
          self.pkl_11fe_macroatom['wavelength_raw'].tolist()[0])
          .astype(np.float))
     
        flux_11fe_macroatom = (np.asarray(
          self.pkl_11fe_macroatom['flux_normalized'].tolist()[0])
          .astype(np.float))
       
        self.ax.plot(wavelength_11fe_macroatom, flux_11fe_macroatom,
          color='b', ls=':', linewidth=1., label=r'${\rm 2011fe} \, - \, '
          + r'{\tt Macroatom}$')

        """05bl"""
        #Observational
        wavelength_obs = np.asarray(
          self.pkl_05bl_obs['wavelength_raw'].tolist()[0]).astype(np.float)
        
        flux_obs = np.asarray(
          self.pkl_05bl_obs['flux_normalized'].tolist()[0]).astype(np.float)
        
        self.ax.plot(
          wavelength_obs, flux_obs + offset, color='r', linewidth=1.,
          label='SN 2005bl')

        #Downbranch
        wavelength_05bl_downbranch = (np.asarray(
          self.pkl_05bl_downbranch['wavelength_raw'].tolist()[0])
          .astype(np.float))
       
        flux_05bl_downbranch = (np.asarray(
          self.pkl_05bl_downbranch['flux_normalized'].tolist()[0])
          .astype(np.float))
          
        self.ax.plot(wavelength_05bl_downbranch, flux_05bl_downbranch + offset,
          color='g', linewidth=1., label=r'${\rm 2005bl} \, - \, '
          + r'{\tt Downbranch}$')         

        #Macroatom
        wavelength_05bl_macroatom = (np.asarray(
          self.pkl_05bl_macroatom['wavelength_raw'].tolist()[0])
          .astype(np.float))
     
        flux_05bl_macroatom = (np.asarray(
          self.pkl_05bl_macroatom['flux_normalized'].tolist()[0])
          .astype(np.float))
       
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
