#!/usr/bin/env python

import os                                                               
import sys

path_tardis_output = os.environ['path_tardis_output']

import pandas as pd
import numpy as np
from astropy import constants as const

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

class Plot_Ti(object):
    
    def __init__(self, element='Ti', show_fig=True, save_fig=False):

        self.element = element
        
        self.velstart_11fe = [13300, 12400, 11300, 10700, 9000, 7850, 6700, 4550]
        self.texp_11fe = [3.7, 5.9, 9.0, 12.1, 16.1, 19.1, 22.4, 28.3]
        self.vstop_11fe = 40000.

        self.velstart_05bl = [8350, 8100, 7600, 6800, 3350]
        self.texp_05bl = [11.0, 12.0, 14.0, 21.8, 29.9]
        self.vstop_05bl = 48000.
        
        self.file_dens_11fe = ('density_11fe_Lgrid_downbranch_es-1.0_ms-1.0'
                               + '_19.1_day.dat')
        self.file_dens_05bl = 'density_05bl_es-0.7_ms-1.0_29.9_day.dat'

        self.file_abun_11fe = 'abundance_11fe_Lgrid_downbranch_19.1_day.dat'
        self.file_abun_05bl = 'abundance_05bl_29.9_day.dat'
        
        
        self.abun_11fe, self.dens_11fe, self.vel_11fe = [], [], []
        self.v_11fe, self.m_11fe = [], []

        self.abun_05bl, self.dens_05bl, self.vel_05bl = [], [], []
        self.v_05bl, self.m_05bl = [], []       
        
        self.show_fig = show_fig
        self.save_fig = save_fig
                        
        self.FIG = plt.figure(figsize=(10,10))
        self.ax = plt.subplot(111)  

        self.fs_label = 26
        self.fs_ticks = 26
        self.fs_legend = 20
        self.fs_text = 22
        self.fs_as = 24
        self.fs_feature = 14
        
        self.run_comparison()
        
    def set_fig_frame(self):

        x_label = r'$\mathrm{Velocity_{photosphere}} \ \mathrm{[km \ s^{-1}}]}$'
        y_label = (r'$\mathrm{M_{'+self.element+'} \ above \ photosphere \ }'
                   + '[M_\odot]$')
        
        self.ax.set_xlabel(x_label,fontsize=self.fs_label)
        self.ax.set_ylabel(y_label,fontsize=self.fs_label)
        self.ax.set_yscale('log')
        self.ax.set_xlim(3000.,15000.)
        #self.ax.set_ylim(1.e-8,1.e-2)       
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs_ticks, pad=8)        
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs_ticks, pad=8)
        self.ax.minorticks_on()
        self.ax.tick_params('both', length=8, width=1, which='major')
        self.ax.tick_params('both', length=4, width=1, which='minor')
        self.ax.xaxis.set_minor_locator(MultipleLocator(1000.))
        self.ax.xaxis.set_major_locator(MultipleLocator(5000.))     

    def get_data(self):
        
        path_dens = (os.path.abspath(os.path.join(path_tardis_output, '..'))
                     + '/INPUT_FILES/DENSITY_FILES/')

        path_abun = (os.path.abspath(os.path.join(path_tardis_output, '..'))
                     + '/INPUT_FILES/STRATIFIED_COMPOSITION_FILES/')
        
        
        #Note densities for 11fe and 05bl have to be scaled differently. To check!
        with open(path_dens + self.file_dens_11fe, 'r') as inp:
            for line in itertools.islice(inp, 2, None, 1):
                column = line.rstrip('\n').split(' ') 
                column = filter(None, column)
                self.vel_11fe.append(float(column[1]))
                self.dens_11fe.append(float(column[2]))
                
        with open(path_dens + self.file_dens_05bl, 'r') as inp:
            for line in itertools.islice(inp, 2, None, 1):
                column = line.rstrip('\n').split(' ') 
                column = filter(None, column)
                self.vel_05bl.append(float(column[1]))
                self.dens_05bl.append(float(column[2]))

        with open(path_abun + self.file_abun_11fe, 'r') as inp:
            for line in itertools.islice(inp, 1, None, 1):
                column = line.rstrip('\n').split(' ') 
                column = filter(None, column)
                self.abun_11fe.append(float(column[-9]))

        with open(path_abun + self.file_abun_05bl, 'r') as inp:
            for line in itertools.islice(inp, 1, None, 1):
                column = line.rstrip('\n').split(' ') 
                column = filter(None, column)
                self.abun_05bl.append(float(column[-9]))
                    
    def compute_11fe_absolute_masses(self):
        
        
        #The density in the 11fe files are given at t=100s, so no need to scale
        #it the volume by the time.
        for k, (time, v_start) in enumerate(zip(
          self.texp_11fe, self.velstart_11fe)):                   
            
            mass_above_photosphere = 0.
                           
            for i, (v_i, v_o, density) in enumerate(zip(
              self.vel_11fe, self.vel_11fe[1:], self.dens_11fe)):                
                           
                if v_i > v_start and v_o <= self.vstop_11fe:
                    mass_above_photosphere += ((4. / 3.) * np.pi
                    * self.abun_11fe[i] * density * 100.**3.
                    * ((v_o * 1.e5)**3. - (v_i * 1.e5)**3.))
          
            mass_above_photosphere = (mass_above_photosphere
                                      / const.M_sun.cgs.value)
                                      
            self.v_11fe.append(v_start)
            self.m_11fe.append(mass_above_photosphere)   

    def compute_05bl_absolute_masses(self):
        
        for k, (time, v_start) in enumerate(zip(
          self.texp_05bl, self.velstart_05bl)):                   
            
            mass_above_photosphere = 0.
            
            for i, (v_i, v_o, density) in enumerate(zip(
              self.vel_05bl, self.vel_05bl[1:], self.dens_05bl)):                
                           
                if v_i > v_start and v_o <= self.vstop_05bl:
                    mass_above_photosphere += ((4. / 3.) * np.pi
                    * self.abun_05bl[i] * density * (time * 24. * 3600.)**3.
                    * ((v_o * 1.e5)**3. - (v_i * 1.e5)**3.))
          
            mass_above_photosphere = (mass_above_photosphere
                                      / const.M_sun.cgs.value)
                                      
            self.v_05bl.append(v_start)
            self.m_05bl.append(mass_above_photosphere)   

    def plot_masses(self):
        """Add lines to axis"""
        
        self.ax.plot(self.v_11fe, self.m_11fe, ls='-', marker='s',
                     color='k', markersize=12., label='SN 2011fe')
        
        self.ax.plot(self.v_05bl, self.m_05bl, ls='-', marker='s',
                     color='r', markersize=12., label='SN 2005bl')
        
        self.ax.legend(frameon=False, fontsize=20., numpoints=1,
                       labelspacing=0.05, loc=1)          
        
    def save_figure(self, extension='pdf', dpi=360):
        if self.save_fig:
            plt.savefig('./../OUTPUT_FILES/FIGURES/Fig_Ti_mass_fraction.'
                        + extension, format=extension, dpi=dpi)
        
    def show_figure(self):
        if self.show_fig:
            plt.show()
                
    def run_comparison(self):
        self.set_fig_frame()
        self.get_data()
        self.compute_11fe_absolute_masses()                                                                 
        self.compute_05bl_absolute_masses()
        self.plot_masses()                                                                  
        plt.tight_layout()
        self.save_figure()
        self.show_figure()  


compare_spectra_object = Plot_Ti(show_fig=True, save_fig=True)
