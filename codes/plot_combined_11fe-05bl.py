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

def mean_flux(w, f, w_min, w_max):
    window_condition = ((w > w_min) & (w < w_max))  
    return np.mean(f[window_condition])

class Compare_Spectra(object):
    
    def __init__(self, line_mode='downbranch', show_fig=True, save_fig=False):
        
        self.list_pkl, self.label = [], []    

        self.line_mode = line_mode
        self.show_fig = show_fig
        self.save_fig = save_fig
               
        self.FIG = plt.figure(figsize=(20,22))
        self.ax1 = plt.subplot(121)  
        self.ax2 = plt.subplot(122, sharey=self.ax1)
        
        self.left_top = []
        self.left_mid = []
        self.left_bot = []       
        self.right_top = []
        self.right_mid = []
        self.right_bot = []

        self.fs_label = 26
        self.fs_ticks = 26
        self.fs_legend = 20
        self.fs_text = 22
        self.fs_as = 24
        self.fs_feature = 14
        
        self.run_comparison()
        
    def set_fig_frame(self):
        """Define the configuration of the figure axes."""
        
        left_x_label = r'$\mathrm{rest \ wavelength} \ \mathrm{[\AA}]}$'
        right_x_label = r'$\mathrm{rest \ wavelength} \ \mathrm{[\AA}]}$'
        left_y_label = r'$\mathrm{Relative \ f}_{\lambda}$'
        
        self.ax1.set_xlabel(left_x_label, fontsize=self.fs_label)
        self.ax1.set_ylabel(left_y_label, fontsize=self.fs_label)
        self.ax1.set_xlim(1500.,10000.)
        self.ax1.set_ylim(-37, 5.)      
        self.ax1.tick_params(axis='y', which='major', labelsize=self.fs_ticks, pad=8)       
        self.ax1.tick_params(axis='x', which='major', labelsize=self.fs_ticks, pad=8)
        self.ax1.minorticks_on()
        self.ax1.tick_params('both', length=8, width=1, which='major')
        self.ax1.tick_params('both', length=4, width=1, which='minor')
        self.ax1.xaxis.set_minor_locator(MultipleLocator(500.))
        self.ax1.xaxis.set_major_locator(MultipleLocator(1000.))
        self.ax1.yaxis.set_minor_locator(MultipleLocator(1.))
        self.ax1.yaxis.set_major_locator(MultipleLocator(5.))       
        self.ax1.tick_params(labelleft='off')       

        self.ax2.set_xlabel(right_x_label, fontsize=self.fs_label)
        self.ax2.set_xlim(1500.,10000.)
        self.ax2.set_ylim(-37., 5.)      
        self.ax2.tick_params(axis='y', which='major', labelsize=self.fs_ticks, pad=8)       
        self.ax2.tick_params(axis='x', which='major', labelsize=self.fs_ticks, pad=8)
        self.ax2.minorticks_on()
        self.ax2.tick_params('both', length=8, width=1, which='major')
        self.ax2.tick_params('both', length=4, width=1, which='minor')
        self.ax2.xaxis.set_minor_locator(MultipleLocator(500.))
        self.ax2.xaxis.set_major_locator(MultipleLocator(1000.))
        self.ax2.yaxis.set_minor_locator(MultipleLocator(1.))
        self.ax2.yaxis.set_major_locator(MultipleLocator(5.))       
        self.ax2.tick_params(labelleft='off')   
    
    def load_spectra(self):

        #Define path variables.
        path_syn_11fe = (path_tardis_output + '11fe_standard_'
                         + self.line_mode + '/')                        
        path_syn_05bl = (path_tardis_output + '05bl_standard_'
                         + self.line_mode + '/')                       
        path_obs_11fe = './../INPUT_FILES/observational_spectra/2011fe/'
        path_obs_05bl = './../INPUT_FILES/observational_spectra/2005bl/'
                                                  
        """11fe -> 05bl spectra"""

        #t_exp = 12.1 - standard
        aux_list = [path_obs_11fe + '2011_09_03.pkl',
                    path_syn_11fe + 'velocity_start-10700_loglum-'
                      + '9.36_time_explosion-12.1.pkl',                                                            
                    path_syn_11fe + 'velocity_start-10700_loglum-'
                      + '9.06_time_explosion-12.1.pkl',                                                            
                    path_syn_11fe + 'velocity_start-10700_loglum-'
                      + '8.89_time_explosion-12.1.pkl',                                                            
                    path_obs_05bl + '2005_04_17.pkl',
                    path_syn_11fe + 'velocity_start-10700_loglum-'
                      + '8.76_time_explosion-12.1.pkl',                                                            
                    path_obs_05bl + '2005_04_17.pkl',
                    path_syn_05bl + 'velocity_start-8100_loglum-'
                      + '8.63_time_explosion-12.0.pkl',
                    path_obs_05bl + '2005_04_17.pkl']                    
        
        for pkl_path in aux_list:
            with open(pkl_path, 'r') as inp:
                self.left_top.append(cPickle.load(inp))

        #t_exp = 19.1 - standard.
        aux_list = [path_obs_11fe + '2011_09_10.pkl',
                    path_syn_11fe + 'velocity_start-7850_loglum-'
                      + '9.54_time_explosion-19.1.pkl',                                                            
                    path_syn_11fe + 'velocity_start-7850_loglum-'
                      + '9.24_time_explosion-19.1.pkl',                                                            
                    path_syn_11fe + 'velocity_start-7850_loglum-'
                      + '9.06_time_explosion-19.1.pkl',                                                            
                    path_obs_05bl + '2005_04_26.pkl',
                    path_syn_11fe + 'velocity_start-7850_loglum-'
                      + '8.94_time_explosion-19.1.pkl',                                                            
                    path_obs_05bl + '2005_04_26.pkl',
                    path_syn_05bl + 'velocity_start-6600_loglum-'
                      + '8.89_time_explosion-21.8.pkl',
                    path_obs_05bl + '2005_04_26.pkl']        
                
        for pkl_path in aux_list:
            with open(pkl_path, 'r') as inp:
                self.left_mid.append(cPickle.load(inp))

        #t_exp = 28.3 - standard.  
        aux_list = [path_obs_11fe + '2011_09_19.pkl',
                    path_syn_11fe + 'velocity_start-4550_loglum-'
                      + '9.36_time_explosion-28.3.pkl',                                                            
                    path_syn_11fe + 'velocity_start-4550_loglum-'
                      + '9.06_time_explosion-28.3.pkl',                                                            
                    path_syn_11fe + 'velocity_start-4550_loglum-'
                      + '8.89_time_explosion-28.3.pkl',                                                            
                    path_obs_05bl + '2005_05_04.pkl',
                    path_syn_11fe + 'velocity_start-4550_loglum-'
                      + '8.76_time_explosion-28.3.pkl',                                                            
                    path_obs_05bl + '2005_05_04.pkl',
                    path_syn_05bl + 'velocity_start-3300_loglum-'
                      + '8.59_time_explosion-29.9.pkl',
                    path_obs_05bl + '2005_05_04.pkl']        
                
        for pkl_path in aux_list:
            with open(pkl_path, 'r') as inp:
                self.left_bot.append(cPickle.load(inp))    

        '''
        05bl -> 11fe spectra
        '''

        #t_exp = 11
        aux_list = [path_obs_11fe + '2011_09_03.pkl',
                    path_syn_11fe + 'velocity_start-10700_loglum-'
                      + '9.36_time_explosion-12.1.pkl',          
                    path_obs_11fe + '2011_09_03.pkl',                                                                                                                                            
                    path_syn_05bl + 'velocity_start-8100_loglum-'
                      + '9.23_time_explosion-12.0.pkl',
                    path_obs_11fe + '2011_09_03.pkl',                                                                                                                                           
                    path_syn_05bl + 'velocity_start-8100_loglum-'
                      + '9.10_time_explosion-12.0.pkl',      
                    path_syn_05bl + 'velocity_start-8100_loglum-'
                      + '8.93_time_explosion-12.0.pkl',      
                    path_syn_05bl + 'velocity_start-8100_loglum-'
                      + '8.63_time_explosion-12.0.pkl',          
                    path_obs_05bl + '2005_04_17.pkl']        
                
        for pkl_path in aux_list:
            with open(pkl_path, 'r') as inp:
                self.right_top.append(cPickle.load(inp)) 

        #t_exp = 21.8
        aux_list = [path_obs_11fe + '2011_09_10.pkl',
                    path_syn_11fe + 'velocity_start-7850_loglum-'
                      + '9.54_time_explosion-19.1.pkl',          
                    path_obs_11fe + '2011_09_10.pkl',                                                                                                                                            
                    path_syn_05bl + 'velocity_start-6600_loglum-'
                      + '9.48_time_explosion-21.8.pkl',
                    path_obs_11fe + '2011_09_10.pkl',                                                                                                                                           
                    path_syn_05bl + 'velocity_start-6600_loglum-'
                      + '9.37_time_explosion-21.8.pkl',      
                    path_syn_05bl + 'velocity_start-6600_loglum-'
                      + '9.19_time_explosion-21.8.pkl',      
                    path_syn_05bl + 'velocity_start-6600_loglum-'
                      + '8.89_time_explosion-21.8.pkl',          
                    path_obs_05bl + '2005_04_26.pkl']        
                
        for pkl_path in aux_list:
            with open(pkl_path, 'r') as inp:
                self.right_mid.append(cPickle.load(inp)) 

        #t_exp = 28.3 - standard.
        aux_list = [path_obs_11fe + '2011_09_19.pkl',
                    path_syn_11fe + 'velocity_start-4550_loglum-'
                      + '9.36_time_explosion-28.3.pkl',          
                    path_obs_11fe + '2011_09_19.pkl',                                                                                                                                            
                    path_syn_05bl + 'velocity_start-3300_loglum-'
                      + '9.20_time_explosion-29.9.pkl',
                    path_obs_11fe + '2011_09_19.pkl',                                                                                                                                           
                    path_syn_05bl + 'velocity_start-3300_loglum-'
                      + '9.01_time_explosion-29.9.pkl',      
                    path_syn_05bl + 'velocity_start-3300_loglum-'
                      + '8.90_time_explosion-29.9.pkl',      
                    path_syn_05bl + 'velocity_start-3300_loglum-'
                      + '8.59_time_explosion-29.9.pkl',          
                    path_obs_05bl + '2005_05_04.pkl']        
                
        for pkl_path in aux_list:
            with open(pkl_path, 'r') as inp:
                self.right_bot.append(cPickle.load(inp)) 
   
    def plotting_11fe_to_05bl(self):
        """Plot the left panel"""
       
        offset_lvl = -2.
        offset_text = -0.5
        text_date_wavelength = 5700.
        text_lum_wavelength = 8600.
        text_set_wavelength = 2000.
        
        text_date_up = [
          r'$t_{\mathrm{11fe}}=\mathrm{12.1 \ d,} \ '
          + 'L_{\mathrm{11fe}}=\mathrm{2.3 \ 10^9 L_{\odot}}$',
          r'$t_{\mathrm{11fe}}=\mathrm{19.1 \ d,} \ '
          + 'L_{\mathrm{11fe}}=\mathrm{3.5 \ 10^9 L_{\odot}}$',
          r'$t_{\mathrm{11fe}}=\mathrm{28.3 \ d,} \ '
          + 'L_{\mathrm{11fe}}=\mathrm{2.3 \ 10^9 L_{\odot}}$']
          
        text_date_down = [
          r'$t_{\mathrm{05bl}}=\mathrm{12 \ d,} \ '
          + 'L_{\mathrm{05bl}}=\mathrm{0.19} L_{\mathrm{11fe}}$',
          r'$t_{\mathrm{05bl}}=\mathrm{21.8 \ d,} \ '
          + 'L_{\mathrm{05bl}}=\mathrm{0.22} L_{\mathrm{11fe}}$',
          r'$t_{\mathrm{05bl}}=\mathrm{29.9 \ d,} \ '
          + 'L_{\mathrm{05bl}}=\mathrm{0.17} L_{\mathrm{11fe}}$']
          
        text_lum = cycle([
          r'$L=L_{\mathrm{11fe}}$',
          r'$L=\mathrm{0.5} L_{\mathrm{11fe}}$',
          r'$L=\mathrm{0.33} L_{\mathrm{11fe}}$',
          r'$L=\mathrm{0.25} L_{\mathrm{11fe}}$']) 

        text_set = cycle([
          r'${\rm \mathbf{a}}$',
          r'${\rm \mathbf{b}}$',
          r'${\rm \mathbf{c}}$'])            
        
        pkl_offset_scaling = [0., 7., 14]
        offset = [0., 0., 1.*offset_lvl, 2.*offset_lvl, 2.*offset_lvl,
                  3.*offset_lvl, 3.*offset_lvl, 4.*offset_lvl, 4.*offset_lvl]
        color = ['k', 'b', 'b', 'b', 'r', 'b', 'r', 'g', 'r']
        alpha = [1., 1., .8, .6, 1., .4, 1., 1., 1.]
        
        for l, pkl_set in enumerate([self.left_top, self.left_mid, self.left_bot]):
            for i, pkl in enumerate(pkl_set):  

                #Plot spectrum
                wavelength = np.asarray(
                  pkl['wavelength_raw'].tolist()[0]).astype(np.float)
                flux_normalized = (np.asarray(
                  pkl['flux_normalized'].tolist()[0]).astype(np.float)
                  + offset[i] + pkl_offset_scaling[l] * offset_lvl)         
                self.ax1.plot(wavelength, flux_normalized, color=color[i],
                              linewidth=2., alpha=alpha[i])

                #Plot photosphere temperature text
                try:
                    text_lvl = mean_flux(wavelength, flux_normalized, 1500., 2500.)
                    self.ax1.text(
                      1700, text_lvl + offset_text,
                      r'$\mathrm{' + str(format(pkl['t_inner'][0] ,'.0f')) + ' \ K}$',
                      fontsize=20., horizontalalignment='left', color=color[i])
                except:
                    pass   
                    
                #Plot date text
                if i == 0:
                    text_lvl = mean_flux(wavelength, flux_normalized, 5500., 6000.)
                    self.ax1.text(
                      text_date_wavelength, text_lvl - offset_text,
                      text_date_up[l], fontsize=20., horizontalalignment='left')                                 
                if i == 8:
                    text_lvl = mean_flux(wavelength, flux_normalized, 8200., 8300.)
                    self.ax1.text(
                      text_date_wavelength, text_lvl + 1.5 * offset_text,
                      text_date_down[l], fontsize=20., horizontalalignment='left')          

                #Plot luminosity text
                if (l == 1) and (i in [1, 2, 3, 5]):
                    text_lvl = mean_flux(wavelength, flux_normalized, 9400., 9600.)
                    self.ax1.text(
                      text_lum_wavelength, text_lvl + offset_text, next(text_lum),
                      fontsize=20., horizontalalignment='left', color='b')
       
                #Add set text
                if i == 1:
                    text_lvl = mean_flux(wavelength, flux_normalized, 1800., 2200.)
                    self.ax1.text(text_set_wavelength, text_lvl - 4. * offset_text,
                    next(text_set), fontsize=20., horizontalalignment='left', color='k')  
                                                 
    def plotting_05bl_to_11fe(self):
        """Plot the right panel"""
        
        offset_lvl = -2.
        offset_text = -0.5
        text_date_wavelength = 5700.
        text_lum_wavelength = 8600.
        text_set_wavelength = 2000.

        text_date_up = [
          r'$t_{\mathrm{11fe}}=\mathrm{12.1 \ d,} \ '
          + 'L_{\mathrm{11fe}}=\mathrm{5.4} L_{\mathrm{05bl}}$',
          r'$t_{\mathrm{11fe}}=\mathrm{19.1 \ d,} \ '
          + 'L_{\mathrm{11fe}}=\mathrm{4.5} L_{\mathrm{05bl}}$',
          r'$t_{\mathrm{11fe}}=\mathrm{28.3 \ d,} \ '
          + 'L_{\mathrm{11fe}}=\mathrm{5.9} L_{\mathrm{05bl}}$']
          
        text_date_down = [
          r'$t_{\mathrm{05bl}}=\mathrm{12 \ d,} \ '
          + 'L_{\mathrm{05bl}}=\mathrm{3.39 \ 10^8 L_{\odot}}$',
          r'$t_{\mathrm{05bl}}=\mathrm{21.8 \ d,} \ '
          + 'L_{\mathrm{05bl}}=\mathrm{7.76 \ 10^8 L_{\odot}}$',
          r'$t_{\mathrm{05bl}}=\mathrm{29.9 \ d,} \ '
          + 'L_{\mathrm{05bl}}=\mathrm{3.89 \ 10^8 L_{\odot}}$']

        text_lum = cycle([
          r'$L=\mathrm{4} L_{\mathrm{05bl}}$',
          r'$L=\mathrm{3} L_{\mathrm{05bl}}$',
          r'$L=\mathrm{2} L_{\mathrm{05bl}}$',
          r'$L=L_{\mathrm{05bl}}$']) 
          
        text_set = cycle([
          r'${\rm \mathbf{d}}$',
          r'${\rm \mathbf{e}}$',
          r'${\rm \mathbf{f}}$'])  
          
        pkl_offset_scaling = [0., 7., 14]
        offset = [0., 0., 1.*offset_lvl, 1.*offset_lvl, 2.*offset_lvl,
                  2.*offset_lvl, 3.*offset_lvl, 4.*offset_lvl, 4.*offset_lvl]
        color = ['k', 'b', 'k', 'g', 'k', 'g', 'g', 'g', 'r']
        alpha = [1., 1., 1., .4, 1., .6, .8, 1., 1.]
        
        for l, pkl_set in enumerate([self.right_top, self.right_mid, self.right_bot]):
            for i, pkl in enumerate(pkl_set): 
                 
                #Plot spectrum
                wavelength = np.asarray(
                  pkl['wavelength_raw'].tolist()[0]).astype(np.float)
                flux_normalized = (np.asarray(
                  pkl['flux_normalized'].tolist()[0]).astype(np.float)
                  + offset[i] + pkl_offset_scaling[l] * offset_lvl)   
                self.ax2.plot(wavelength, flux_normalized, color=color[i],
                              linewidth=2., alpha=alpha[i])

                #Plot photosphere temperature text
                try:
                    text_lvl = mean_flux(wavelength, flux_normalized, 1500., 2500.)
                    self.ax2.text(
                      1700, text_lvl + offset_text,
                      r'$\mathrm{' + str(format(pkl['t_inner'][0] ,'.0f')) + ' \ K}$',
                      fontsize=20., horizontalalignment='left', color=color[i])
                except:
                    pass   
                    
                #Plot date text
                if i == 0:
                    text_lvl = mean_flux(wavelength, flux_normalized, 5500., 6000.)
                    self.ax2.text(
                      text_date_wavelength, text_lvl - offset_text,
                      text_date_up[l], fontsize=20., horizontalalignment='left')                                 
                if i == 8:
                    text_lvl = mean_flux(wavelength, flux_normalized, 8200., 8300.)
                    self.ax2.text(
                      text_date_wavelength, text_lvl + 1.5 * offset_text,
                      text_date_down[l], fontsize=20., horizontalalignment='left') 
                            
                #Plot luminosity text
                if (l == 1) and (i in [3, 5, 6, 7]):
                    text_lvl = mean_flux(wavelength, flux_normalized, 8600., 8700.)
                    self.ax2.text(
                      text_lum_wavelength, text_lvl - 0.5 * offset_text, next(text_lum),
                      fontsize=20., horizontalalignment='left', color='g')

                #Add set text
                if i == 1:
                    text_lvl = mean_flux(wavelength, flux_normalized, 1800., 2200.)
                    self.ax2.text(text_set_wavelength, text_lvl - 4. * offset_text,
                    next(text_set), fontsize=20., horizontalalignment='left', color='k')                    
                       
    def add_legend(self):
        lw = 2.
        self.ax1.plot([np.nan], [np.nan], color='k', linewidth=lw, label='SN 2011fe')
        self.ax1.plot([np.nan], [np.nan], color='b', linewidth=lw, label='SN 2011fe $-$ synthetic')
        self.ax1.plot([np.nan], [np.nan], color='r', linewidth=lw, label='SN 2005bl')
        self.ax1.plot([np.nan], [np.nan], color='g', linewidth=lw, label='SN 2005bl $-$ synthetic')
        self.ax1.legend(frameon=False, fontsize=20., numpoints=1, ncol=2,  columnspacing=3.0, labelspacing=0.05, loc=2) 
        
        self.ax2.plot([np.nan], [np.nan], color='k', linewidth=lw, label='SN 2011fe')
        self.ax2.plot([np.nan], [np.nan], color='b', linewidth=lw, label='SN 2011fe $-$ synthetic')
        self.ax2.plot([np.nan], [np.nan], color='r', linewidth=lw, label='SN 2005bl')
        self.ax2.plot([np.nan], [np.nan], color='g', linewidth=lw, label='SN 2005bl $-$ synthetic') 
        self.ax2.legend(frameon=False, fontsize=20., numpoints=1, ncol=2,  columnspacing=3.0, labelspacing=0.05, loc=1)         
        
    def save_figure(self, extension='pdf', dpi=360):
        if self.save_fig:
            plt.savefig('./../OUTPUT_FILES/FIGURES/Fig_combined_11fe_05bl_transition_'+self.line_mode+'.'+extension, format=extension, dpi=dpi)
        
    def show_figure(self):
        if self.show_fig:
            plt.show()
                
    def run_comparison(self):
        self.set_fig_frame()
        self.load_spectra()
        plt.tight_layout()
        self.plotting_11fe_to_05bl()
        self.plotting_05bl_to_11fe()
        self.add_legend()
        self.save_figure()
        self.show_figure()  
        return None         


compare_spectra_object = Compare_Spectra(line_mode='downbranch', show_fig=False,
                                         save_fig=True)


