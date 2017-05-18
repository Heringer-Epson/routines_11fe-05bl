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

import colormaps as cmaps
                                                
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

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
        self.ax1.set_xlim(2500.,10000.)
        self.ax1.set_ylim(-21.,4.)      
        self.ax1.tick_params(axis='y', which='major', labelsize=self.fs_ticks, pad=8)       
        self.ax1.tick_params(axis='x', which='major', labelsize=self.fs_ticks, pad=8)
        self.ax1.minorticks_on()
        self.ax1.tick_params('both', length=8, width=1, which='major')
        self.ax1.tick_params('both', length=4, width=1, which='minor')
        self.ax1.xaxis.set_minor_locator(MultipleLocator(500.))
        self.ax1.xaxis.set_major_locator(MultipleLocator(1000.))
        self.ax1.yaxis.set_minor_locator(MultipleLocator(0.5))
        self.ax1.yaxis.set_major_locator(MultipleLocator(2.))       
        self.ax1.tick_params(labelleft='off')       

        self.ax2.set_xlabel(right_x_label, fontsize=self.fs_label)
        self.ax2.set_xlim(2500.,10000.)
        self.ax2.set_ylim(-21.,4.)      
        self.ax2.tick_params(axis='y', which='major', labelsize=self.fs_ticks, pad=8)       
        self.ax2.tick_params(axis='x', which='major', labelsize=self.fs_ticks, pad=8)
        self.ax2.minorticks_on()
        self.ax2.tick_params('both', length=8, width=1, which='major')
        self.ax2.tick_params('both', length=4, width=1, which='minor')
        self.ax2.xaxis.set_minor_locator(MultipleLocator(500.))
        self.ax2.xaxis.set_major_locator(MultipleLocator(1000.))
        self.ax2.yaxis.set_minor_locator(MultipleLocator(0.5))
        self.ax2.yaxis.set_major_locator(MultipleLocator(2.))       
        self.ax2.tick_params(labelleft='off')   
    
    def load_spectra(self):

        #Define path variables.
        path_syn_11fe = (path_tardis_output + '11fe_standard_'
                         + self.line_mode + '_lowSN/')                        
        path_syn_05bl = (path_tardis_output + '05bl_standard_'
                         + self.line_mode + '_lowSN/')                       
        path_obs_11fe = ('./../INPUT_FILES/observational_spectra/2011fe/')
        path_obs_05bl = ('./../INPUT_FILES/observational_spectra/2005bl/')
                                                  
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
                    path_obs_05bl + '2005_05_04.pkl']        
                
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
       
        offset_lvl = -0.8
        offset = [0., 0., 1.*offset_lvl, 2.*offset_lvl, 2.*offset_lvl,
                  3.*offset_lvl, 3.*offset_lvl, 4.*offset_lvl, 4.*offset_lvl]
        color = ['k', 'b', 'b', 'b', 'r', 'b', 'r', 'g', 'r']
        lw = [2., 2., 2., 2., 2., 2., 2., 2., 2.]
        alpha = [1., 1., .8, .6, 1., .4, 1., 1., 1.]
        
        for i, pkl in enumerate(self.left_top):  
            wavelength = np.asarray(
              pkl['wavelength_raw'].tolist()[0]).astype(np.float)
            flux_normalized = np.asarray(
              pkl['flux_normalized'].tolist()[0]).astype(np.float) + offset[i]         
            self.ax1.plot(wavelength, flux_normalized, color=color[i],
                          linewidth=2., alpha=alpha[i])
            
        for i, pkl in enumerate(self.left_mid):      
            wavelength = np.asarray(
              pkl['wavelength_raw'].tolist()[0]).astype(np.float)
            flux_normalized = (np.asarray(
              pkl['flux_normalized'].tolist()[0]).astype(np.float)
              + offset[i] + 7. * offset_lvl)
            self.ax1.plot(wavelength, flux_normalized, color=color[i],
                          linewidth=2., alpha=alpha[i])

        for i, pkl in enumerate(self.left_bot):      
            wavelength = np.asarray(
              pkl['wavelength_raw'].tolist()[0]).astype(np.float)
            flux_normalized = (np.asarray(
              pkl['flux_normalized'].tolist()[0]).astype(np.float)
              + offset[i] + 14. * offset_lvl)
            self.ax1.plot(wavelength, flux_normalized, color=color[i],
                          linewidth=2., alpha=alpha[i])
        
    def plotting_05bl_to_11fe(self):
        """Plot the right panel"""
        offset_lvl = -0.8
        offset = [0., 0., 1.*offset_lvl, 1.*offset_lvl, 2.*offset_lvl,
                  2.*offset_lvl, 3.*offset_lvl, 4.*offset_lvl, 4.*offset_lvl]
        color = ['k', 'b', 'k', 'g', 'k', 'g', 'g', 'g', 'r']
        alpha = [1., 1., 1., .4, 1., .6, .8, 1., 1.]
        
        for i, pkl in enumerate(self.right_top):         
            wavelength = np.asarray(
              pkl['wavelength_raw'].tolist()[0]).astype(np.float)
            flux_normalized = np.asarray(
              pkl['flux_normalized'].tolist()[0]).astype(np.float) + offset[i]
            self.ax2.plot(wavelength, flux_normalized, color=color[i],
                          linewidth=2., alpha=alpha[i])
        
        for i, pkl in enumerate(self.right_mid):      
            wavelength = np.asarray(
              pkl['wavelength_raw'].tolist()[0]).astype(np.float)
            flux_normalized = (np.asarray(
              pkl['flux_normalized'].tolist()[0]).astype(np.float)
              + offset[i] + 7. * offset_lvl)
            self.ax2.plot(wavelength, flux_normalized, color=color[i],
                          linewidth=2., alpha=alpha[i])

        for i, pkl in enumerate(self.right_bot):      
            wavelength = np.asarray(
              pkl['wavelength_raw'].tolist()[0]).astype(np.float)
            flux_normalized = (np.asarray(
              pkl['flux_normalized'].tolist()[0]).astype(np.float)
              + offset[i] + 14. * offset_lvl)
            self.ax2.plot(wavelength, flux_normalized, color=color[i],
                          linewidth=2., alpha=alpha[i])
        
    def add_text_ax1(self):
        
        '''
        Luminosity texts
        '''
        self.ax1.text(8600,-5.75, r'$L=L_{\mathrm{11fe}}$', fontsize=20., horizontalalignment='left', color='b')
        self.ax1.text(8600,-6.45, r'$L=\mathrm{0.5} L_{\mathrm{11fe}}$', fontsize=20., horizontalalignment='left', color='b')
        self.ax1.text(8600,-7.2, r'$L=\mathrm{0.33} L_{\mathrm{11fe}}$', fontsize=20., horizontalalignment='left', color='b')
        self.ax1.text(8600,-7.95, r'$L=\mathrm{0.25} L_{\mathrm{11fe}}$', fontsize=20., horizontalalignment='left', color='b')
        
        '''
        Date text
        '''
        #(-1week, max, +1week)
        
        self.ax1.text(5700, .6, r'$t_{\mathrm{11fe}}=\mathrm{12.1 \ d,} \ L_{\mathrm{11fe}}=\mathrm{2.3 \ 10^9 L_{\odot}}$', fontsize=20., horizontalalignment='left')
        self.ax1.text(5700, -3.5, r'$t_{\mathrm{05bl}}=\mathrm{12 \ d,} \ L_{\mathrm{05bl}}=\mathrm{0.19} L_{\mathrm{11fe}}$', fontsize=20., horizontalalignment='left')
        
        self.ax1.text(5700, -5.0, r'$t_{\mathrm{11fe}}=\mathrm{19.1 \ d,} \ L_{\mathrm{11fe}}=\mathrm{3.5 \ 10^9 L_{\odot}}$', fontsize=20., horizontalalignment='left')
        self.ax1.text(5700, -8.9, r'$t_{\mathrm{05bl}}=\mathrm{21.8 \ d,} \ L_{\mathrm{05bl}}=\mathrm{0.22} L_{\mathrm{11fe}}$', fontsize=20., horizontalalignment='left')

        self.ax1.text(5700, -10.6, r'$t_{\mathrm{11fe}}=\mathrm{28.3 \ d,} \ L_{\mathrm{11fe}}=\mathrm{2.3 \ 10^9 L_{\odot}}$', fontsize=20., horizontalalignment='left')
        self.ax1.text(5700, -14.75, r'$t_{\mathrm{05bl}}=\mathrm{29.9 \ d,} \ L_{\mathrm{05bl}}=\mathrm{0.17} L_{\mathrm{11fe}}$', fontsize=20., horizontalalignment='left')
            
        '''
        Temperature text
        '''
        self.ax1.text(1700, -0.235, r'$\mathrm{'+str(format(self.list_pkl_texp12[1]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='b')
        self.ax1.text(1700, -0.99, r'$\mathrm{'+str(format(self.list_pkl_texp12[2]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='b')
        self.ax1.text(1700, -1.80, r'$\mathrm{'+str(format(self.list_pkl_texp12[3]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='b')
        self.ax1.text(1700, -2.60, r'$\mathrm{'+str(format(self.list_pkl_texp12[5]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='b')
        self.ax1.text(1700, -3.35, r'$\mathrm{'+str(format(self.list_pkl_texp12[7]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='g')
        
        self.ax1.text(1700, -5.75, r'$\mathrm{'+str(format(self.list_pkl_texp19[1]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='b')
        self.ax1.text(1700, -6.58, r'$\mathrm{'+str(format(self.list_pkl_texp19[2]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='b')
        self.ax1.text(1700, -7.40, r'$\mathrm{'+str(format(self.list_pkl_texp19[3]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='b')
        self.ax1.text(1700, -8.21, r'$\mathrm{'+str(format(self.list_pkl_texp19[5]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='b')
        self.ax1.text(1700, -8.96, r'$\mathrm{'+str(format(self.list_pkl_texp19[7]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='g')
                
        self.ax1.text(1700, -11.38, r'$\mathrm{'+str(format(self.list_pkl_texp28[1]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='b')
        self.ax1.text(1700, -12.20, r'$\mathrm{'+str(format(self.list_pkl_texp28[2]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='b')
        self.ax1.text(1700, -13.02, r'$\mathrm{'+str(format(self.list_pkl_texp28[3]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='b')
        self.ax1.text(1700, -13.83, r'$\mathrm{'+str(format(self.list_pkl_texp28[5]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='b')
        self.ax1.text(1700, -14.60, r'$\mathrm{'+str(format(self.list_pkl_texp28[7]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='g')
        #print self.list_pkl_texp28[1]['velocity_f7'][0], self.list_pkl_texp28[2]['velocity_f7'][0], self.list_pkl_texp28[3]['velocity_f7'][0], self.list_pkl_texp28[5]['velocity_f7'][0], self.list_pkl_texp28[7]['velocity_f7'][0]

        '''
        Letters for sets
        '''
        self.ax1.text(2000, 0.6, r'${\rm \mathbf{a}}$', fontsize=20., horizontalalignment='left', color='k')
        self.ax1.text(2000, -5.0, r'${\rm \mathbf{b}}$', fontsize=20., horizontalalignment='left', color='k')
        self.ax1.text(2000, -10.6, r'${\rm \mathbf{c}}$', fontsize=20., horizontalalignment='left', color='k')
        
                        
        return None

    def add_text_ax2(self):
        
        '''
        Luminosity texts
        '''
        self.ax2.text(8900,-6.25, r'$L=\mathrm{4} L_{\mathrm{05bl}}$', fontsize=20., horizontalalignment='left', color='g')
        self.ax2.text(8900,-7.00, r'$L=\mathrm{3} L_{\mathrm{05bl}}$', fontsize=20., horizontalalignment='left', color='g')
        self.ax2.text(8900,-7.80, r'$L=\mathrm{2} L_{\mathrm{05bl}}$', fontsize=20., horizontalalignment='left', color='g')
        self.ax2.text(8900,-8.40, r'$L=L_{\mathrm{05bl}}$', fontsize=20., horizontalalignment='left', color='g')
        
        '''
        Date text
        '''


        #(-1week, max, +1week)
        
        self.ax2.text(5700, .6, r'$t_{\mathrm{11fe}}=\mathrm{12.1 \ d,} \ L_{\mathrm{11fe}}=\mathrm{5.4} L_{\mathrm{05bl}}$', fontsize=20., horizontalalignment='left')
        self.ax2.text(5700, -3.5, r'$t_{\mathrm{05bl}}=\mathrm{12 \ d,} \ L_{\mathrm{05bl}}=\mathrm{3.39 \ 10^8 L_{\odot}}$', fontsize=20., horizontalalignment='left')
        
        self.ax2.text(5700, -5.0, r'$t_{\mathrm{11fe}}=\mathrm{19.1 \ d,} \ L_{\mathrm{11fe}}=\mathrm{4.5} L_{\mathrm{05bl}}$', fontsize=20., horizontalalignment='left')
        self.ax2.text(5700, -8.9, r'$t_{\mathrm{05bl}}=\mathrm{21.8 \ d,} \ L_{\mathrm{05bl}}=\mathrm{7.76 \ 10^8 L_{\odot}}$', fontsize=20., horizontalalignment='left')

        self.ax2.text(5700, -10.6, r'$t_{\mathrm{11fe}}=\mathrm{28.3 \ d,} \ L_{\mathrm{11fe}}=\mathrm{5.9} L_{\mathrm{05bl}}$', fontsize=20., horizontalalignment='left')
        self.ax2.text(5700, -14.75, r'$t_{\mathrm{05bl}}=\mathrm{29.9 \ d,} \ L_{\mathrm{05bl}}=\mathrm{3.89 \ 10^8 L_{\odot}}$', fontsize=20., horizontalalignment='left')
            
        '''
        Temperature text
        '''
        self.ax2.text(1700,  0.19, r'$\mathrm{'+str(format(self.list_pkl_texp11[1]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='b')
        self.ax2.text(1700, -0.79, r'$\mathrm{'+str(format(self.list_pkl_texp11[3]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='g')
        self.ax2.text(1700, -1.75, r'$\mathrm{'+str(format(self.list_pkl_texp11[5]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='g')
        self.ax2.text(1700, -2.50, r'$\mathrm{'+str(format(self.list_pkl_texp11[6]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='g')
        self.ax2.text(1700, -3.34, r'$\mathrm{'+str(format(self.list_pkl_texp11[7]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='g')
        
        self.ax2.text(1700, -5.75, r'$\mathrm{'+str(format(self.list_pkl_texp21[1]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='b')
        self.ax2.text(1700, -6.58, r'$\mathrm{'+str(format(self.list_pkl_texp21[3]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='g')
        self.ax2.text(1700, -7.40, r'$\mathrm{'+str(format(self.list_pkl_texp21[5]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='g')
        self.ax2.text(1700, -8.21, r'$\mathrm{'+str(format(self.list_pkl_texp21[6]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='g')
        self.ax2.text(1700, -8.96, r'$\mathrm{'+str(format(self.list_pkl_texp21[7]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='g')
                
        self.ax2.text(1700, -11.38, r'$\mathrm{'+str(format(self.list_pkl_texp30[1]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='b')
        self.ax2.text(1700, -12.20, r'$\mathrm{'+str(format(self.list_pkl_texp30[3]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='g')
        self.ax2.text(1700, -13.00, r'$\mathrm{'+str(format(self.list_pkl_texp30[5]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='g')
        self.ax2.text(1700, -13.78, r'$\mathrm{'+str(format(self.list_pkl_texp30[6]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='g')
        self.ax2.text(1700, -14.63, r'$\mathrm{'+str(format(self.list_pkl_texp30[7]['t_inner'][0] ,'.0f'))+' \ K}$', fontsize=20., horizontalalignment='left', color='g')

        '''
        Letters for sets
        '''
        self.ax2.text(2000, 0.6, r'${\rm \mathbf{d}}$', fontsize=20., horizontalalignment='left', color='k')
        self.ax2.text(2000, -5.0, r'${\rm \mathbf{e}}$', fontsize=20., horizontalalignment='left', color='k')
        self.ax2.text(2000, -10.6, r'${\rm \mathbf{f}}$', fontsize=20., horizontalalignment='left', color='k')      
                            
        return None

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
        
        return None

    def save_figure(self, extension='pdf', dpi=360):
        if self.save_fig:
            plt.savefig('./../OUTPUT_FILES/FIGURES/Fig_combined_11fe_05bl_transition_'+self.line_mode+extension, format=extension, dpi=dpi)
        return None
        
    def show_figure(self):
        if self.show_fig:
            plt.show()
        return None
                
    def run_comparison(self):
        self.set_fig_frame()
        self.load_spectra()
        plt.tight_layout()
        self.plotting_11fe_to_05bl()
        self.plotting_05bl_to_11fe()
        #self.add_text_ax1()
        #self.add_text_ax2()
        #self.add_legend()
        #self.save_figure()
        self.show_figure()  
        return None         


compare_spectra_object = Compare_Spectra(line_mode='downbranch', show_fig=True,
                                         save_fig=False)


