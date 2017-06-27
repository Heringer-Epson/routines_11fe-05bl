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

def mean_flux(w, f, w_min, w_max):
    window_condition = ((w > w_min) & (w < w_max))  
    return np.mean(f[window_condition])
    
class L_Grid(object):
    """Makes a figure with two panels:
    Left: Sequence of TARDIS spectra where the luminosity is scaled.
    Right: Sequence of TARDIS spectra where the Ti mass fraction is scaled
    
    Parameters
    ----------
    line_mode : ~str
        Options are 'downbranch' or 'macroatom'.
        Used to determine the directory where the data will be collected.
        
    left_panel : ~str
        '11fe' or '05bl'.
        Used to determine the directory of the spectra that will be plotted in
        the left panel.
        
    Notes
    -----
    *One needs to make sure that the directories where the spectra are available
    do correspond to the SN requested (11fe/05bl) and the correct line_mode.
    *Requires 'path_tardis_output' to be defined in the bash file.
    """
    
    def __init__(self, line_mode='downbranch', left_panel='11fe',
                 show_pEW=False, show_fig=True, save_fig=False):

        self.line_mode = line_mode
        self.left_panel = left_panel
        self.show_pEW = show_pEW
        self.show_fig = show_fig
        self.save_fig = save_fig 
        self.L_array = list(np.logspace(8.544, 9.72, 20)[::-1])
        self.list_pkl, self.label = [], []  
        self.list_pkl_bright, self.list_pkl_faint = [], []
        self.FIG = plt.figure(figsize=(20,22))
        self.ax1 = plt.subplot(121) 
        self.ax2 = plt.subplot(122, sharey=self.ax1)

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
        self.ax2.set_ylim(-37.,5.)      
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
        
        """Get the pkl files for the luminosity grid."""
        path_data = (path_tardis_output + self.left_panel + '_Lgrid_'
                     + self.line_mode)
        
        for L in self.L_array:
            L_str = str(format(np.log10(L), '.3f')) + '.pkl'        
            with open(path_data + '/loglum-' + L_str, 'r') as inp:
                self.list_pkl.append(cPickle.load(inp))

        """Load pkl files for the titanium grid."""
        #path_data = (path_tardis_output + '11fe_Ti_' + self.line_mode)
        #Ti plot not yet run with macroatom.
        path_data = (path_tardis_output + '11fe_Ti_' + self.line_mode) 
        
        list_X_Ti = ['2000', '1000', '500', '200',
                     '100', '050', '020', '010', '005', '000']
                
        for X_Ti in list_X_Ti:
           
            filepath = path_data + '_' + X_Ti + '/loglum-9.544.pkl'
            with open(filepath, 'r') as inp:
                self.list_pkl_bright.append(cPickle.load(inp))            

            filepath = path_data + '_' + X_Ti + '/loglum-8.942.pkl'
            with open(filepath, 'r') as inp:
                self.list_pkl_faint.append(cPickle.load(inp))            
    
    def plotting(self):
       
        """Plot the left panel: Spectra assorted in a luminosity grid."""     
        cmap_L = cmaps.viridis
        Norm_L = colors.Normalize(vmin=0., vmax=len(self.list_pkl) + 11.)
        offset_lvl_L = -2.
        offset_global = 1.
       
        for i, (pkl,L) in enumerate(zip(self.list_pkl,self.L_array)):                           
           
            wavelength = (np.asarray(pkl['wavelength_raw'].tolist()[0])
                          .astype(np.float))
           
            flux_norm = (np.asarray(pkl['flux_normalized'].tolist()[0])
                         .astype(np.float) + offset_global + offset_lvl_L * i)
           
            #Restrict the plotted wavelength range so that there is room
            #for the luminosity text.
            selection = (wavelength >= 1500.) & (wavelength <= 8900.)

            w = wavelength[selection]
            f = flux_norm[selection]
            
            self.ax1.plot(w, f, color=cmap_L(Norm_L(i)))            
                 
            #Add luminosity text to plotted spectra.
            text_lvl = mean_flux(w, f, 8850., 8980.) - 0.15
            
            L_text = (r'$\mathrm{' + str(format(self.L_array[i] / (3.5e9), '.2f'))
            + '}L_{\mathrm{11fe}}$')
            
            self.ax1.text(
              8980, text_lvl, L_text, fontsize=20., horizontalalignment='left',
              color=cmap_L(Norm_L(i)))
            
            #Draw pEW region, if requested.
            if self.show_pEW:
                for key,color in zip(['6', '7'],['b','r']):
                    try:
                        wavelength_region = (np.asarray(
                          pkl['wavelength_region_f' + key].tolist()[0])
                          .astype(np.float))
                        
                        flux_region = (np.asarray(
                          pkl['flux_normalized_region_f' + key].tolist()[0])
                          .astype(np.float) + offset_global + offset_lvl_L * i)
                        
                        pseudo_flux = (np.asarray(
                          pkl['pseudo_cont_flux_f' + key].tolist()[0])
                          .astype(np.float) + offset_global + offset_lvl_L * i)
                        
                        self.ax1.plot(wavelength_region, pseudo_flux, ls='--',
                                      color=color, marker='None')
                        
                        self.ax1.fill_between(wavelength_region, flux_region,
                                              pseudo_flux, color=color,
                                              alpha=0.1,zorder=1)
                    except:
                        pass                
                    

        """ Plot the right panel: Spectra accroding to the titanium content of
        the ejecta.
        The top set of spectra assume the same brightness of SN 2011fe at max,
        whereas the bottom set assumes a quarter of that value.
        """ 
        cmap_Ti = cmaps.plasma      
        Norm_Ti = colors.Normalize(vmin=0., vmax=len(self.list_pkl_bright) + 5.)

        models = ['20', '10', '5', '2', '1', '0.5', '0.2',
                  '0.1', '0.05', '0']    
        
        offset_lvl_Ti = -2.
                    
        T_bright, T_faint = [], []
        for i, pkl in enumerate(self.list_pkl_bright):      
            wavelength = (np.asarray(pkl['wavelength_raw'].tolist()[0])
                          .astype(np.float))
                       
            flux_raw = (np.asarray(pkl['flux_normalized'].tolist()[0])
                        .astype(np.float) + 2. + offset_lvl_Ti * i)
            
            selection = (wavelength >= 1500.) & (wavelength <= 8900.)

            w = wavelength[selection]
            f = flux_raw[selection]

            self.ax2.plot(w, f, color=cmap_Ti(Norm_Ti(i)))      
            
            #Add Ti text to plotted spectra.
            text_lvl = mean_flux(w, f, 8850., 8980.) - 0.15        
            Ti_text = r'$\mathrm{' + models[i] + '}X\mathrm{(Ti)}$'                   
            self.ax2.text(
              8980, text_lvl, Ti_text, fontsize=20., horizontalalignment='left',
              color=cmap_Ti(Norm_Ti(i)))

            #Add Luminosity text
            if i == 0:
                text_lvl = mean_flux(w, f, 1700., 1800.) + 1.        
                self.ax2.text(1700, text_lvl, r'$L=L_{\mathrm{11fe,max}}$',
                fontsize=20., horizontalalignment='left', color='k')              
            
            T_bright.append(pkl['t_inner'][0])

        for i, pkl in enumerate(self.list_pkl_faint):           
            wavelength = (np.asarray(pkl['wavelength_raw'].tolist()[0])
                          .astype(np.float))
            
            flux_raw = (np.asarray(pkl['flux_normalized'].tolist()[0])
                        .astype(np.float) - 19. + offset_lvl_Ti * i)
            
            selection = (wavelength >= 1500.) & (wavelength <= 8900.)

            w = wavelength[selection]
            f = flux_raw[selection]

            self.ax2.plot(w, f, color=cmap_Ti(Norm_Ti(i)))
                       
            #Add Ti text to plotted spectra.
            text_lvl = mean_flux(w, f, 8850., 8980.) - 0.15        
            Ti_text = r'$\mathrm{' + models[i] + '}X\mathrm{(Ti)}$'                   
            self.ax2.text(
              8980, text_lvl, Ti_text, fontsize=20., horizontalalignment='left',
              color=cmap_Ti(Norm_Ti(i)))

            #Add Luminosity text
            if i == 0:
                text_lvl = mean_flux(w, f, 1700., 1800.) + 1.        
                self.ax2.text(1700, text_lvl,
                r'$L=\mathrm{0.25}L_{\mathrm{11fe,max}}$',
                fontsize=20., horizontalalignment='left', color='k') 

            T_faint.append(pkl['t_inner'][0])

        print ('\n\n\nMean Temperature for brigther and fainter models:',
               np.mean(T_bright), np.mean(T_faint))
        print ('\n\n\nTypical temperature difference for Ti models at high and'
               + ' low brightnesses: ', np.mean(np.asarray(
               [T1-T2 for (T1,T2) in zip(T_bright,T_faint)])))
            
    def save_figure(self, extension='pdf', dpi=360):
                
        directory = './../OUTPUT_FILES/FIGURES/'
        if self.save_fig:
            if self.show_pEW:
                filename = (directory + 'Fig_' + self.left_panel + '_'
                            + self.line_mode + '_L_and_Ti_grid_pEW.'
                            + extension)
            else:
                filename = (directory + 'Fig_' + self.left_panel + '_'
                            + self.line_mode + '_L_and_Ti_grid.' + extension)         
            plt.savefig(filename, format=extension, dpi=dpi)
        
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

#compare_spectra_object = L_Grid(line_mode='downbranch', left_panel='11fe',
#                                show_pEW=True, show_fig=True, save_fig=False)


#Run and save all options

compare_spectra_object = L_Grid(line_mode='downbranch',left_panel='11fe',
                                show_pEW=False, show_fig=False, save_fig=True)
compare_spectra_object = L_Grid(line_mode='downbranch',left_panel='11fe',
                                show_pEW=True, show_fig=False, save_fig=True)
compare_spectra_object = L_Grid(line_mode='downbranch',left_panel='05bl',
                                show_pEW=False, show_fig=False, save_fig=True)
compare_spectra_object = L_Grid(line_mode='downbranch',left_panel='05bl',
                                show_pEW=True, show_fig=False, save_fig=True)
compare_spectra_object = L_Grid(line_mode='macroatom',left_panel='11fe',
                                show_pEW=False, show_fig=False, save_fig=True)
compare_spectra_object = L_Grid(line_mode='macroatom',left_panel='11fe',
                                show_pEW=True, show_fig=False, save_fig=True)
compare_spectra_object = L_Grid(line_mode='macroatom',left_panel='05bl',
                                show_pEW=False, show_fig=False, save_fig=True)
compare_spectra_object = L_Grid(line_mode='macroatom',left_panel='05bl',
                                show_pEW=True, show_fig=False, save_fig=True)

