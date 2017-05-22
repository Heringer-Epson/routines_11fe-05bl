#!/usr/bin/env python

############################  IMPORTS  #################################

import matplotlib
import cPickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
from matplotlib.ticker import MultipleLocator
                                                        
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

def subtype2marker(subtype_list):       
    converter = {'Ia-norm': 's', 'Ia-91bg': '*', 'Ia-91T': '^',
    'Ia-99aa': 'v', 'other': 'o'}   
    subtype_list_standard = [subtype if subtype in converter.keys()
                             else 'other' for subtype in subtype_list]                         
    marker_list = [converter[subtype] for subtype in subtype_list_standard] 
    return marker_list

def flag2filling(flag_list):
    filling = []
    for flag in flag_list:
        if flag == 0.:
            filling.append('full')
        else:
            filling.append('none')
    return filling
    
class get_BSNIP(object):
    def __init__(self):
        with open('./../OUTPUT_FILES/BSNIP.pkl', 'r') as inp:
            self.df_BSNIP = cPickle.load(inp)
            self.keys = ['6', '7']
            self.keys_to_fit = ['6', '7']
            self.labels = [r'Si$_{\mathrm{II}} \ \lambda$5972', r'Si$_{\mathrm{II}} \ \lambda$6355']
                    
class Compare_Feature(get_BSNIP):

    def __init__(self, feature, key, feature_range, show_fig=True,
                 save_fig=False):
                                     
        get_BSNIP.__init__(self)
                            
        self.show_fig = show_fig
        self.save_fig = save_fig
        
        self.FIG = plt.figure(figsize=(10, 10))
        self.ax = plt.subplot(111)  
        
        self.feature = feature
        self.key = key
        self.feature_range = feature_range

        self.fs_label = 26
        self.fs_ticks = 26
        self.fs_legend = 20
        self.fs_text = 22
        self.fs_as = 24
        self.fs_feature = 14
        
        self.run_feature()
        
    def set_fig_frame(self):
        
        x_label = r''+self.feature+' $_{\mathrm{this \, work}}$'
        y_label = r''+self.feature+' $_{\mathrm{BSNIP}}$'
        
        self.ax.set_xlabel(x_label, fontsize=self.fs_label)
        self.ax.set_ylabel(y_label, fontsize=self.fs_label)
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs_ticks, pad=8)
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs_ticks, pad=8)
        self.ax.minorticks_on()
        self.ax.tick_params('both', length=8, width=1, which='major')
        self.ax.tick_params('both', length=4, width=1, which='minor')   

    def plot_comparison(self):
            
        list_phase = np.asarray(
          self.df_BSNIP['phase'].tolist()).astype(np.float)      
       
        list_subtype = np.asarray(
          self.df_BSNIP['subtype'].tolist())[(
          list_phase >= -5.) & (list_phase <= 5.)]
       
        list_linepEW_idx = np.asarray(
          [idx for idx in self.df_BSNIP.index.values.tolist()])[(
          list_phase >= -5.) & (list_phase <= 5.)].astype(np.float)
       
        #Read the features which were computed and the ones from BSNIP.
        list_linepEW_computed = np.asarray([
          pew for pew in self.df_BSNIP[self.feature + '_f' + self.key]
          .tolist()])[(list_phase >= -5.) & (list_phase <= 5.)].astype(np.float)
       
        list_linepEW_read = np.asarray(
          self.df_BSNIP['BSNIP_' + self.feature + '_f' + self.key].tolist())[(
          list_phase >= -5.) & (list_phase <= 5.)].astype(np.float)      
        
        #Attempt to read the BSNIP uncertainties.
        try:
            list_linepEW_read_unc = np.asarray(
              self.df_BSNIP['BSNIP_' + self.feature + '_unc_f' + self.key]
              .tolist())[(list_phase >= -5.)
              & (list_phase <= 5.)].astype(np.float)
        
            list_linepEW_read_unc_processed = list_linepEW_read_unc[(
              np.isnan(list_linepEW_computed) == False)
              & (np.isnan(list_linepEW_read) == False)]
          
        except:
            list_linepEW_read_unc_processed = np.full(
            len(list_linepEW_read), np.nan)

        #Attempt to read the computed uncertainties.
        try:
            list_linepEW_computed_unc = np.asarray(
              self.df_BSNIP[self.feature + '_unc' + '_f' + self.key]
              .tolist())[(list_phase >= -5.)
              & (list_phase <= 5.)].astype(np.float)
              
            list_linepEW_computed_unc_processed = list_linepEW_computed_unc[(
              np.isnan(list_linepEW_computed) == False)
              & (np.isnan(list_linepEW_computed) == False)]
              
        except:
            list_linepEW_computed_unc_processed = np.full(
              len(list_linepEW_computed), np.nan)

        #Attempt to read the flag of the computed objects.
        try:
            list_linepEW_flag = np.asarray(
              self.df_BSNIP[self.feature + '_flag' + '_f' + self.key]
              .tolist())[(list_phase >= -5.)
              & (list_phase <= 5.)].astype(np.float)
            
            list_linepEW_flag_processed = list_linepEW_flag[(
              np.isnan(list_linepEW_flag) == False)
              & (np.isnan(list_linepEW_flag) == False)]
              
            filling_processed = flag2filling(list_linepEW_flag_processed)
        
        except:
            filling_processed = ['none' for i in list_linepEW_computed] 

        #If plotting velocity, then convert units from 10^km/s to km/s.
        if self.feature == 'velocity':
            list_linepEW_computed = -1. * list_linepEW_computed
            list_linepEW_computed_unc_processed = (
              -1. * list_linepEW_computed_unc_processed)
                      
        list_linepEW_idx_processed = list_linepEW_idx[(
          np.isnan(list_linepEW_computed) == False)
          & (np.isnan(list_linepEW_read) == False)]
      
        list_subtype_processed = list_subtype[(
          np.isnan(list_linepEW_computed) == False)
          & (np.isnan(list_linepEW_read) == False)]
      
        list_linepEW_computed_processed = list_linepEW_computed[(
          np.isnan(list_linepEW_computed) == False)
          & (np.isnan(list_linepEW_read) == False)]
      
        list_linepEW_read_processed = list_linepEW_read[(
          np.isnan(list_linepEW_computed) == False)
          & (np.isnan(list_linepEW_read) == False)]
                
        markers_processed = subtype2marker(list_subtype_processed)

        for x, y, x_err, y_err, idx, marker, marker_fill in zip(
          list_linepEW_computed_processed, list_linepEW_read_processed,
          list_linepEW_computed_unc_processed, list_linepEW_read_unc_processed,
          list_linepEW_idx_processed, markers_processed, filling_processed):

            self.ax.errorbar(x, y, xerr=x_err, yerr=y_err, ls='None',
                             marker=marker, fillstyle=marker_fill, color='k',
                             zorder=3)
            
            if abs((x-y) / y) > 0.1:
                print int(idx), format(x,'.1f'), y, marker_fill
        
    def plot_reference(self):
        self.ax.plot(self.feature_range, self.feature_range, ls='--',
                     color='k', linewidth=2., zorder=1)
        self.ax.set_xlim(self.feature_range[0], self.feature_range[1])
        self.ax.set_ylim(self.feature_range[0], self.feature_range[1])
        return None
        
    def add_legend(self):
        subtypes = ['Ia-norm', 'Ia-91bg', 'Ia-91T', 'Ia-99aa', 'other']
        markers = subtype2marker(subtypes)
        for (subtype,marker) in zip(subtypes, subtype2marker(subtypes)):
            self.ax.plot([np.nan], [np.nan], ls='None', marker=marker,
                         color='k',label=subtype)     
        self.ax.legend(frameon=True, fontsize=self.fs_legend, numpoints=1,
                       ncol=1,  columnspacing=0., labelspacing=0.1, loc=2)

    def save_figure(self, extension='pdf', dpi=360):
        if self.save_fig:
            try:
                fig_name = 'compare_' + self.feature + '_f' + self.key
                plt.savefig('./../OUTPUT_FILES/FIGURES/Fig_' + fig_name + '.'
                            + extension, format=extension, dpi=dpi)
            except:
                pass             
        
    def show_figure(self):
        if self.show_fig:
            plt.show()
        
    def run_feature(self):
        self.set_fig_frame()
        self.plot_comparison()
        self.plot_reference()
        self.add_legend()
        self.save_figure(extension='pdf')
        self.show_figure()
        
Compare_Feature(feature='pEW', key='7', feature_range=[0., 200.],
                show_fig=True, save_fig=False) 

#Compare_Feature(feature='pEW', key='6', feature_range=[0., 70.],
#                show_fig=True, save_fig=False)                                      

#Compare_Feature(feature='depth', key='7', feature_range=[0., 1.],
#                show_fig=True, save_fig=False) 

#Compare_Feature(feature='depth', key='6', feature_range=[0., 1.],
#                show_fig=True, save_fig=False)

#Compare_Feature(feature='velocity', key='7', feature_range=[7., 15.],
#                show_fig=True, save_fig=False)               

#Compare_Feature(feature='velocity', key='6', feature_range=[7., 15.],
#                show_fig=True, save_fig=False)                  
