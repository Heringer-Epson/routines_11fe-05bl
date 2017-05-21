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
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
from matplotlib import colors
import matplotlib.collections as mcoll

import colormaps as cmaps
                                                
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

def flag2filling_twolists(flag_list1,flag_list2):
    filling_bolean = [(flag_f7 != 1. and flag_f6 != 1.) for (flag_f7,flag_f6)
                      in zip(flag_list1,flag_list2)]
    converter = {True: 'full', False: 'none'}
    filling = [converter[bolean] for bolean in filling_bolean]  
    return filling

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments     

def colorline(x, y, z, cmap, norm, linestyle, linewidth, alpha, zorder):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,linestyle=linestyle,linewidth=linewidth, alpha=alpha,zorder=zorder)

    return lc

class Get_BSNIP(object):
    def __init__(self):

        self.fs_label = 26
        self.fs_ticks = 26
        self.fs_legend = 20
        self.fs_text = 22
        self.fs_as = 24
        self.fs_feature = 14

        with open('./../OUTPUT_FILES//BSNIP_test.pkl', 'r') as inp:
            self.df_BSNIP = cPickle.load(inp)
                   
class Feature_Parspace(Get_BSNIP):

    def __init__(self, line_mode='downbranch', show_fig=True, save_fig=False):
                        
        Get_BSNIP.__init__(self)
        
        self.line_mode = line_mode
        self.show_fig = show_fig
        self.save_fig = save_fig
                
        self.L_array = list(np.logspace(8.544, 9.72, 40))[::-1]
               
        self.list_label_11fe, self.list_label_05bl = [], []
        self.list_pkl_11fe, self.list_L_05bl = [], []
      
        self.FIG = plt.figure(figsize=(10, 10))
        self.ax = plt.subplot(111)  
        self.color_11fe = None
           
        self.run_parspace()
        
    def set_fig_frame(self):
       
        self.ax.set_xlabel(r'pEW $\mathrm{[\AA]}$ of '
          + r'$\rm{Si}\,\mathrm{II} \ \lambda$6355' ,fontsize=self.fs_label) 
        self.ax.set_ylabel(r'pEW $\mathrm{[\AA]}$ of '
          + r'$\rm{Si}\,\mathrm{II} \ \lambda$5972' ,fontsize=self.fs_label)        
        self.ax.set_xlim(40.,180.)
        self.ax.set_ylim(0.,70.)
        self.ax.xaxis.set_minor_locator(MultipleLocator(5.))
        self.ax.xaxis.set_major_locator(MultipleLocator(20.))
        self.ax.yaxis.set_minor_locator(MultipleLocator(2.))
        self.ax.yaxis.set_major_locator(MultipleLocator(10.))
        self.ax.tick_params(axis='y', which='major', labelsize=self.fs_ticks, pad=8)
        self.ax.tick_params(axis='x', which='major', labelsize=self.fs_ticks, pad=8)
        self.ax.minorticks_on()
        self.ax.tick_params('both', length=8, width=1, which='major')
        self.ax.tick_params('both', length=4, width=1, which='minor')
    
    def plot_BSNIP(self):
        list_phase = np.asarray(
          self.df_BSNIP['phase'].tolist()).astype(np.float)      
        
        list_subtype = np.asarray(
          self.df_BSNIP['subtype'].tolist())[(list_phase >= -5.)
                                             & (list_phase <= 5.)]
        
        list_ID = np.asarray(
          self.df_BSNIP['SNID'].tolist())[(list_phase >= -5.)
                                          & (list_phase <= 5.)]
        
        list_x = (np.asarray(
          self.df_BSNIP['pEW_f7'].tolist()).astype(np.float)[
          (list_phase >= -5.) & (list_phase <= 5.)])
        
        list_x_unc = (np.asarray(self.df_BSNIP['pEW_unc_f7'].tolist())
          .astype(np.float)[(list_phase >= -5.) & (list_phase <= 5.)])
        
        list_x_flag = (np.asarray(self.df_BSNIP['pEW_flag_f7'].tolist())
          .astype(np.float)[(list_phase >= -5.) & (list_phase <= 5.)])
        
        list_x_Silv = (np.asarray(self.df_BSNIP['BSNIP_pEW_f7'].tolist())
        .astype(np.float)[(list_phase >= -5.) & (list_phase <= 5.)])
                
        list_y = (np.asarray(self.df_BSNIP['pEW_f6'].tolist())
          .astype(np.float)[(list_phase >= -5.) & (list_phase <= 5.)])
       
        list_y_unc = (np.asarray(self.df_BSNIP['pEW_unc_f6'].tolist())
          .astype(np.float)[(list_phase >= -5.) & (list_phase <= 5.)])
        
        list_y_flag = (np.asarray(self.df_BSNIP['pEW_flag_f6'].tolist())
          .astype(np.float)[(list_phase >= -5.) & (list_phase <= 5.)])
       
        list_y_Silv = (np.asarray(self.df_BSNIP['BSNIP_pEW_f6'].tolist())
          .astype(np.float)[(list_phase >= -5.) & (list_phase <= 5.)])
            
        markers = subtype2marker(list_subtype)
        filling = flag2filling_twolists(list_x_flag,list_y_flag)
                
        for x, y, x_err, y_err, marker, fill, x_Silv, y_Silv in zip(
          list_x, list_y, list_x_unc, list_x_unc, markers, filling,
          list_x_Silv, list_y_Silv):
           
            if (fill == 'full' and x_err<10. and y_err<10.
              and np.isnan(x_Silv) == False and  np.isnan(y_Silv) == False):
                    
                if (abs((x_Silv - x) / x_Silv) < 0.1
                  and abs((y_Silv - y) / y_Silv)):
                    
                    self.ax.errorbar(x, y, xerr=x_err, yerr=y_err, ls='None',
                                     marker=marker, fillstyle=fill, capsize=0.,
                                     color='gray', alpha=0.5, markersize=9.,
                                     zorder=3)                     
                

    def add_11fe_synthetic_spectra(self):

        #path_data_11fe = (path_tardis_output + '11fe_Lgrid_' + self.line_mode)        
        path_data_11fe = (path_tardis_output + '11fe_Lgrid_' + self.line_mode + '_old')        
        
        cmap_L = cmaps.viridis
        Norm_L = colors.Normalize(vmin=0., vmax=len(self.L_array) + 11.)                 

        list_pkl_11fe = []
        x_list_line, y_list_line, z_list_line = [], [], []

        for i, L in enumerate(self.L_array):
            L_str = str(format(np.log10(L), '.2f')) + '.pkl'        
            with open(path_data_11fe + '/loglum-' + L_str, 'r') as inp:
                pkl = cPickle.load(inp)
                
                print 'here', L_str, pkl['pEW_f7'].tolist()[0], pkl['pEW_f6'].tolist()[0]
                
                list_x = np.asarray(pkl['pEW_f7'].tolist()).astype(np.float)
                
                list_x_unc = 1.2 * np.asarray(
                  pkl['pEW_unc_f7'].tolist()).astype(np.float)
                
                list_x_flag = np.asarray(
                  pkl['pEW_flag_f7'].tolist()).astype(np.float)
                        
                list_y = np.asarray(pkl['pEW_f6'].tolist()).astype(np.float)
                
                list_y_unc = 1.2 * np.asarray(
                  pkl['pEW_unc_f6'].tolist()).astype(np.float)
                
                list_y_flag = np.asarray(
                  pkl['pEW_flag_f6'].tolist()).astype(np.float)
                    
                filling = ('full' if (list_x_flag != 1. and list_y_flag != 1.)
                           else 'none')
                           
                color = cmap_L(Norm_L(i))
                
                if filling == 'full': 
                    #Do not include objects where the features start to blend.
                    if  L/3.5e9 <= 1.5 and L/3.5e9 > 0.28:
                        x_list_line.append(pkl['pEW_f7'].tolist())
                        y_list_line.append(pkl['pEW_f6'].tolist())
                        z_list_line.append(i)
                                        
                        self.ax.errorbar(
                          list_x, list_y, xerr=list_x_unc,yerr=list_y_unc,
                          ls='None', marker='D', markersize=10.,
                          fillstyle=filling, capsize=0., color=color, zorder=4)                       
                        
                        self.color_11fe = color
                        
        #PLot line with gradient
        self.plot_line, = self.ax.plot(
          [np.nan], [np.nan], ls='--', linewidth=6.0, marker='None', color='k',
           zorder=1)
        
        lc = colorline(x_list_line, y_list_line, z=z_list_line, cmap=cmap_L,
                       norm=Norm_L, linestyle='-', linewidth=2., alpha=1.0,
                       zorder=1)
       
        self.ax.add_collection(lc)
    
    """
    def add_05bl_synthetic_spectra(self):           

        list_pkl_05bl, label_05bl = [], []
        for pkl_path in self.list_syn_pkl_05bl:
            with open(pkl_path, 'r') as inp:
                list_pkl_05bl.append(cPickle.load(inp))
                self.list_label_05bl.append(
                  pkl_path.split('/')[-1].split('.pkl')[0])
    
        x_list_line, y_list_line, z_list_line = [], [], []
    
        for i, (pkl,L) in enumerate(zip(list_pkl_05bl,self.list_L_05bl)):

            list_x                                                      = np.asarray(pkl[self.x_feature+'_f'+self.x_key].tolist()).astype(np.float)
            list_x_unc                                                  = 1.2*np.asarray(pkl[self.x_feature+'_unc_f'+self.x_key].tolist()).astype(np.float)
            list_x_flag                                                 = np.asarray(pkl[self.x_feature+'_flag_f'+self.x_key].tolist()).astype(np.float)
                    
            list_y                                                      = np.asarray(pkl[self.y_feature+'_f'+self.y_key].tolist()).astype(np.float)
            list_y_unc                                                  = 1.2*np.asarray(pkl[self.y_feature+'_unc_f'+self.y_key].tolist()).astype(np.float)
            list_y_flag                                                 = np.asarray(pkl[self.y_feature+'_flag_f'+self.y_key].tolist()).astype(np.float)
                
            filling                                                     = 'full' if (list_x_flag != 1. and list_y_flag != 1.) else 'none'
            #print 'item', i, 'L', self.list_L[i]/3.5e9, 'filling', filling
            
            if filling == 'full': #This condition alone (no restrain in L/3.5e9) will also include the L/3.5e9 = 0.264 case, where a 'v' starts to form on the left shoulder of the Si_II 6355A feature. 
                if  L/3.5e9 <= 1.05 and L/3.5e9 > 0.18:
                    x_list_line.append(pkl[self.x_feature+'_f'+self.x_key].tolist())
                    y_list_line.append(pkl[self.y_feature+'_f'+self.y_key].tolist())
                    z_list_line.append(i)
                                    
                    self.ax.errorbar(list_x,list_y,xerr=list_x_unc,yerr=list_y_unc,ls='None',marker='p',markersize=14.,fillstyle=filling,capsize=0.,color='g',zorder=3)                     
                    
        #PLot line 
        self.ax.plot(x_list_line, y_list_line, ls=':', marker='None', color='g', zorder=1)
    """
    

    def add_observational_spectra(self):

        path_data = './../INPUT_FILES/observational_spectra/'
        
        """Add 11fe observation data"""
        
        pkl_11fe_file = open(path_data + '2011fe/2011_09_10.pkl', 'r')
        pkl_11fe = cPickle.load(pkl_11fe_file)

        list_x_11fe = np.asarray(pkl_11fe['pEW_f7'].tolist()).astype(np.float)
               
        list_x_unc_11fe = 1.2 * np.asarray(
          pkl_11fe['pEW_unc_f7'].tolist()).astype(np.float)
          
        list_x_flag_11fe = np.asarray(
          pkl_11fe['pEW_flag_f7'].tolist()).astype(np.float)
                 
        list_y_11fe = np.asarray(pkl_11fe['pEW_f6'].tolist()).astype(np.float)
        
        list_y_unc_11fe = 1.2 * np.asarray(
          pkl_11fe['pEW_unc_f6'].tolist()).astype(np.float)
        
        list_y_flag_11fe = np.asarray(
          pkl_11fe['pEW_flag_f6'].tolist()).astype(np.float)
       
        filling_11fe = ('full' if (list_x_flag_11fe != 1.
                        and list_y_flag_11fe != 1.) else 'none')

        self.ax.errorbar(list_x_11fe, list_y_11fe, xerr=list_x_unc_11fe,
                         yerr=list_y_unc_11fe, ls='None', marker='>',
                         markersize=10., fillstyle=filling_11fe, capsize=0.,
                         color='k',zorder=3,label=r'SN 2011fe')                     

        """Add 05bl observation data"""

        pkl_05bl_file = open(path_data + '2005bl/2005_04_26.pkl', 'r')
        pkl_05bl = cPickle.load(pkl_05bl_file)
     
        list_x_05bl = np.asarray(pkl_05bl['pEW_f7'].tolist()).astype(np.float)
               
        list_x_unc_05bl = 1.2 * np.asarray(
          pkl_05bl['pEW_unc_f7'].tolist()).astype(np.float)
          
        list_x_flag_05bl = np.asarray(
          pkl_05bl['pEW_flag_f7'].tolist()).astype(np.float)
                 
        list_y_05bl = np.asarray(pkl_05bl['pEW_f6'].tolist()).astype(np.float)
        
        list_y_unc_05bl = 1.2 * np.asarray(
          pkl_05bl['pEW_unc_f6'].tolist()).astype(np.float)
        
        list_y_flag_05bl = np.asarray(
          pkl_05bl['pEW_flag_f6'].tolist()).astype(np.float)
       
        filling_05bl = ('full' if (list_x_flag_05bl != 1.
                        and list_y_flag_05bl != 1.) else 'none')
     
        self.ax.errorbar(list_x_05bl, list_y_05bl, xerr=list_x_unc_05bl,
                         yerr=list_y_unc_05bl, ls='None', marker='<',
                         markersize=10., fillstyle=filling_05bl, capsize=0.,
                         color='r', zorder=3, label=r'SN 2005bl')                     

    def plot_boundary_lines(self):
                
        #lines
        self.ax.plot([60.,90.], [30.,0.], ls='-',linewidth=3., marker='None',
                     color='k',zorder=1)
        self.ax.plot([100.,100.], [0.,30.], ls='-',linewidth=3., marker='None',
                     color='k',zorder=1)
        self.ax.plot([40.,180.], [30.,30.], ls='-',linewidth=3., marker='None',
                     color='k',zorder=1)
        
        #text
        self.ax.text(50.0, 20., 'SS', fontsize=self.fs_label, color='k',
                     ha='center', va='center',zorder=1)
        self.ax.text(80.0, 20., 'CN', fontsize=self.fs_label, color='k',
                     ha='center', va='center',zorder=1)
        self.ax.text(140., 20., 'BL', fontsize=self.fs_label, color='k',
                     ha='center', va='center',zorder=1)
        self.ax.text(80.0, 35., 'CL', fontsize=self.fs_label, color='k',
                     ha='center', va='center',zorder=1)
    
    def add_legend(self):
        
        self.ax.errorbar([np.nan], [np.nan], xerr=[np.nan], yerr=[np.nan],
                         ls='-', marker='D', markersize=10., capsize=0.,
                         color=self.color_11fe, label=r'SN 2011fe grid')                       
        
        self.ax.errorbar([np.nan], [np.nan], xerr=[np.nan], yerr=[np.nan],
                         ls=':', marker='p', markersize=14., capsize=0.,
                         color='g',label=r'SN 2005bl grid')                       
        
        subtypes = ['Ia-norm', 'Ia-91bg', 'Ia-91T', 'Ia-99aa', 'other']      
        markers = subtype2marker(subtypes)
               
        #Plot nan to get legend entries.
        for (subtype,marker) in zip(subtypes,markers):
            self.ax.errorbar([np.nan], [np.nan], xerr=[np.nan], yerr=[np.nan],
                             ls='None', marker=marker, color='gray', alpha=0.5,
                             markersize=9., capsize=0., label=subtype)        
        
        self.ax.legend(frameon=True, fontsize=20., numpoints=1, ncol=1,
                       handletextpad=0.2, labelspacing=0.05, loc=2)

    def save_figure(self, extension='pdf', dpi=360):        
        directory = './../OUTPUT_FILES/FIGURES/'
        if self.save_fig:
            plt.savefig(directory + 'Fig_parspace_' + self.line_mode + '.'
                        + extension, format=extension, dpi=dpi)
        
    def show_figure(self):
        if self.show_fig:
            plt.show()
        
    def run_parspace(self):
        self.set_fig_frame()
        self.plot_BSNIP()   
        self.add_11fe_synthetic_spectra()
        #self.add_05bl_synthetic_spectra()
        self.add_observational_spectra()
        self.plot_boundary_lines()
        self.add_legend()
        self.save_figure(extension='pdf')
        self.show_figure()              

parspace_object = Feature_Parspace(line_mode='downbranch', show_fig=True,
                                   save_fig=False)


