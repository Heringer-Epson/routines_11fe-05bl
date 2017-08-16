#!/usr/bin/env python

import os                                                               
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd   
import csv
from astropy import constants as const
from matplotlib.ticker import MultipleLocator

import tardis.tardistools.compute_features as cp

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

class BSNIP_Database(object):
    """Collects data from the the BSNIP program to create a .pkl file with the
    relevant information. Spectral features are re-computed.
    
    More information about BSNIP can be found at
    [[http://adsabs.harvard.edu/abs/2012MNRAS.425.1789S]].
    
    Parameters
    ----------
    subset_objects_idx : ~list
        The list contains the index of spectra in BSNIP to be processed.
        Default is None and runs through all objects in BSNIP.
        
    Notes
    -----
    When computing features using our routine, the smoothing window and number
    of MC runs (used for uncertainties) are currently hard coded in the
    'compute_observables' functions below. By default:
    smoothing_window=51, N_MC_runs=3000.
    """
    
    def __init__(self, filename='BSNIP', subset_objects_idx=None,
                 make_figures=False):
        
        self.filename = filename
        self.subset_objects_idx = subset_objects_idx
        self.make_figures = make_figures
                
        os.system('clear')      
        self.time_BSNIP = time.time()
        print '\n*COLLECTING BSNIP DATA.'
        self.BSNIP_fp = './../data/BSNIP_I/'
        self.BSNIP_spectra_fp = self.BSNIP_fp + 'paper_I/Spectra_database/'      
        self.df = None
        
        self.run_BSNIP_database()
   
    def remove_duplicate_files(self):
        """Function to remove 'duplicates' in BSNIP. Should not be used again!
        ---
        It removed files which also had both 'corrected' and 'uncorrected'
        versions. After running this, only the 'corrected' version remains.
        ---
        I kept this function for documentation purposes.
        """     
        start_time = time.time()
        for file_spectra in os.listdir(self.BSNIP_spectra_fp):
            if file_spectra[-13:-4] == 'corrected':
                os.remove(BSNIP_fp + file_spectra[0:-14] + file_spectra[-4::]) 

    #@profile
    def initialize_dataframe_and_get_spectra(self):
        """Collects information such as the ID and date which are available
        as part of the BSNIP filenames and information available in the spectra
        files. Initialises a dictionary where the wavelength, flux, ID and
        date are stored.
        """
        print '  -RUNNING: Retrieving spectra...'  
        list_base, list_index = [], []                                                                
        for file_spectra in os.listdir(self.BSNIP_spectra_fp):
            file_name_parts = (file_spectra.rstrip('\n').replace('.flm', '')
                               .split('-'))
            try:
                date = str(format(float(file_name_parts[1]), '.3f'))
                if len(date) < 7:
                    date = str(format(float(file_name_parts[2]), '.3f'))
            except:
                date = str(format(float(file_name_parts[2]), '.3f')) 
            SNID = file_name_parts[0].upper()            

            wavelength, flux = [], []
            with open(self.BSNIP_spectra_fp + file_spectra, 'r') as f:
                file_contend = csv.reader(f, delimiter=' ')
                for row in file_contend:
                    row = filter(None, row)
                    wavelength.append(float(row[0]))
                    flux.append(float(row[1]))         
            
            #Prep the data that will be stored in the dataframe. In particular,
            #the dataframe indexes will contain the SNID and data to uniquely
            #identify the entries.
            list_index.append(SNID + '|' + date)            
            list_base.append({'SNID': SNID, 'date': date,
                             'wavelength_raw': np.asarray(wavelength),
                             'flux_raw': np.asarray(flux)})  

        
        #Set the combination of SNID + date as the dataframe index.
        self.df = pd.DataFrame(list_base, index=list_index)
        self.df = self.df.sort_index()
        
        #Then attribute the dataframe indexes as the spectra ID 
        self.df['ID'] = self.df.index
        
        #Then drop dataframe index as the info is now stored in the 'ID'
        #column, so that the indexes become integers again, but this time in
        #more controlled and ordenated fashion.
        self.df = self.df.reset_index(drop=True)
        
    def read_general_info(self):
        """Get data from table 1 of paper I.
        Includes: subtype, host morfology, host redshift and foreground
        extinction E(B-V). 
        """
        print '  -RUNNING: Retrieving BSNIP spectral data...'      
        list_base = []
        with open(self.BSNIP_fp + 'paper_I/table1.dat', 'r') as f:
            for row in f:
                
                SNID = 'SN' + row[3:9].replace(' ', '').upper() 
                                
                subtype = row[10:18].replace(' ', '')
                if subtype == '':
                    subtype = np.nan                
                
                morp = row[50:57].replace(' ', '')
                if morp == '':
                    morp = np.nan
                                    
                redshift = row[57:63].replace(' ', '')
                if redshift == '':
                    redshift = np.nan                
                else:
                    redshift = float(redshift) / const.c.to('km/s').value
                                    
                extinction = row[63:69].replace(' ', '')
                if extinction == '':
                    extinction = np.nan  

                list_base.append({
                  'SNID': SNID, 'subtype': subtype, 'host_morphology': morp,
                  'host_redshift': float(redshift),
                  'foreground_extinction': float(extinction)})
        
        df_add_table1 = pd.DataFrame(list_base)
        
        self.df = pd.merge(self.df, df_add_table1,
                           on='SNID', how='left').set_index(self.df.index)
                           
    def read_phase_info(self):
        """Get data from table 2 of paper I.
        Includes: phase, wavelength range, reliability of spectrophotometry
        and flux correction.
        ---
        Because the format of the dates not always exactly match between
        the spectra files and the tables, I need to use the ID to sort the
        dataframe and match dataframes by index.
        ---
        Note: python will sort (e.g.) SN1997Y after 1997br
        (sorts at same position), while table 2
        sorts single letters first and SN1997Y appears before 1997br,
        causing a mistach between the indexes here and on the table.
        But because our routines also sort the table using python, all is fine.
        """
        print '  -RUNNING: Retrieving phases...'      
        list_base, list_index = [], []                                                                
        
        with open(self.BSNIP_fp + 'paper_I/table2.dat', 'r') as f:
            for row in f:
               
                SNID = 'SN' + row[3:9].replace(' ', '').upper()
                reliable = row[9:10].replace(' ', '')
                if reliable == '*':
                    reliable = '0'
                else:
                    reliable = '1'
                    
                date = row[11:25].replace(' ', '')
                date = date.replace('/', '')             
               
                phase = row[37:44].replace(' ', '')
                if phase == '':
                    phase = np.nan
                                   
                wavelength_min = row[47:52].replace(' ', '')            
                wavelength_max = row[52:59].replace(' ', '')               
                correction = row[115:116].replace(' ', '')
              
                list_index.append(SNID + '|' + date)
                
                list_base.append({
                  'reliable': reliable, 'phase': phase,
                  'wavelength_min': wavelength_min,
                  'wavelength_max': wavelength_max,
                  'flux_correction': correction})

        df_add_table2 = pd.DataFrame(list_base, index=list_index)
        df_add_table2 = df_add_table2.sort_index()
        df_add_table2 = df_add_table2.reset_index(drop=True)  
        self.df = self.df.join(df_add_table2)
      
    def read_types(self):
        """Get data from table 1 of paper II.
        Includes: Suptype according to Benetti, Branch and Wang schemes.
        """
        print '  -RUNNING: Appending subtypes...'      
        list_base = []
        
        with open(self.BSNIP_fp + 'paper_II/tablea1.dat', 'r') as f:
            for row in f:
               
                SNID = 'SN' + row[3:9].replace(' ', '').replace('\n', '').upper()
              
                subtype = row[75:82].replace(' ', '').replace('\n', '')
                if subtype == '':
                    subtype = np.nan
                    
                type_Benetti = (row[83:89].replace(' ', '').replace('\r', '')
                                .replace('\n', ''))                
                if type_Benetti == '':
                    type_Benetti = np.nan

                type_Branch = row[90:92].replace(' ', '').replace('\n', '')
                if type_Branch == '':
                    type_Branch = np.nan
                                    
                type_Wang = row[93:95].replace(' ', '').replace('\n', '')
                if type_Wang == '':
                    type_Wang = np.nan                 
                            
                list_base.append({
                  'SNID': SNID, 'subtype_II': subtype,
                  'type_Benetti': type_Benetti,
                  'type_Branch': type_Branch, 'type_Wang': type_Wang})
        
        df_add_tablea1 = pd.DataFrame(list_base)
        
        self.df = pd.merge(self.df, df_add_tablea1,
                           on='SNID', how='left').set_index(self.df.index)         
       
    def read_features(self):
        """Get teh spectral features from table b1-9 of paper II.
        Includes: pEW, velocity, depth, etc...
        """        
        print '  -RUNNING: Appending features...'      
        phase_list = list(self.df['phase'].tolist())
        SNID_list = list(self.df['SNID'].tolist())
                
        for i in range(9):
            key = str(i + 1)
            list_base = []
            
            variables = [
              'SNID', 'phase', 'BSNIP_flux_b_f' + key, 'BSNIP_flux_r_f' + key,
              'BSNIP_pEW_f' + key, 'BSNIP_pEW_unc_f' + key,
              'BSNIP_velocity_f' + key, 'BSNIP_velocity_unc_f' + key,
              'BSNIP_depth_f' + key, 'BSNIP_depth_unc_f' + key,
              'BSNIP_FWHM_f' + key, 'BSNIP_FWHM_unc_f' + key]
            
            with open(self.BSNIP_fp + 'paper_II/tableb' + str(i + 1)
                      + '.dat', 'r') as f:
                
                for row in f:
                    dv = {}
                                
                    dv['SNID'] = ('SN' + row[3:9].replace(' ', '')
                                  .replace('\n', '').upper())
                                
                    dv['phase'] = row[11:17].replace(' ', '').replace('\n', '')
                    
                    dv['BSNIP_flux_b_f' + key] = (row[18:24].replace(' ', '')
                                                .replace('\n', ''))
                   
                    dv['BSNIP_flux_r_f' + key] = (row[32:38].replace(' ', '')
                                                .replace('\n', ''))
                   
                    dv['BSNIP_pEW_f' + key]  = (row[46:51].replace(' ', '')
                                              .replace('\n', ''))
                   
                    dv['BSNIP_pEW_unc_f' + key] = (row[53:57].replace(' ', '')
                                                .replace('\n', '')) 
                   
                    dv['BSNIP_velocity_f' + key] = (row[73:78].replace(' ', '')
                                                  .replace('\n', '')) 
                   
                    dv['BSNIP_velocity_unc_f' + key] = (
                      row[80:84].replace(' ', '').replace('\n', '')) 
                   
                    dv['BSNIP_depth_f' + key] = (row[86:91].replace(' ', '')
                                               .replace('\n', ''))
                   
                    dv['BSNIP_depth_unc_f' + key] = (
                      row[93:98].replace(' ', '').replace('\n', ''))
                                                   
                    dv['BSNIP_FWHM_f' + key] = (
                      row[100:105].replace(' ', '').replace('\n', ''))
                                              
                    dv['BSNIP_FWHM_unc_f' + key] = (
                      row[107:110].replace(' ', '').replace('\n', ''))
                    
                    for var in  variables:
                        if dv[var] == '': dv[var] = np.nan
                    list_base.append(dv)
           
            df_add_tableb = pd.DataFrame(list_base)
           
            self.df = pd.merge(self.df, df_add_tableb, on=['SNID','phase'],
                               how='left').set_index(self.df.index)    
        
    def trim_by_phase_and_indexes(self):
        """Remove the spectra whose epoch is nowhere near maximum (i.e. >20d).
        This ensures the files created is smaller and the computation of
        observables faster.
        """
        print '  -REMOVING SPECTRA NOT NEAR MAXIMUM...'       
        if self.subset_objects_idx is not None: 
            self.df = self.df.loc[self.subset_objects_idx, :]       
        
        self.df = self.df.drop(
          self.df[self.df.phase.astype(float).fillna(100.) > 20.].index)
        
        self.df = self.df.dropna(subset=['phase'])
        self.df[['phase']] = self.df[['phase']].astype(str)


    def compute_observables(self):
        """Use the 'compute_features' routine in 'tardistools' to compute
        the pEW, velocity and depth of features in BSNIP.
        """

        print '  -COMPUTING FEATURES...'       
        
        out_list_dicts, list_index = [], []

        for index, row in self.df.iterrows():
                                                
            print '    -AT INDEX: ' + str(index)
            
            #Note that the passed extinction is zero. The pEW feature **is**
            #to be computed without correcting for extinction.
            out_row_dict = cp.Analyse_Spectra(
              wavelength=row['wavelength_raw'], flux=row['flux_raw'],
              redshift=row['host_redshift'], extinction=0., D=row.to_dict(),
              smoothing_window=51, deredshift_and_normalize=True).run_analysis()

            out_row_dict = cp.Compute_Uncertainty(
              D=out_row_dict, smoothing_window=51, N_MC_runs=3000).run_uncertainties() 

            out_list_dicts.append(out_row_dict)
            list_index.append(index)

        self.df = pd.DataFrame(out_list_dicts, index=list_index)
        
    def save_output(self):
        self.df.to_pickle('./../OUTPUT_FILES/' + self.filename + '.pkl')
        
    #@profile
    def run_BSNIP_database(self):
        self.initialize_dataframe_and_get_spectra()
        self.read_general_info()
        self.read_phase_info()
        self.read_types()
        self.read_features()
        self.trim_by_phase_and_indexes()
        self.compute_observables()
        self.save_output()

BSNIP_object = BSNIP_Database(filename='BSNIP', make_figures=False)
#BSNIP_object = BSNIP_Database(filename='BSNIP2', make_figures=False)
#BSNIP_object = BSNIP_Database(filename='BSNIP2', subset_objects_idx=[333], make_figures=False)

#BSNIP_object = BSNIP_Database(filename='BSNIP2', subset_objects_idx=np.arange(350,380,1), make_figures=False)


#BSNIP_object = BSNIP_Database(filename='BSNIP_test', subset_objects_idx=np.arange(136,141,1), make_figures=False)

#BSNIP_object = BSNIP_Database(subset_objects_idx=np.arange(100,200,1),
#                              make_figures=True)

