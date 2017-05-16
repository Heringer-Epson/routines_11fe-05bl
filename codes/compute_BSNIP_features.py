#!/usr/bin/env python

#Needs to re-implement extinction correction. Or perhaps data is already corrected for extinction and redshift?

############################  IMPORTS  #################################

import os																
import sys
import time

version																	= os.path.basename(__file__).split('_v')[-1].split('.py')[0]
analyse																	= __import__('retrieve_observables_uncertainty_v'+version).analyse_spectra
uncertainty_routine														= __import__('retrieve_observables_uncertainty_v'+version).uncertainty

path_S																	= os.environ["PATH_subtypes_main"]
path_data																= os.environ["PATH_SNIa_data"]
sys.path.insert(1,path_S+'codes/modules/')

from astropy															import constants as const
from scipy.signal														import savgol_filter														

import numpy															as np
import pandas															as pd	
import csv
import copy

import scipy

#############################  CONSTANTS  ##############################

c																		= const.c.to('km/s').value

##########################  MASTER CODE  ###############################

class BSNIP_database(object):
	
	def __init__(self, subset_objects_idx=None):
		
		self.subset_objects_idx											= subset_objects_idx
				
		os.system('clear')		
		self.time_BSNIP													= time.time()
		print '\n*COLLECTING BSNIP DATA.'
		self.BSNIP_filepath												= path_data+'BSNIP_I/'
		self.BSNIP_spectra_filepath										= self.BSNIP_filepath+'paper_I/Spectra_database/'		
		self.df															= None
		self.run_BSNIP_database()
	def remove_duplicate_files(self):
		"""
		This function is not to be used again, it removed files which also had a 'corrected' version,
		so that only the 'corrected' version remains. I kept this function for documentation purposes.
		"""		
		print '  -RUNNING: Removing duplicate files...',		
		start_time														= time.time()
		for file_spectra in os.listdir(self.BSNIP_spectra_filepath):
			if file_spectra[-13:-4] == 'corrected':
				os.remove(BSNIP_filepath+file_spectra[0:-14]+file_spectra[-4::]) 
		print 'DONE ('+str(format(time.time()-start_time, '.1f'))+'s)'
		return None

	def initialize_dataframe_and_get_spectra(self):
		print '  -RUNNING: Retrieving spectra...',	
		start_time														= time.time()
		list_base, list_index											= [], []																
		for file_spectra in os.listdir(self.BSNIP_spectra_filepath):
			file_name_parts												= file_spectra.rstrip('\n').replace('.flm', '').split('-')
			try:
				date													= str(format(float(file_name_parts[1]), '.3f'))
				if len(date) < 7:
					date												= str(format(float(file_name_parts[2]), '.3f'))
			except:
				date													= str(format(float(file_name_parts[2]), '.3f'))	
			SNID														= file_name_parts[0].upper()			

			wavelength, flux											= [], []
			with open(self.BSNIP_spectra_filepath+file_spectra, 'r') as f:
				file_contend											= csv.reader(f, delimiter=' ')
				for row in file_contend:
					row													= filter(None, row)
					wavelength.append(row[0])
					flux.append(row[1])	

			list_index.append(SNID+'|'+date)			
			list_base.append({'SNID': SNID, 'date': date, 'wavelength_raw': wavelength, 'flux_raw': flux})	

		self.df															= pd.DataFrame(list_base, index=list_index)
		self.df															= self.df.sort_index()
		self.df['ID']													= self.df.index	#This make sure the index have an obvious relation with the sorted SNID+date
		self.df															= self.df.reset_index(drop=True)
		print 'DONE ('+str(format(time.time()-start_time, '.1f'))+'s)'
		return None

	def read_general_info(self):
		"""
		Includes: subtype, host morfology, host redshift, foreground extinction E(B-V).
		Information comes from Table 1 in Silverman+ 2012, BSNIP paper I.
		Unfortunately the file formatting is not optimal and requires some gymnastic to
		extract the data.
		"""
		print '  -RUNNING: Retrieving BSNIP spectral data...',		
		start_time														= time.time()
		list_base														= []
		with open(self.BSNIP_filepath+'paper_I/table1.dat', 'r') as f:
			for row in f:
				SNID, subtype, morp, redshift, extinction				= 'sn'+row[3:9].replace(' ', ''), row[10:18].replace(' ', ''), row[50:57].replace(' ', ''), row[57:63].replace(' ', ''), row[63:69].replace(' ', '')
				
				SNID													= SNID.upper()				
							
				if subtype == '':
					subtype												= np.nan
				if morp == '':
					morp												= np.nan
				if redshift == '':
					redshift											= np.nan
				else:
					redshift											= str(float(redshift)/c)		
				if extinction == '':
					extinction											= np.nan				

				list_base.append({'SNID': SNID, 'subtype': subtype, 'host_morphology': morp, 'host_redshift': redshift, 'foreground_extinction': extinction})
		df_add_table1													= pd.DataFrame(list_base)
		self.df															= pd.merge(self.df, df_add_table1, on='SNID', how='left').set_index(self.df.index)	
		print 'DONE ('+str(format(time.time()-start_time, '.1f'))+'s)'
		return None

	def read_phase_info(self):
		"""
		Includes: phase, wavelength range, reliability of of spectrophotometry and flux correction.
		Information comes from Table 2 in Silverman+ 2012, BSNIP paper I.
		Unfortunately the file formatting is not optimal and requires some gymnastic to
		extract the data.
		Because the dates not always exactly match between the spectra files and the tables,
		I need to use the ID to sort the dataframe and match dataframes by index.
		Note: python will sort (e.g.) SN1997Y after 1997br (sorts at same position), while table 2
		sorts single letters first and SN1997Y appears before 1997br, causing a mistach between
		the indexes here and on the table. Since we also sort the table using python, all is fine.
		"""
		print '  -RUNNING: Retrieving phases...',		
		start_time														= time.time()
		list_base, list_index											= [], []																
		with open(self.BSNIP_filepath+'paper_I/table2.dat', 'r') as f:
			for row in f:
				SNID, reliable, date, phase, wavelength_min, wavelength_max, correction	= 'sn'+row[3:9].replace(' ', ''), row[9:10].replace(' ', ''), row[11:25].replace(' ', ''), row[37:44].replace(' ', ''), row[47:52].replace(' ', ''), row[52:59].replace(' ', ''), row[115:116].replace(' ', '')
				
				SNID													= SNID.upper()
				date													= date.replace('/', '')				

				if reliable	== '*':
					reliable											= '0'
				else:
					reliable											= '1'
				if phase == '':
					phase												= np.nan

				list_index.append(SNID+'|'+date)
				list_base.append({'reliable': reliable, 'phase': phase, 'wavelength_min': wavelength_min, 'wavelength_max': wavelength_max, 'flux_correction': correction})

		df_add_table2													= pd.DataFrame(list_base, index=list_index)
		df_add_table2													= df_add_table2.sort_index()
		df_add_table2													= df_add_table2.reset_index(drop=True)	
		self.df															= self.df.join(df_add_table2)
		print 'DONE ('+str(format(time.time()-start_time, '.1f'))+'s)'
		return None

	def read_types(self):
		print '  -RUNNING: Appending subtypes...',		
		start_time														= time.time()
		list_base														= []
		with open(self.BSNIP_filepath+'paper_II/tablea1.dat', 'r') as f:
			for row in f:
				SNID, subtype, type_Benetti, type_Branch, type_Wang		= 'sn'+row[3:9].replace(' ', '').replace('\n', ''), row[75:82].replace(' ', '').replace('\n', ''), row[83:89].replace(' ', '').replace('\r', '').replace('\n', ''), row[90:92].replace(' ', '').replace('\n', ''), row[93:95].replace(' ', '').replace('\n', '')
				
				SNID													= SNID.upper()				
							
				if subtype == '':
					subtype												= np.nan
				if type_Benetti == '':
					type_Benetti										= np.nan
				if type_Branch == '':
					type_Branch											= np.nan
				if type_Wang == '':
					type_Wang											= np.nan				

				list_base.append({'SNID': SNID, 'subtype_II': subtype, 'type_Benetti': type_Benetti, 'type_Branch': type_Branch, 'type_Wang': type_Wang})
		df_add_tablea1													= pd.DataFrame(list_base)
		self.df															= pd.merge(self.df, df_add_tablea1, on='SNID', how='left').set_index(self.df.index)			
		print 'DONE ('+str(format(time.time()-start_time, '.1f'))+'s)'
		return None

	def read_features(self):
		print '  -RUNNING: Appending features...',		
		start_time														= time.time()
		
		phase_list														= list(self.df['phase'].tolist())
		SNID_list														= list(self.df['SNID'].tolist())
				
		for i in range(9):
			key															= str(i+1)
			list_base													= []
			variables													= ['SNID', 'phase', 'BSNIP_flux_b_f'+key, 'BSNIP_flux_r_f'+key, 'BSNIP_pEW_f'+key, 'BSNIP_pEW_unc_f'+key, 'BSNIP_velocity_f'+key, 'BSNIP_velocity_unc_f'+key, 'BSNIP_depth_f'+key, 'BSNIP_depth_unc_f'+key, 'BSNIP_FWHM_f'+key, 'BSNIP_FWHM_unc_f'+key]
			with open(self.BSNIP_filepath+'paper_II/tableb'+str(i+1)+'.dat', 'r') as f:
				for row in f:
					dv													= {}
								
					dv['SNID'], dv['phase'], dv['BSNIP_flux_b_f'+key], dv['BSNIP_flux_r_f'+key], dv['BSNIP_pEW_f'+key]	= 'sn'+row[3:9].replace(' ', '').replace('\n', ''), row[11:17].replace(' ', '').replace('\n', ''), row[18:24].replace(' ', '').replace('\n', ''), row[32:38].replace(' ', '').replace('\n', ''), row[46:51].replace(' ', '').replace('\n', '')
					dv['BSNIP_pEW_unc_f'+key], dv['BSNIP_velocity_f'+key], dv['BSNIP_velocity_unc_f'+key], dv['BSNIP_depth_f'+key] = row[53:57].replace(' ', '').replace('\n', ''), row[73:78].replace(' ', '').replace('\n', ''), row[80:84].replace(' ', '').replace('\n', ''), row[86:91].replace(' ', '').replace('\n', '')
					dv['BSNIP_depth_unc_f'+key], dv['BSNIP_FWHM_f'+key], dv['BSNIP_FWHM_unc_f'+key] = row[93:98].replace(' ', '').replace('\n', ''),  row[100:105].replace(' ', '').replace('\n', ''), row[107:110].replace(' ', '').replace('\n', '')
					
					dv['SNID']											= dv['SNID'].upper()

					for var in 	variables:
						if dv[var] == '': dv[var] = np.nan

					list_base.append(dv)
			df_add_tableb												= pd.DataFrame(list_base)
			self.df														= pd.merge(self.df, df_add_tableb, on=['SNID','phase'], how='left').set_index(self.df.index)	
		print 'DONE ('+str(format(time.time()-start_time, '.1f'))+'s)'
		return None

	def trim_by_phase_and_indexes(self):
		print '  -REMOVING SPECTRA NOT NEAR MAXIMUM...',		
		start_time														= time.time()
		if self.subset_objects_idx is not None:	
			self.df														= self.df.loc[self.subset_objects_idx, :]		
		self.df															= self.df.drop(self.df[self.df.phase.astype(float).fillna(100.) > 20.].index)
		self.df															= self.df.dropna(subset=['phase'])
		self.df[['phase']]												= self.df[['phase']].astype(str)

		print 'DONE ('+str(format(time.time()-start_time, '.1f'))+'s)'
		return None

	def compute_observables(self):
		self.df															= analyse(self.df, smoothing_window=51, verbose=True).run_analysis()		
		self.df															= uncertainty_routine(self.df, smoothing_window=51, N_MC_runs=3000).run_uncertainties()		
		#print self.df['pEW_f7'], self.df['pEW_unc_f7'], self.df['pEW_f6'], self.df['pEW_unc_f6']
		return None

	def save_output(self):
		print '\n*SAVING OUTPUT AS PICKLE AT '+path_S+'data/dataframes/ ...',
		start_time														= time.time()
		self.df.to_pickle(path_S+'data/dataframes/BSNIP_trash.pkl')
		print 'DONE ('+str(format(time.time()-start_time, '.1f'))+'s)\n\n'
		return None

	def run_BSNIP_database(self):
		self.initialize_dataframe_and_get_spectra()
		self.read_general_info()
		self.read_phase_info()
		self.read_types()
		self.read_features()
		self.trim_by_phase_and_indexes()
		print "    -TOTAL TIME ELAPSED COLLECTING BSNIP DATA: "+str(format(time.time()-self.time_BSNIP, '.1f'))+'s'		
		print '    *** COLLECTING BSNIP DATA FINISHED SUCCESSFULLY ***'
		self.compute_observables()
		self.save_output()		
		return None

BSNIP_object															= BSNIP_database()
#BSNIP_object															= BSNIP_database(subset_objects_idx=[280])
#BSNIP_object															= BSNIP_database(subset_objects_idx=np.arange(100,200,1))






