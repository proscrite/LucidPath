#!/gpfs0/arazi/projects/miniconda/bin/python3


###
###  Import libraries
###

# Import standard libraries
import os
import sys

# Import third party libraries
import pandas as pd

import scipy.cluster as cls
import scipy.spatial as spt

from invisible_cities.io.mcinfo_io import load_mchits_df as ldhits

###
###  Set variables
###

#Set variables
data_path = sys.argv[1]
file_name = sys.argv[2]



###
###  Load relevant information
###

#For deconvolved events
#deco_df = pd.read_hdf(data_path + 'NEW/DE/deco/src_deco/' + file_name, '/DECO/Events')
#deco_df = pd.read_hdf(data_path + 'NEW/Qbb/deco/src/' + file_name, '/DECO/Events')

#For mc events
#deco_df = ldhits(data_path + 'NEW/DE/deco/src_deco/' + file_name)
deco_df = ldhits(data_path + 'NEW/Qbb/deco/src/' + file_name)
deco_df.reset_index(drop=False, inplace=True)
deco_df.rename(columns={'x':'X','y':'Y','z':'Z','energy':'E','event_id':'event'}, inplace=True)

#Correct for known energy calibration issue in NEW QBB MC dataset
#deco_df.E *= 0.9945

try:
  deco_df = deco_df[deco_df.E > 0]
except:
  sys.exit(f'Failed to remove zero energy points for {file_name}.')

if len(deco_df) == 0:
  sys.exit(f'Empty deco file for {file_name}.')


#Initialize container to save output
file_df = pd.DataFrame()
compiled_energy_df = pd.DataFrame()

#Remove events outside of energy ROI of 1% about 2.458 MeV
#grps = deco_df.groupby('event').sum()
#event_list = grps[(grps.E > 2.458*0.99) & (grps.E < 2.458*1.01)].index.values
#event_list = grps[(grps.E > 1.593*0.99) & (grps.E < 1.593*1.01)].index.values
#del grps

event_list = deco_df.event.unique()

#for event_number in [15]:
for event_number in event_list:
  #Get information about the deconvolved data for this event
  event_df = deco_df[deco_df.event == event_number]
  event_df.reset_index(inplace=True, drop=False)

  #Record the total energy of the event
  event_energy = event_df.E.sum()

  #Single track detection
  #Agglomerative clustering identifies groups sperated by multitrack_distance mm
  #  Retain only the portion of the event containing the most energy.
  pdist = spt.distance.pdist(event_df[['X','Y','Z']].values)
  Z = cls.hierarchy.linkage(pdist)

  my_dendro = cls.hierarchy.dendrogram(Z, truncate_mode='lastp')
  sep_lists = [each[1] for each in my_dendro['dcoord'] if each[1] > 2.5]



  sep_df = pd.DataFrame(data=[[sep_lists]], columns=['Seps'])
  sep_df['Event'] = event_number
  sep_df['File_Name'] = file_name
  sep_df['E'] = event_energy

  file_df = pd.concat([file_df,sep_df])



#data_path = data_path + 'dissertation/NEW/Qbb/deco/recon/'
data_path = data_path + 'dissertation/NEW/Qbb/mc/recon/'

#data_path = data_path + 'dissertation/NEW/DE/deco/recon/'
save_path = data_path + 'sats/'

#Record info for energies
if not os.path.exists(save_path):
  os.mkdir(save_path)

file_df.reset_index(drop=True, inplace=True)
file_df.to_hdf(save_path + file_name, '/Sats')
