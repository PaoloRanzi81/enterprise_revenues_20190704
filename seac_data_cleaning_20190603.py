"""
Seac dataset: it contains revenues per year for each italian 
enterprise whose accounting is run -indirectly- by Seac (Trento, Italy).
"VolumeAffari" is the most important variable to be analyzed. 
Research question: find small niches (namely, macroareas) which could be
profitable to invest in (thus avoiding major players). 

Paolo Ranzi 
"""

# In order to set the correct pathways/folders, check which system are you
# using. It should be either Linux laptop (release == '4.20.7-042007-generic') 
# or Linux server (release == '4.4.0-143-generic').
import platform
import sys
import datetime
import os


RELEASE= platform.release()

if RELEASE == '4.18.0-22-generic': # Linux laptop
   BASE_DIR = ('/home/paolo/Dropbox/analyses/python/tensorflow/seac_data/')


else:
   BASE_DIR = '/home/Lan/paolo_scripts/exp_seasonality/seac_data'





## 1. IMPORTING MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


##############################################################################
# SCRIPT TO BE RUN ONLY ONCE FOR SAVING CLEAN DATASET
## 2. LOADING
#loading the "Dati" dataset from .csv file.
df_1 = pd.read_csv(os.path.sep.join([BASE_DIR, 
                                     'db_Contabilita_worklab2_ditte.csv']), 
    encoding='UTF-8', delimiter =';', header =0)


#loading the "Dati" dataset from .csv file.
df_2 = pd.read_csv(os.path.sep.join([BASE_DIR, 
                                     'db_Contabilita_worklab2_dati.csv']), 
    encoding='UTF-8', delimiter =';', header =0, thousands=',')

   
############################################################################### 
## 3.1.  PRE-PROCESSING df_1   

# drop duplicates
df_1 = df_1.drop_duplicates() 
   
# drop redundant columns
df_1 = df_1.drop(columns =['UfficioBase'])

# rename column which will be used for joining the two DataFrames
df_1.rename(columns={'CodContab': 'CodDitta', 
                     'COD.ATT.': 'correct_ateco'}, inplace =True)
    
# convert from int64 to string
df_1['correct_ateco'] = df_1['correct_ateco'].astype(str)    

# convert from int64 to string
df_1['PARTITA IVA'] = df_1['PARTITA IVA'].astype(str)

# in 'PARTITA IVA' remove trailing '.0' 
df_1['PARTITA IVA'] = df_1['PARTITA IVA'].replace({ '(\.\d*?)0+$': ''}, 
    regex=True)


###############################################################################   
## 3.2.  PRE-PROCESSING df_2
# drop rows when 'VolumeAffari' is empty
df_2 = df_2.drop(df_2[df_2['VolumeAffari']==' '].index)


# in 'VolumeAffari', 'IvaCredito' and 'IvaDebito' remove ',' for 
# indicating millions
df_2['VolumeAffari'] = df_2['VolumeAffari'].replace({ ',': ''}, regex=True)
df_2['IvaCredito'] = df_2['IvaCredito'].replace({ ',': ''}, regex=True)
df_2['IvaDebito'] = df_2['IvaDebito'].replace({ ',': ''}, regex=True)


# drop redundant columns
df_2 = df_2.drop(columns =['Mese'])


# force some columns to be float64 
df_2['VolumeAffari'] = pd.to_numeric(df_2.VolumeAffari, errors='coerce')
df_2['NrMovimenti'] = pd.to_numeric(df_2.NrMovimenti , errors='coerce')


# convert from int64 to string
df_2['Ateco'] = df_2['Ateco'].astype(str)
df_2['Anno'] = df_2['Anno'].astype(str)


# drop duplicates
df_2 = df_2.drop_duplicates()

# drop rows when 'VolumeAffari' is negative 
df_2 = df_2.drop(df_2[df_2['VolumeAffari'] < 0].index)


# drop rows when 'NrMovimenti' is < 12 
df_2 = df_2.drop(df_2[df_2['NrMovimenti'] < 12].index)


## drop rows when 'Ateco' is 'N'
#df_2 = df_2.drop(df_2[df_2['Ateco']=='N'].index)


# round column
df_2['VolumeAffari']  = df_2['VolumeAffari'].round(decimals=2)


# TEST: sampling randomly from dataset
#df_2 = df_2.sample(n=10000, axis=0, random_state= 81)
          

# select only enterprises where all four years of yearly income are present
df_3 = pd.DataFrame() 
for enterprise_id in pd.unique(df_2['CodDitta']):
    single_enterprise = df_2[df_2['CodDitta'] == enterprise_id]
    if (single_enterprise['Anno'].str.contains('2014', regex =True).any() &
        single_enterprise['Anno'].str.contains('2015', regex =True).any() &
        single_enterprise['Anno'].str.contains('2016', regex =True).any() &
        single_enterprise['Anno'].str.contains('2017', regex =True).any()):
        df_3= df_3.append(single_enterprise, ignore_index=True) 


# merge the two DataFrames (df_2, df_1) in one (df_7).
df_4 = pd.merge(df_3, df_1, how='inner', on= 'CodDitta')

# drop duplicates
columns_names = ['Anno', 'UfficioBase', 'Ufficio', 'CodDitta', 'Ateco', 'TipoIva',
       'NrMovimenti', 'NrDocA', 'NrDocV', 'VolumeAffari', 'IvaCredito',
       'IvaDebito', 'PARTITA IVA', 'CODICE FISCALE', 'NATURA GIURIDICA',
       'NATURA GIURIDICA DESC', 'COMMERCIALE_COMUNE', 'STATO DITTA',
       'TIPO']

# drop duplicates
df_4.drop_duplicates(subset = columns_names, inplace =True)

# reset index
df_4.reset_index(drop=True, inplace=True)


                             
# it eliminates '0' from 'correct_ateco'.
df_5 = pd.DataFrame() 
for index in df_4['correct_ateco'].index:
    # eliminating useless ateco code (i.e. letters of the alphabet)
    if  df_4.iloc[index]['correct_ateco'] == '0':
        df_4.loc[index, ['correct_ateco']] = df_4.iloc[index]['Ateco'] 
        df_5= df_5.append(df_4.loc[index], ignore_index=True)
    else:
        df_5= df_5.append(df_4.loc[index], ignore_index=True)
    

# add a leading '0' to Ateco's code which has unusual 5 digits (wrong) instead 
# of the 6 digits (correct).         
df_6 = pd.DataFrame() 
for index in df_5['correct_ateco'].index:
    # eliminating useless ateco code (i.e. letters of the alphabet)
    if  (len(df_5.iloc[index]['correct_ateco']) == 5):
        padded_code = str(df_5.iloc[index]['correct_ateco']).zfill(6) 
        df_5.loc[index, ['correct_ateco']] = padded_code
        df_6= df_6.append(df_5.loc[index], ignore_index=True)
    else:
        df_6= df_6.append(df_5.loc[index], ignore_index=True)   
    

# double-check that each enterprise has a consistent Ateco code for all 4 years
df_7 = pd.DataFrame() 
for enterprise_id in pd.unique(df_6['CodDitta']):
    single_enterprise = df_6[df_6['CodDitta'] == enterprise_id]
    ateco_code = pd.unique(single_enterprise['Ateco'])
    if len(ateco_code) == 1:
        code_count = single_enterprise['Ateco'].str.contains(str(ateco_code), regex =True).sum()
        if code_count == 4 :
            #print(code_count)
            df_7= df_7.append(single_enterprise, ignore_index=True) 

# drop redundant columns
df_7 = df_7.drop(columns =['Ateco'])


# save cleaned DataFrame as .csv file
#df_7.to_csv('C:/miscellaneus/python/tensorflow/seac_data/clean_data_1.csv', 
#index= False)
df_7.to_csv(os.path.sep.join([BASE_DIR, 
                                     'clean_data_1.csv']), index= False)


