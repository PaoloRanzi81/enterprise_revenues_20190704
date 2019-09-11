"""
Seac dataset: t-SNE analyses: 
1. t_sne_aggregated (21 x 4): t-SNE with aggregated data (median) by using a 
21 (Ateco code) X 4 (Years) matrix as input;  
2. t_sne_not_aggregated (12047 x 4): by using a 12047 (Ateco code) X 4 (Years) 
matrix as input

"""




# 1. SETTING PATHS
# In order to set the correct pathways/folders, check which system are you
# using. It should be either Linux laptop (release == '4.20.7-042007-generic') 
# or Linux server (release == '4.4.0-143-generic').
import platform
import sys
import datetime
import os
import datetime
import argparse

RELEASE = platform.release()

if RELEASE == '4.18.0-22-generic': # Linux laptop
   BASE_DIR = ('/home/paolo/Dropbox/analyses/python/tensorflow/seac_data/')

else:
   BASE_DIR = '/home/Lan/paolo_scripts/exp_seasonality/seac_data'

BASE_REP = BASE_DIR



# set whether use parallel computing (parallel = True) or 
# single-core computing (parallel = False).
 # construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--parallel",  dest='parallel', action='store_true',
	help="# enable multi-core computation")
ap.add_argument("-no-p", "--no-parallel",  dest='parallel', action='store_false',
	help="# disable multi-core computation")
args = vars(ap.parse_args())

# grab the "parallel" statment and store it in a convenience variable
# you have to set it from the command line
parallel = args["parallel"] 




## 2. IMPORTING MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from matplotlib.ticker import NullFormatter



## 3. LOAD 
# load cleaned dataset    
df_2 = pd.read_csv(os.path.sep.join([BASE_DIR, 
                                     'clean_data_1.csv']))
  

## 4. DATA MANAGMENT    
# convert columns with int64 to string
df_2['correct_ateco'] = df_2['correct_ateco'].astype(str)
df_2['Anno'] = df_2['Anno'].astype(str)
df_2['PARTITA IVA'] = df_2['PARTITA IVA'].astype(str)
 

# in 'PARTITA IVA' remove trailing '.0' 
df_2['PARTITA IVA'] = df_2['PARTITA IVA'].replace({ '(\.\d*?)0+$': ''}, 
    regex=True)
   

# WARNING: do not use it when only 22 time-series are computed
# Z-scores
from sklearn.preprocessing import StandardScaler
df_2.loc[:, ['VolumeAffari']] = StandardScaler().fit_transform\
(df_2.loc[:, ['VolumeAffari']])


# for each 'correct_ateco' and for 4 years compute standard deviation. 
correct_ateco_numeric = pd.to_numeric(df_2.correct_ateco, errors='coerce')

list_temp_1 = []
list_temp_2 = []
for ateco_code_tmp in pd.unique(correct_ateco_numeric):
    ateco_all = df_2[df_2['correct_ateco'] == str(ateco_code_tmp)]
    ateco_all_std = (np.std(ateco_all.loc[:, ['VolumeAffari']])).to_numpy()
    list_temp_1.append(str(ateco_code_tmp))
    list_temp_2.append(ateco_all_std[0])
 
# append list to DataFrame
df_2_std =  pd.DataFrame(columns = ['correct_ateco'])    
df_2_std['correct_ateco'] =  list_temp_1
df_2_std['sd_4_years'] =  list_temp_2




# convert back columns with int64 to string
df_2['correct_ateco'] = df_2['correct_ateco'].astype(str)

###############################################################################
## 2.6. 't_sne_aggregated (21 x 4)' 

# pivot table from brin
time_series_tsne_1= pd.pivot_table(df_2, 
                                 index='CodDitta', columns='Anno', 
                                 values='VolumeAffari')


# from index of the pivot table create a new column with the same values
time_series_tsne_1['CodDitta'] = time_series_tsne_1.index


# reset index
time_series_tsne_1.reset_index(drop=True, inplace=True) 


# for each 'CodDitta' find the corresponding Ateco code
list_temp_df  =[]
for enterprise_id in time_series_tsne_1['CodDitta']:
    index = df_2[df_2['CodDitta'] == enterprise_id].index
    ateco_code = df_2.iloc[index[0]]['correct_ateco']
    list_temp_df.append(ateco_code)


# append list of 'correct_ateco' to DataFrame    
# from index of the pivot table create a new column with the same values
time_series_tsne_1['correct_ateco'] = list_temp_df    
    

# append a new column with the first two Ateco digits.
time_series_tsne_1['first two ateco'] =\
time_series_tsne_1['correct_ateco'].apply(lambda ateco: ateco[0:2])






# check whether NaN are present
# print(np.isnan(time_series_tsne[time_series_tsne['first two ateco']]).any())
# print(np.isnan(time_series_tsne[time_series_tsne['correct_ateco']]).any())


# count the frequency of first two digits Ateco codes.    
z_two_digits = defaultdict(int)
for x in time_series_tsne_1['first two ateco']:
    z_two_digits[x] += 1


# since we have the highest number of data (5034 x 4 years) with enterprises 
# beloging to the macroarea '47', we analyze only such a macroarea. 
# Nevertheless, all other should be analyze if time consents. 
    
df_3 = time_series_tsne_1[time_series_tsne_1['first two ateco'] == '47']
 


    
"""
# cumulative distribution of microareas with xticks
sns.set()
sns.set_style('ticks')
pic= sns.distplot(df_3['2015'])
plt.xticks(range(1, 101, 2), range(1,101,2), rotation =70)       
"""   




# count the frequency of first four digits Ateco codes.    
z_six_digits = defaultdict(int)
for x in df_3['correct_ateco']:
    z_six_digits[x] += 1


# converting dictionary to DataFrame
z_df = pd.DataFrame.from_dict(z_six_digits, orient='index',
                           columns =['Frequency'])
z_df['six_digits'] = z_df.index
z_df.reset_index(drop=True, inplace=True) 
#z_df.info()


# select the 'first two ateco' with > 120 istances (30 instances * 4 years)
z_df_30_tmp_1 = z_df[z_df.Frequency > 30]
z_df_30_tmp_1.reset_index(drop=True, inplace=True)


# concatenate column with standard deviation for each Ateco code selected
z_df_30_tmp_2 = []
#z_df_30_tmp_1= pd.DataFrame(columns =['sd_4_years'])
for iii in z_df_30_tmp_1['six_digits']:
    single_std = df_2_std[df_2_std['correct_ateco'] == iii] 
    single_std.reset_index(drop=True, inplace=True)
    std = single_std.loc[0, 'sd_4_years']
    z_df_30_tmp_2.append(std)

z_df_30 = z_df_30_tmp_1.copy()
z_df_30['sd_4_years'] = z_df_30_tmp_2
    



# sub-set dataset by taking only a small subset
df_4_0_10 = pd.DataFrame() 
criteria = (z_df_30.sort_values(by= ['sd_4_years'], 
                                        ascending = True)[:5])
criteria.reset_index(drop=True, inplace=True)
for ateco_120 in criteria['six_digits']:
    ateco_first = df_3[df_3['correct_ateco']== ateco_120]
    df_4_0_10 = df_4_0_10.append(ateco_first, ignore_index=True) 




# convert DataFrame to input (not Numpy surprisingly) to feed into t-SNE
time_series_tsne = df_4_0_10.copy()     



###############################################################################
# 8. TIME-SERIES: build a 22 (Ateco code) X 4 (years) array


# build a time-series with the most frequent ateco codes (i.e. > 30 measures
# per year)
# select only enterprises where all four years of yearly income is present
list_temp_df  =[]
for ateco_120 in z_df_30['first two ateco']:
    ateco_first = df_3[df_3['first two ateco']== ateco_120]
    for years in pd.unique(ateco_first['Anno']):
        ateco_first_years = ateco_first[ateco_first['Anno'] == years]
        revenue_median = np.median(ateco_first_years['VolumeAffari'])
        list_temp = [int(ateco_120), years, int(revenue_median)]
        list_temp_df.append(list_temp)


# convert list into a dataframe
time_series_ateco_120 = pd.DataFrame(list_temp_df) 

# convert columns' names to string
time_series_ateco_120.columns = time_series_ateco_120.columns.astype(str)
#time_series_ateco_120.columns.map(type)

# rename DataFrame's columns
time_series_ateco_120.rename(columns={'0': 'first two ateco',
                               '1': 'Anno', 
                               '2': 'revenue_median'}, inplace =True)

    
# Z-scores    
from sklearn.preprocessing import StandardScaler
time_series_ateco_120.loc[:,['revenue_median']] = StandardScaler().\
fit_transform(time_series_ateco_120.loc[:,['revenue_median']])   
    
    
# convert array from long to wide:
time_series_tsne = pd.pivot_table(time_series_ateco_120, 
                                 index='first two ateco', columns='Anno', 
                                 values='revenue_median')


# create new column with index values (the Ateco code)
time_series_tsne['first two ateco'] = time_series_tsne.index 

# TEST: trying to build a global median for each Ateco code. The goal is to 
# have a parameter to be fed into t-SNE's plot which can control the size of 
# the spheres representing 'first two ateco'.
#time_series_tsne['total_median'] = np.median(time_series_tsne.loc[:, 
# ['2014', '2015']])

# reset index
time_series_tsne.reset_index(drop=True, inplace=True)


# convert int64 to string
time_series_tsne['first two ateco']= \
time_series_tsne['first two ateco'].astype(str)




###############################################################################
# 9. CLUSTERING: SUB-SETTING BY YEAR AND BY FIRST 10 MOST FREQUENT Ateco's codes
time_series_22_t_1 = df_3.loc[:, ['first two ateco', 'Anno', 'VolumeAffari']]
time_series_22_t_1.rename(columns={'VolumeAffari': 'revenue_median'}, 
                      inplace =True)

# extract a selection of Ateco code
criteria_1 = z_df_30.sort_values(by= ['Frequency'], 
                                        ascending = False)[:7]
criteria_1 = list(criteria_1.loc[:, 'first two ateco'])


# when sub-setting by year
#time_series_22_t_2 = time_series_22_t_1[time_series_22_t_1['Anno'] == 2014] 

# when NOT sub-setting by year
time_series_22_t_2 = time_series_22_t_1

# select only the 22 enterprise with > 120 data points 
time_series_tsne_long = pd.DataFrame() 
for enterprise_id in criteria_1:
    single_enterprise = time_series_22_t_2[time_series_22_t_2['first two ateco'] == enterprise_id]
    time_series_tsne_long= time_series_tsne_long.append(single_enterprise, ignore_index=True)


# building a balanced dataset: the same number of items per macroarea (i.e. 
# Ateco's codes) 
# "defaultdict" library    
z = defaultdict(int)
for x in time_series_tsne_long['first two ateco']:
    z[x] += 1


# when sub-setting by year
# building a balanced dataset: select only the 34 istances for each macroareas 
time_series_tsne_34 = pd.DataFrame() 
for enterprise_id in criteria_1:
    single_enterprise = time_series_tsne_long[time_series_tsne_long['first two ateco'] == enterprise_id]
    time_series_tsne_34= time_series_tsne_34.append(single_enterprise.iloc[0:34, :], ignore_index=True)


## double-check: it should contain 34 counts for each macroareas   
#z = defaultdict(int)
#for x in time_series_tsne_34['first two ateco']:
#    z[x] += 1
 
# when sub-setting by year: drop unused column
#time_series_tsne_34.drop(columns = 'Anno', inplace = True)


# it aggregates by creating a 7*4 DataFrame
#t= pd.pivot_table(time_series_tsne_long,index='first two ateco', 
# columns='Anno', values='revenue_median')

time_series_tsne_34 = time_series_tsne_long

# preparing input to TSNE
time_series_tsne= time_series_tsne_34




## For making the dataset from long to wide
# reshape dataframe from long to wide
t_tmp= time_series_tsne_34.to_numpy()
t = t_tmp[:, 1]
time_series_tsne= np.reshape(t, (7,-1)) 


###############################################################################
# 10. PLOT: most frequent Ateco code
# tranfrom dict to list 
z_z= sorted([(value, key) for (key, value) in z.items()], reverse = True)
np_arr = np.array(z_z[:20])
np_arr_df = pd.DataFrame(np_arr,columns = ['frequency', 'labels'])
np_arr_df['macroareas'] = np_arr_df.index
np_arr_df['labels'] = pd.to_numeric(np_arr_df['labels'], downcast='integer')

#  plot dataframe of labels frequency to a horizontal barplot
sns.catplot(y = 'macroareas', x = 'frequency', data = np_arr_df,  
            kind='bar', orient ='h')
plt.yticks(range(0, 20), np_arr_df['labels'])
#plt.ylabel('sector')

# extraction of ateco code (not cumulative)
y = []
for x in df_3['first two ateco']:
    try: 
        y.append(int(x))
    except ValueError: 
        pass
#print(y)

# microareas: cumulative distribution of microareas with xticks
sns.set()
sns.set_style('ticks')
pic= sns.distplot(y, bins = 100)
plt.xticks(range(1, 101, 2), range(1,101,2), rotation =70)














    

  


























#%reset -f

