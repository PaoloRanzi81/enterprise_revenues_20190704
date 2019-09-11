"""
Analysese of the following data-set: 
    
23_seac_47_ateco
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

if RELEASE == '4.18.0-25-generic': # Linux laptop
   #BASE_DIR = '/home/paolo/Dropbox/analyses/python/tensorflow/seac_data/'
   BASE_DIR = '/home/paolo/Dropbox/analyses/python/tensorflow/seac_data/136_seac_hr_ateco'
   #BASE_DIR_1 = '/home/paolo/Dropbox/analyses/python/tensorflow/seac_data/61_seac_hr_ateco'
   
else:
   BASE_DIR = '/home/Lan/paolo_scripts/exp_seasonality/seac_data'

BASE_REP = BASE_DIR







## 2. IMPORTING MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


## 3. LOAD 
# load cleaned dataset    
df_data_criteria = pd.read_csv(os.path.sep.join([BASE_DIR, 
                                     'data_criteria_01.csv']))




"""
## 3. LOAD 
# load cleaned dataset    
df_2_A = pd.read_csv(os.path.sep.join([BASE_DIR, 
                                     'cv_results.csv']))

df_2_B = pd.read_csv(os.path.sep.join([BASE_DIR_1, 
                                     'cv_results.csv']))     
    

median_MCC_A = np.median(df_2_A.loc[:, 'mean_test_matthews_corrcoef'])
median_MCC_B = np.median(df_2_B.loc[:, 'mean_test_matthews_corrcoef'])
median_log_loss_A = np.median(df_2_A.loc[:, 'mean_test_log_loss'])
median_log_lossB = np.median(df_2_B.loc[:, 'mean_test_log_loss'])

# summary statistics
summary_statistics_A = df_2_A.describe()
summary_statistics_B = df_2_B.describe()


##############################################################################
## EXPLORATORY DATA  ANALYSIS (EDA): ECDF + PDF visualization
   
## Empirical Cumulative Density Function (ECDF) for each year overlapping
# on the same plot all the Ateco codes distribution. 
from mlxtend.plotting import ecdf    
    
ax, _, _ = ecdf(x = df_2_A.loc[:, 'mean_test_matthews_corrcoef'], 
                x_label='MCC')
ax, _, _ = ecdf(x = df_2_B.loc[:, 'mean_test_matthews_corrcoef'], 
                x_label='MCC')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles,
          labels = ['A', 'B'], 
          framealpha=0.3, scatterpoints=1, loc='upper left')

#pdf = plt.title('year = %s' %  (year))
# close pic in order to avoid overwriting with previous pics
plt.clf()
"""



  
"""
## Probability Density Function (PDF) for each year overlapping
# on the same plot all the Ateco codes distribution.
pdf = plt.figure(figsize=(26,14))
# double-check that each enterprise has a consistent Ateco code for all 4 years
for ateco_120 in pd.unique(df_2['correct_ateco']):
    ateco_first = df_2[df_2['correct_ateco']== ateco_120]
    for year in (ateco_first.columns[0:4]):
        ateco_first_year = ateco_first.loc[:, year]
        pdf = sns.distplot(ateco_first_year, bins = 30)
        pdf = plt.title('ateco code = %.0f year = %s' %  (ateco_120, year))
        date = str(datetime.datetime.now())
        plt.savefig(os.path.sep.join([BASE_REP, 
                          date[0:10]+ "_" + date[11:len(date)]+".jpg"]))
        # close pic in order to avoid overwriting with previous pics
        plt.clf()
"""



###############################################################################
## QUANTILES OF ATECO CODES

# load cleaned dataset    
df_complete = pd.read_csv(os.path.sep.join([BASE_DIR, 
                                     'complete_list_ateco_codes.csv']))


# convert columns with int64 to string
df_complete['correct_ateco'] = df_complete['correct_ateco'].astype(str)

    
# append a new column with the first two Ateco digits.
df_complete['first_two_ateco'] =\
df_complete['correct_ateco'].apply(lambda ateco: ateco[0:2])
    

# sub-sample 
df_3 = df_complete[df_complete['first_two_ateco'] == '47']


# Plot 'median_VolumeAffari'
sns.distplot(df_3.loc[:, 'median_VolumeAffari'], rug=True)


# delete extreme outliers
df_4 = df_3.loc[(df_3['median_VolumeAffari'] > -0.9) & (df_3['median_VolumeAffari'] < 0.9)]
df_4.reset_index(drop=True, inplace=True)

# Plot 'median_VolumeAffari'
#sns.distplot(df_4.loc[:, 'median_VolumeAffari'], rug=True)

# compute 10 quantiles:
df_4_series = pd.Series(df_4.loc[:, 'median_VolumeAffari'])
quantiles_1 = df_4_series.quantile(np.linspace(0.1, 1, 10, dtype = float), interpolation='nearest')
quantiles_index = pd.DataFrame(quantiles_1.index, columns = ['quantiles'], dtype = 'float32')
quantiles_1.reset_index(drop=True, inplace=True) 


# concatanate quantiles with quantiles' percentages
quantiles = pd.concat([quantiles_1, quantiles_index], axis = 1) 


# divide 'median_VolumeAffari' in quantiles
df_4_intervals_1 = pd.qcut(x = df_4.loc[:, 'median_VolumeAffari'], 
                          q = np.linspace(0.1, 1, 10, dtype = float), 
                          labels = False, retbins = False)
df_4_intervals_1.reset_index(drop=True, inplace=True) 
df_4_intervals = pd.DataFrame(df_4_intervals_1)
df_4_intervals.columns = ['quantiles_groups']


# concatanate quantiles with quantiles' percentages
df_4 = pd.concat([df_4, df_4_intervals], axis = 1) 


# ateco codes belongig to a specific quantile (put the number one-by-one manually)
ateco_codes_quantiles = df_4[df_4['quantiles_groups'] == 5].correct_ateco





# delete outliers from original dataset
#threshold = pd.Series(clusterer.outlier_scores_).quantile(0.95)
#outliers = np.where(clusterer.outlier_scores_ > threshold)[0]

   
