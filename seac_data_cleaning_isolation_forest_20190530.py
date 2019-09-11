"""
Seac dataset: 

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


# reduce the number of threads used by each CPU by intervening on OpenBLAS. 
# In order to avoid multi-threading (thus sucking all server's CPU power) run 
# the following command before importing Numpy. Lastly set n_jobs to a small 
# number (thus freeing up resources, but slowing down computation).
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
#os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
#np.show_config()




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
df_2['PROV'] = df_2['PROV'].astype(str)
 

# in 'PARTITA IVA' remove trailing '.0' 
df_2['PARTITA IVA'] = df_2['PARTITA IVA'].replace({ '(\.\d*?)0+$': ''}, 
    regex=True)


# Standardize values: RobustScaler. z-scores and MixMax gave poor results with
#MDS and tSNE. Better use RobustScaler().
from sklearn.preprocessing import RobustScaler
df_2.loc[:, ['VolumeAffari']] = RobustScaler(quantile_range=(25, 75)).fit_transform\
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



"""
###############################################################################
## 2.7. TIME-SERIES SINGLE-CODE: (563 x 4) 

# pivot table 
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
df_3 = time_series_tsne_1[time_series_tsne_1['first two ateco'] == '47']
 
  
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


# concatenate lsit as a new DataFrame column
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
## 2.4. TIME-SERIES AGGREGATED: build a 22 (Ateco code) X 4 (years) array

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
# 4. TIME-SERIES AGGREGATED: build a 7 (Ateco code) X 4 (years) array
# by sub-setting by yeras  and by first 7 most frequent Ateco's codes. Only 
# 'first two ateco' with > 120 istances (=> > 30 istances for each year)

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
"""



###############################################################################
## 5. TIME-SERIES HIGHEST-REVENUE ENTERPRISES: 6-digits ateco most frequent code 
# build a 555 (Ateco code) X 4 (years) array

###############################################################################
## 2.6. 'HIGHEST-REVENUE' ENTERPRISES SUB-SET  
# subsetting with the 3 highest revenues enterprise by Ateco's code 
# summing the 4 years altogether.

total_single_ateco = dict()
for ateco in pd.unique(df_2['correct_ateco']):
        single_ateco = df_2[df_2['correct_ateco'] == ateco]
        ateco_median = np.median(single_ateco['VolumeAffari'])
        total_single_ateco.update({ateco: ateco_median})


# converting dictionary to DataFrame
total_single_ateco_df = pd.DataFrame.from_dict(total_single_ateco, 
                                               orient='index', 
                                               columns =['VolumeAffari'])
total_single_ateco_df['correct_ateco'] = total_single_ateco_df.index
total_single_ateco_df.reset_index(drop=True, inplace=True) 


# change name (because the pipeline below depends on criteria_1). I should 
# change the whole code below, but I am lazy and it works fine anyhow ;-) !
criteria_1  = total_single_ateco_df



# build a vector with the 4-years median revenue for the 6-digits Ateco code
# within the "criteria" DataFrame
criteria_std_column  = pd.DataFrame()
for ateco in criteria_1['correct_ateco']:
    sd_4_years = df_2_std[df_2_std['correct_ateco'] == ateco]
    criteria_std_column = criteria_std_column.append(sd_4_years, 
                                                     ignore_index=True) 


# reset index for "criteria" DataFrame 
criteria_1.reset_index(drop=True, inplace=True)


# merge the two DataFrames 
criteria_2= pd.merge(criteria_1, criteria_std_column, how ='inner', 
                     on = 'correct_ateco')


# Is it the following step necessary?
# build DataFrame by subsetting it by "criteria" DataFrame
df_4_0_10 = pd.DataFrame()  
for ateco_120 in criteria_2['correct_ateco']:
    ateco_first = df_2[df_2['correct_ateco']== ateco_120]
    df_4_0_10 = df_4_0_10.append(ateco_first, ignore_index=True) 


# pivot table 
time_series_tsne_1= pd.pivot_table(df_2, 
#time_series_tsne_1= pd.pivot_table(df_4_0_10, 
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
# print(np.isnan(time_series_tsne_1[time_series_tsne_1['first two ateco']]).any())
# print(np.isnan(time_series_tsne_1[time_series_tsne_1['correct_ateco']]).any())
# print(np.isnan(time_series_tsne_1[time_series_tsne_1['2017']]).any())


# count the frequency of 6-digits Ateco codes.    
z_six_digits_sub = defaultdict(int)
for x in time_series_tsne_1['correct_ateco']:
    z_six_digits_sub[x] += 1


# converting dictionary to DataFrame
z_six_digits_criteria = pd.DataFrame.from_dict(z_six_digits_sub, orient='index',
                           columns =['Frequency'])
z_six_digits_criteria['correct_ateco'] = z_six_digits_criteria.index
z_six_digits_criteria.reset_index(drop=True, inplace=True) 


# merge the two DataFrames 
criteria_3= pd.merge(criteria_2, z_six_digits_criteria, how ='inner', 
                     on = 'correct_ateco')


# select the 'Frequency' with > 30 istances (30 instances * 4 years)
criteria_4 = criteria_3[criteria_3.Frequency > 30]

# reset index
criteria_4.reset_index(drop=True, inplace=True)

# build DataFrame by subsetting it by "criteria" DataFrame
df_4_0_11 = pd.DataFrame()  

criteria = (criteria_4.sort_values(by= ['VolumeAffari'], 
                                       ascending = False)[:5])
#criteria = (criteria_4.sort_values(by= ['Frequency'], 
#                                       ascending = False)[:10])
#criteria = (z_df_30.sort_values(by= ['sd_4_years'], 
#                                       ascending = True)[:10])
criteria.reset_index(drop=True, inplace=True)
for ateco_120 in criteria['correct_ateco']:
    if ateco_120 == '473000':  # exclude a particualry big ateco code 
        pass
    elif ateco_120 == '561011':    
        pass
    else:    
        ateco_first = time_series_tsne_1[time_series_tsne_1['correct_ateco']== ateco_120]
        df_4_0_11 = df_4_0_11.append(ateco_first, ignore_index=True) 


# convert DataFrame to input (not Numpy surprisingly) to feed into t-SNE
time_series_tsne = df_4_0_11.copy()  

# print summary statistics of the ateco groups used
print(criteria)

###############################################################################
# 2.6. metric MultiDimensional Scaling (MDS) 


## FIRST ROUND CLEANING    
# import libraries 
from sklearn.manifold import MDS  
from multiprocessing import cpu_count

# initialize MDS parameters: 
n_components = 2
metric = True
#n_init = 2
n_init = 64
#max_iter = 30
max_iter = 300000
verbose = 1
eps = 0.001
n_jobs = int(round(((cpu_count()/4)-10),0))
#n_jobs = -4 # valid for MDS only. For Isolation Forest has been set to 1.
random_state = None # valid both for MDS and Isolation Forest
dissimilarity = 'euclidean'


# initialize Isolation Forest parameters: 
max_samples ='auto'
n_estimators = 100000
#n_estimators = 100 
behaviour ='new' 
contamination = 0.1
bootstrap = True 


## COMPUTE MDS AGAIN (first time)
# initialize MDS model
mds = MDS(n_components = n_components, 
          metric = metric, 
          n_init = n_init, 
          max_iter = max_iter, 
          verbose = verbose, 
          eps = eps, 
          n_jobs = n_jobs, 
          random_state = random_state, 
          dissimilarity = dissimilarity)


# fit model
Y =[]
Y = mds.fit_transform(time_series_tsne.loc[:, ['2014','2015','2016','2017']])


# MDS stress metric
mds_stress = round(mds.stress_, 2)
print(mds_stress)



## OUTLIER DELETION: ISOLATION FOREST
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import f1_score
#from sklearn.metrics import accuracy_score, matthews_corrcoef, r2_score
from sklearn.metrics import make_scorer


# initialize classifier
clf = IsolationForest(max_samples = max_samples,
                      n_estimators = n_estimators, 
                      behaviour = behaviour, 
                      contamination = contamination,
                      bootstrap = bootstrap, 
                      random_state = random_state, 
                      n_jobs = 1)
clf.fit(Y)
y_pred = clf.predict(Y)

## GETTING RID OF OUTLIERS from original dataset
dataset_no_outliers_01 = pd.DataFrame()  
for enterprise_id in time_series_tsne['CodDitta']:
    single_enterprise =\
    time_series_tsne[time_series_tsne['CodDitta'] == enterprise_id]
    index =\
    time_series_tsne[time_series_tsne['CodDitta'] == enterprise_id].index
    if y_pred[index] == 1:
        dataset_no_outliers_01 =\
        dataset_no_outliers_01.append(single_enterprise, ignore_index=True)
        

# reset index
dataset_no_outliers_01.reset_index(drop=True, inplace=True)

# save cleaned DataFrame as .csv file
dataset_no_outliers_01.to_csv(os.path.sep.join([BASE_DIR, 
                                     'data_criteria_01.csv']), index= False)

    
## COMPUTE MDS AGAIN (second time)
time_series_tsne = pd.DataFrame()  
time_series_tsne = dataset_no_outliers_01.copy()


## COUNT THE FREQUENCY (second time) of 6-digits Ateco codes.    
z_six_digits_01 = defaultdict(int)
for x in time_series_tsne['correct_ateco']:
    z_six_digits_01[x] += 1
    
# converting dictionary to DataFrame
z_six_digits_criteria_01 = pd.DataFrame.from_dict(z_six_digits_01, orient='index',
                           columns =['Frequency'])
z_six_digits_criteria_01['correct_ateco'] = z_six_digits_criteria_01.index
z_six_digits_criteria_01.reset_index(drop=True, inplace=True) 


# select the 'Frequency' with > 30 istances (30 instances * 4 years)
criteria_01 = z_six_digits_criteria_01[z_six_digits_criteria_01.Frequency > 30]

# reset index
criteria_01.reset_index(drop=True, inplace=True)

# build DataFrame by subsetting it by "criteria" DataFrame
df_4_0_01 = pd.DataFrame()  
for ateco_120 in criteria_01['correct_ateco']:
    ateco_first = time_series_tsne[time_series_tsne['correct_ateco']== ateco_120]
    df_4_0_01 = df_4_0_01.append(ateco_first, ignore_index=True) 


# convert DataFrame to input (not Numpy surprisingly) to feed into t-SNE
time_series_tsne = df_4_0_01.copy()  

# print summary statistics of the ateco groups used
print(criteria_01)

# save 'criteria_01' as .csv   
criteria_01.to_csv(os.path.sep.join([BASE_DIR, 
                                     'criteria_01.csv']), index= False)

# fit model
Y =[]
Y = mds.fit_transform(time_series_tsne.loc[:, ['2014','2015','2016','2017']])


# save Y as .csv   
Y_01 = pd.DataFrame(Y)
Y_01.to_csv(os.path.sep.join([BASE_DIR, 
                                     'Y_criteria_01.csv']), index= False)

# MDS stress metric
mds_stress = round(mds.stress_, 2)
print(mds_stress)

    
## PLOT FIRST ROUND RESULT 
time_series_tsne['tsne-2d-one'] = Y[:,0]
time_series_tsne['tsne-2d-two'] = Y[:,1]

time_series_tsne['MDS-2d-one'] = time_series_tsne['tsne-2d-one'] 
time_series_tsne['MDS-2d-two'] = time_series_tsne['tsne-2d-two']


mds_plot = plt.figure(figsize=(16,10))
mds_plot = sns.scatterplot(
    x="MDS-2d-one", y="MDS-2d-two",
    
    hue="correct_ateco",
    
    # as follows variables to used only when we know we have JUST 4 Ateco codes!
    #style="correct_ateco",
    #markers= ('s', '^', 'o', 'X'),
    #palette=sns.color_palette("muted", criteria_01.shape[0]), 
    
    palette=sns.color_palette("hls", criteria_01.shape[0]),
    data = time_series_tsne,
    legend = "full",
    alpha = 0.9 
)


# add title                   
mds_plot = plt.title('mds_stress = %.2f\n\
                     first round cleaning' %  (mds_stress))
#plt.show()


# GENERATE FIGURES 
date = str(datetime.datetime.now())
plt.savefig(os.path.sep.join([BASE_REP, 
                          date[0:10]+ "_" + date[11:len(date)]+".jpg"]))

# close pic in order to avoid overwriting with previous pics
plt.clf()






## SECOND ROUND CLEANING    
## OUTLIER DELETION: ISOLATION FOREST

# initialize classifier
clf = IsolationForest(max_samples = max_samples,
                      n_estimators = n_estimators, 
                      behaviour = behaviour, 
                      contamination = contamination,
                      bootstrap = bootstrap, 
                      random_state = random_state, 
                      n_jobs = 1)
clf.fit(Y)
y_pred = []
y_pred = clf.predict(Y)


## GETTING RID OF OUTLIERS from original dataset
dataset_no_outliers_02 = pd.DataFrame()  
for enterprise_id in time_series_tsne['CodDitta']:
    single_enterprise = time_series_tsne[time_series_tsne['CodDitta'] == enterprise_id]
    index = time_series_tsne[time_series_tsne['CodDitta'] == enterprise_id].index
    if y_pred[index] == 1:
        dataset_no_outliers_02 = dataset_no_outliers_02.append(single_enterprise, ignore_index=True)
        

# reset index
dataset_no_outliers_02.reset_index(drop=True, inplace=True)

# save cleaned DataFrame as .csv file
dataset_no_outliers_02.to_csv(os.path.sep.join([BASE_DIR, 
                                     'data_criteria_02.csv']), index= False)

## COMPUTE MDS AGAIN (third time)
time_series_tsne = pd.DataFrame()  
time_series_tsne = dataset_no_outliers_02.copy()


## COUNT THE FREQUENCY (third time) of 6-digits Ateco codes.    
z_six_digits_02 = defaultdict(int)
for x in time_series_tsne['correct_ateco']:
    z_six_digits_02[x] += 1
    
# converting dictionary to DataFrame
z_six_digits_criteria_02 = pd.DataFrame.from_dict(z_six_digits_02, orient='index',
                           columns =['Frequency'])
z_six_digits_criteria_02['correct_ateco'] = z_six_digits_criteria_02.index
z_six_digits_criteria_02.reset_index(drop=True, inplace=True) 


# select the 'Frequency' with > 30 istances (30 instances * 4 years)
criteria_02 = z_six_digits_criteria_02[z_six_digits_criteria_02.Frequency > 30]

# reset index
criteria_02.reset_index(drop=True, inplace=True)

# build DataFrame by subsetting it by "criteria" DataFrame
df_4_0_02 = pd.DataFrame()  
for ateco_120 in criteria_02['correct_ateco']:
    ateco_first = time_series_tsne[time_series_tsne['correct_ateco']== ateco_120]
    df_4_0_02 = df_4_0_02.append(ateco_first, ignore_index=True) 


# convert DataFrame to input (not Numpy surprisingly) to feed into t-SNE
time_series_tsne = df_4_0_02.copy()  


# print summary statistics of the ateco groups used
print(criteria_02)


# save 'criteria_02' as .csv   
criteria_02.to_csv(os.path.sep.join([BASE_DIR, 
                                     'criteria_02.csv']), index= False)


# fit model
Y =[]
Y = mds.fit_transform(time_series_tsne.loc[:, ['2014','2015','2016','2017']])


# save Y as .csv   
Y_02 = pd.DataFrame(Y)
Y_02.to_csv(os.path.sep.join([BASE_DIR, 
                                     'Y_criteria_02.csv']), index= False)

# MDS stress metric
mds_stress = round(mds.stress_, 2)
print(mds_stress)

    
## PLOT SECOND ROUND RESULT 
time_series_tsne = pd.DataFrame()
time_series_tsne = dataset_no_outliers_02.copy() 
time_series_tsne['tsne-2d-one'] = Y[:,0]
time_series_tsne['tsne-2d-two'] = Y[:,1]

time_series_tsne['MDS-2d-one'] = time_series_tsne['tsne-2d-one'] 
time_series_tsne['MDS-2d-two'] = time_series_tsne['tsne-2d-two']


mds_plot = plt.figure(figsize=(16,10))
mds_plot = sns.scatterplot(
    x="MDS-2d-one", y="MDS-2d-two",
    
    hue="correct_ateco",
    
    # as follows variables to used only when we know we have JUST 4 Ateco codes!
    #style="correct_ateco",
    #markers= ('s', '^', 'o', 'X'),
    #palette=sns.color_palette("muted", criteria_02.shape[0]), 
    
    palette=sns.color_palette("hls", criteria_02.shape[0]),
    data = time_series_tsne,
    legend = "full",
    alpha = 0.9 
)


# add title                   
mds_plot = plt.title('mds_stress = %.2f\n\
                     second round cleaning' %  (mds_stress))
plt.show()


# GENERATE FIGURES 
date = str(datetime.datetime.now())
plt.savefig(os.path.sep.join([BASE_REP, 
                          date[0:10]+ "_" + date[11:len(date)]+".jpg"]))

# close pic in order to avoid overwriting with previous pics
plt.clf()






## THIRD ROUND CLEANING    
## OUTLIER DELETION: ISOLATION FOREST

# initialize classifier
clf = IsolationForest(max_samples = max_samples,
                      n_estimators = n_estimators, 
                      behaviour = behaviour, 
                      contamination = contamination,
                      bootstrap = bootstrap, 
                      random_state = random_state, 
                      n_jobs = 1)
clf.fit(Y)
y_pred = []
y_pred = clf.predict(Y)


## GETTING RID OF OUTLIERS from original dataset
dataset_no_outliers_03 = pd.DataFrame()  
for enterprise_id in time_series_tsne['CodDitta']:
    single_enterprise = time_series_tsne[time_series_tsne['CodDitta'] == enterprise_id]
    index = time_series_tsne[time_series_tsne['CodDitta'] == enterprise_id].index
    if y_pred[index] == 1:
        dataset_no_outliers_03 = dataset_no_outliers_03.append(single_enterprise, ignore_index=True)
        

# reset index
dataset_no_outliers_03.reset_index(drop=True, inplace=True)

# save cleaned DataFrame as .csv file
dataset_no_outliers_03.to_csv(os.path.sep.join([BASE_DIR, 
                                     'data_criteria_03.csv']), index= False)


    
    

## COMPUTE MDS AGAIN (third time)
time_series_tsne = pd.DataFrame()  
time_series_tsne = dataset_no_outliers_03.copy()


## COUNT THE FREQUENCY (third time) of 6-digits Ateco codes.    
z_six_digits_03 = defaultdict(int)
for x in time_series_tsne['correct_ateco']:
    z_six_digits_03[x] += 1
    
# converting dictionary to DataFrame
z_six_digits_criteria_03 = pd.DataFrame.from_dict(z_six_digits_03, orient='index',
                           columns =['Frequency'])
z_six_digits_criteria_03['correct_ateco'] = z_six_digits_criteria_03.index
z_six_digits_criteria_03.reset_index(drop=True, inplace=True) 


# select the 'Frequency' with > 30 istances (30 instances * 4 years)
criteria_03 = z_six_digits_criteria_03[z_six_digits_criteria_03.Frequency > 30]

# reset index
criteria_03.reset_index(drop=True, inplace=True)

# build DataFrame by subsetting it by "criteria" DataFrame
df_4_0_03 = pd.DataFrame()  
for ateco_120 in criteria_03['correct_ateco']:
    ateco_first = time_series_tsne[time_series_tsne['correct_ateco']== ateco_120]
    df_4_0_03 = df_4_0_03.append(ateco_first, ignore_index=True) 


# convert DataFrame to input (not Numpy surprisingly) to feed into t-SNE
time_series_tsne = df_4_0_03.copy()  


# print summary statistics of the ateco groups used
print(criteria_03)


# save 'criteria_03' as .csv   
criteria_03.to_csv(os.path.sep.join([BASE_DIR, 
                                     'criteria_03.csv']), index= False)


# fit model
Y =[]
Y = mds.fit_transform(time_series_tsne.loc[:, ['2014','2015','2016','2017']])


# save Y as .csv 
Y_03 = pd.DataFrame(Y)  
Y_03.to_csv(os.path.sep.join([BASE_DIR, 
                                     'Y_criteria_03.csv']), index= False)

# MDS stress metric
mds_stress = round(mds.stress_, 2)
print(mds_stress)

    
## PLOT THIRD ROUND RESULT 
time_series_tsne = pd.DataFrame()
time_series_tsne = dataset_no_outliers_03.copy() 
time_series_tsne['tsne-2d-one'] = Y[:,0]
time_series_tsne['tsne-2d-two'] = Y[:,1]

time_series_tsne['MDS-2d-one'] = time_series_tsne['tsne-2d-one'] 
time_series_tsne['MDS-2d-two'] = time_series_tsne['tsne-2d-two']


mds_plot = plt.figure(figsize=(16,10))
mds_plot = sns.scatterplot(
    x="MDS-2d-one", y="MDS-2d-two",
    
    hue="correct_ateco",
    
    # as follows variables to used only when we know we have JUST 4 Ateco codes!
    #style="correct_ateco",
    #markers= ('s', '^', 'o', 'X'),
    #palette=sns.color_palette("muted", criteria_03.shape[0]),
    
    palette=sns.color_palette("hls", criteria_03.shape[0]),
    data = time_series_tsne,
    legend = "full",
    alpha = 0.9 
)


# add title                   
mds_plot = plt.title('mds_stress = %.2f\n\
                     third round cleaning' %  (mds_stress))
plt.show()


# GENERATE FIGURES 
date = str(datetime.datetime.now())
plt.savefig(os.path.sep.join([BASE_REP, 
                          date[0:10]+ "_" + date[11:len(date)]+".jpg"]))

# close pic in order to avoid overwriting with previous pics
plt.clf()





## FOURTH ROUND CLEANING    

## OUTLIER DELETION: ISOLATION FOREST

# initialize classifier
clf = IsolationForest(max_samples = max_samples,
                      n_estimators = n_estimators, 
                      behaviour = behaviour, 
                      contamination = contamination,
                      bootstrap = bootstrap, 
                      random_state = random_state, 
                      n_jobs = 1)
clf.fit(Y)
y_pred = []
y_pred = clf.predict(Y)


## GETTING RID OF OUTLIERS from original dataset
dataset_no_outliers_04 = pd.DataFrame()  
for enterprise_id in time_series_tsne['CodDitta']:
    single_enterprise = time_series_tsne[time_series_tsne['CodDitta'] == enterprise_id]
    index = time_series_tsne[time_series_tsne['CodDitta'] == enterprise_id].index
    if y_pred[index] == 1:
        dataset_no_outliers_04 = dataset_no_outliers_04.append(single_enterprise, ignore_index=True)
        

# reset index
dataset_no_outliers_04.reset_index(drop=True, inplace=True)

# save cleaned DataFrame as .csv file
dataset_no_outliers_04.to_csv(os.path.sep.join([BASE_DIR, 
                                     'data_criteria_04.csv']), index= False)



## COMPUTE MDS AGAIN (third time)
time_series_tsne = pd.DataFrame()  
time_series_tsne = dataset_no_outliers_04.copy()


## COUNT THE FREQUENCY (third time) of 6-digits Ateco codes.    
z_six_digits_04 = defaultdict(int)
for x in time_series_tsne['correct_ateco']:
    z_six_digits_04[x] += 1
    
# converting dictionary to DataFrame
z_six_digits_criteria_04 = pd.DataFrame.from_dict(z_six_digits_04, orient='index',
                           columns =['Frequency'])
z_six_digits_criteria_04['correct_ateco'] = z_six_digits_criteria_04.index
z_six_digits_criteria_04.reset_index(drop=True, inplace=True) 


# select the 'Frequency' with > 30 istances (30 instances * 4 years)
criteria_04 = z_six_digits_criteria_04[z_six_digits_criteria_04.Frequency > 30]

# reset index
criteria_04.reset_index(drop=True, inplace=True)

# build DataFrame by subsetting it by "criteria" DataFrame
df_4_0_04 = pd.DataFrame()  
for ateco_120 in criteria_04['correct_ateco']:
    ateco_first = time_series_tsne[time_series_tsne['correct_ateco']== ateco_120]
    df_4_0_04 = df_4_0_04.append(ateco_first, ignore_index=True) 


# convert DataFrame to input (not Numpy surprisingly) to feed into t-SNE
time_series_tsne = df_4_0_04.copy()  

# print summary statistics of the ateco groups used
print(criteria_04)


# save 'criteria_04' as .csv   
criteria_04.to_csv(os.path.sep.join([BASE_DIR, 
                                     'criteria_04.csv']), index= False)


# fit model
Y =[]
Y = mds.fit_transform(time_series_tsne.loc[:, ['2014','2015','2016','2017']])


# save Y as .csv 
Y_04 = pd.DataFrame(Y)  
Y_04.to_csv(os.path.sep.join([BASE_DIR, 
                                     'Y_criteria_04.csv']), index= False)

# MDS stress metric
mds_stress = round(mds.stress_, 2)
print(mds_stress)

    
## PLOT FOURTH ROUND RESULT 
time_series_tsne = pd.DataFrame()
time_series_tsne = dataset_no_outliers_04.copy() 
time_series_tsne['tsne-2d-one'] = Y[:,0]
time_series_tsne['tsne-2d-two'] = Y[:,1]

time_series_tsne['MDS-2d-one'] = time_series_tsne['tsne-2d-one'] 
time_series_tsne['MDS-2d-two'] = time_series_tsne['tsne-2d-two']


mds_plot = plt.figure(figsize=(16,10))
mds_plot = sns.scatterplot(
    x="MDS-2d-one", y="MDS-2d-two",
    
    hue="correct_ateco",
    
    # as follows variables to used only when we know we have JUST 4 Ateco codes!
    #style="correct_ateco",
    #markers= ('s', '^', 'o', 'X'),
    #palette=sns.color_palette("muted", criteria_04.shape[0]),
    
    palette=sns.color_palette("hls", criteria_04.shape[0]),
    data = time_series_tsne,
    legend = "full",
    alpha = 0.9 
)


# add title                   
mds_plot = plt.title('mds_stress = %.2f\n\
                     fourth round cleaning' %  (mds_stress))
plt.show()


# GENERATE FIGURES 
date = str(datetime.datetime.now())
plt.savefig(os.path.sep.join([BASE_REP, 
                          date[0:10]+ "_" + date[11:len(date)]+".jpg"]))

# close pic in order to avoid overwriting with previous pics
plt.clf()


"""
# for loop for all labels (e.g. Ateco codes) in order to add them.
for line in range(0, time_series_tsne.loc[:,['correct_ateco']].shape[0]):
     mds_plot.text(time_series_tsne.loc[line,['tsne-2d-one']]+0.0001,
                    time_series_tsne.loc[line,['tsne-2d-two']]+0.0001, 
                    time_series_tsne.loc[line]['correct_ateco'], 
                    horizontalalignment='left', size='medium') 
"""



"""
# add title                   
tsne_plot = plt.title('perplexity = %.0f\n\
                      learning_rate = %.0f\n KL = %.8f' % \
                      (perplexity, learning_rate, 
                       rounded_kl_divergence)) 
"""























































#%reset -f

