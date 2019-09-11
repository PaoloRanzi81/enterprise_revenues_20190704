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
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
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


# WARNING: do not use standardization/z-scores when only 22 time-series 
#are computed

#PowerTransformer
from sklearn.preprocessing import PowerTransformer
df_2.loc[:, ['VolumeAffari']] = PowerTransformer(method='box-cox',).fit_transform\
(df_2.loc[:, ['VolumeAffari']])


"""
#Normalizer (NOT WORKING)
from sklearn.preprocessing import Normalizer
df_2.loc[:, ['VolumeAffari']] = Normalizer(norm ='l2',).fit_transform\
(df_2.loc[:, ['VolumeAffari']])
"""


"""
# QuantileTransformer
from sklearn.preprocessing import QuantileTransformer
df_2.loc[:, ['VolumeAffari']] = QuantileTransformer(n_quantiles = (df_2.shape[0]/4),
        output_distribution = 'normal').fit_transform\
(df_2.loc[:, ['VolumeAffari']])
"""


"""
# RobustScaler
from sklearn.preprocessing import RobustScaler
df_2.loc[:, ['VolumeAffari']] = RobustScaler(quantile_range=(25, 75)).fit_transform\
(df_2.loc[:, ['VolumeAffari']])
"""

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
## 2.7. TIME-SERIES SINGLE-CODE: (single enterprises x 4 years) 

# computing the standardized median gross revenue for each Ateco's code 
# for all 4 years 
total_single_ateco = dict()
for ateco in pd.unique(df_2['correct_ateco']):
        single_ateco = df_2[df_2['correct_ateco'] == ateco]
        ateco_median = np.median(single_ateco['VolumeAffari'])
        total_single_ateco.update({ateco: ateco_median})


# converting dictionary to DataFrame
total_single_ateco_df = pd.DataFrame.from_dict(total_single_ateco, 
                                               orient='index', 
                                               columns =['median_VolumeAffari'])
total_single_ateco_df['correct_ateco'] = total_single_ateco_df.index
total_single_ateco_df.reset_index(drop=True, inplace=True) 


# computing the standardized median gross revenue in Euro for each Ateco's code 
# for all 4 years 
# load cleaned dataset    
df_5 = pd.read_csv(os.path.sep.join([BASE_DIR, 
                                     'clean_data_1.csv']))

    
# convert columns with int64 to string
df_5['correct_ateco'] = df_5['correct_ateco'].astype(str)    


# Dictionary of all median gross revenue per Ateco code    
total_single_ateco_1 = dict()
for ateco in pd.unique(df_5['correct_ateco']):
        single_ateco_1 = df_5[df_5['correct_ateco'] == ateco]
        ateco_median = round(np.median(single_ateco_1['VolumeAffari']),2)
        total_single_ateco_1.update({ateco: ateco_median})


# converting dictionary to DataFrame
total_single_ateco_df_1 = pd.DataFrame.from_dict(total_single_ateco_1, 
                                               orient='index', 
                                               columns =['median_VolumeAffari_euro'])
total_single_ateco_df_1['correct_ateco'] = total_single_ateco_df_1.index
total_single_ateco_df_1.reset_index(drop=True, inplace=True) 


# change name (because the pipeline below depends on criteria_1). I should 
# change the whole code below, but I am lazy and it works fine anyhow ;-) !
criteria_1  = total_single_ateco_df


# reset index for "criteria" DataFrame 
criteria_1.reset_index(drop=True, inplace=True)


# build a vector with the standard deviation of 4-years median revenue for each 
# the 6-digits Ateco code
criteria_std_column  = pd.DataFrame()
for ateco in criteria_1['correct_ateco']:
    sd_4_years = df_2_std[df_2_std['correct_ateco'] == ateco]
    criteria_std_column = criteria_std_column.append(sd_4_years, 
                                                     ignore_index=True) 


# build a DataFrame with both median gross revenue and standard deviation for 
# each for each 6-digits Ateco code
criteria_3 = pd.merge(criteria_1, criteria_std_column, how ='inner', 
                     on = 'correct_ateco')

criteria_4 = pd.merge(criteria_3, total_single_ateco_df_1, how ='inner', 
                     on = 'correct_ateco')

# re-arrange columns' names
criteria_5 = criteria_4[['correct_ateco', 'sd_4_years','median_VolumeAffari',
       'median_VolumeAffari_euro']]

# count the frequency of Ateco codes in the whole dataset.    
z_six_digits_complete = defaultdict(int)
for x in df_5['correct_ateco']:
    z_six_digits_complete[x] += 1


# converting dictionary to DataFrame
z_df_complete = pd.DataFrame.from_dict(z_six_digits_complete, orient='index',
                           columns =['Frequency'])
z_df_complete['correct_ateco'] = z_df_complete.index
z_df_complete.reset_index(drop=True, inplace=True) 
#z_df.info()


# merge
criteria_2 = pd.merge(criteria_5, z_df_complete, how ='inner', 
                     on = 'correct_ateco')


# reset index for "criteria" DataFrame 
criteria_2.reset_index(drop=True, inplace=True)


# save 'complete_list_ateco_codes' as .csv file 
#criteria_2.to_csv(os.path.sep.join([BASE_DIR, 
#                                     'complete_list_ateco_codes.csv']), index= False)


"""
# build DataFrame by subsetting it by "criteria" DataFrame
df_4_0_10 = pd.DataFrame()  
for ateco_120 in criteria_2['correct_ateco']:
    ateco_first = df_2[df_2['correct_ateco']== ateco_120]
    df_4_0_10 = df_4_0_10.append(ateco_first, ignore_index=True) 
"""

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
    

# append a new column with the first_two_ateco digits.
time_series_tsne_1['first_two_ateco'] =\
time_series_tsne_1['correct_ateco'].apply(lambda ateco: ateco[0:2])


# check whether NaN are present
# print(np.isnan(time_series_tsne[time_series_tsne['first_two_ateco']]).any())
# print(np.isnan(time_series_tsne[time_series_tsne['correct_ateco']]).any())


# count the frequency of first two digits Ateco codes.    
z_two_digits = defaultdict(int)
for x in time_series_tsne_1['first_two_ateco']:
    z_two_digits[x] += 1


##############################################################################
## CHOOSE OPTION: EITHER ONE SINGLE MACROAREA OR THE WHOLE DATA-SET
# since we have the highest number of data (5034 x 4 years) with enterprises 
# beloging to the macroarea '47', we analyze only such a macroarea 
# (i.e. first two digits Ateco code). 
df_3 = time_series_tsne_1[time_series_tsne_1['first_two_ateco'] == '47']
#df_3 = time_series_tsne_1.copy()  # to be used in case we want to use all data and not only
# a macroarea (i.e. first two digits Ateco code).

  
# count the frequency of first the Ateco codes in either our subsample (e.g. 
# macroarea 47) or the whole dataset.    
z_six_digits = defaultdict(int)
for x in df_3['correct_ateco']:
    z_six_digits[x] += 1


# converting dictionary to DataFrame
z_df = pd.DataFrame.from_dict(z_six_digits, orient='index',
                           columns =['Frequency'])
z_df['correct_ateco'] = z_df.index
z_df.reset_index(drop=True, inplace=True) 
#z_df.info()


# select the 'first_two_ateco' with > 30 istances. Useful for robust statistics.
z_df_30_tmp_1 = z_df[z_df.Frequency > 30]
z_df_30_tmp_1.reset_index(drop=True, inplace=True)


# concatenate column with standard deviation for each Ateco code selected
z_df_30_tmp_2 = pd.DataFrame()
#z_df_30_tmp_2 = []
for iii in z_df_30_tmp_1['correct_ateco']:
    single_std = df_2_std[df_2_std['correct_ateco'] == iii] 
    single_std.reset_index(drop=True, inplace=True)
    z_df_30_tmp_2 = z_df_30_tmp_2.append(single_std, ignore_index=True) 


# merge 'Frequency' + 'sd_4_years' + median of 'median_VolumeAffari' for
# each Ateco code.
z_df_30_1 = pd.merge(z_df_30_tmp_1, z_df_30_tmp_2, how ='inner', on = 'correct_ateco')
z_df_30 = pd.merge(z_df_30_1, criteria_1, how ='inner', on = 'correct_ateco')


# sub-set dataset by taking only a small subset
df_4_0_10 = pd.DataFrame() 

#criteria = (z_df_30.sort_values(by= ['Frequency'], 
#                                       ascending = False)[:5])
criteria = (z_df_30.sort_values(by= ['median_VolumeAffari'], 
                                       ascending = False)[:5])
#criteria = (z_df_30.sort_values(by= ['sd_4_years'], 
#                                       ascending = True)[:5])
criteria.reset_index(drop=True, inplace=True)


# for each 'correct_ateco' and for 4 years compute standard deviation. 
correct_ateco_numeric = pd.to_numeric(criteria.correct_ateco, errors='coerce')
for ateco_120 in correct_ateco_numeric:
    #if ateco_120 == int('472600'):  # exclude a particualry big ateco code 
    #    pass
    #elif ateco_120 == int('478201'):    
    #    pass
    #else:
        ateco_first = df_3[df_3['correct_ateco']== str(ateco_120)]
        df_4_0_10 = df_4_0_10.append(ateco_first, ignore_index=True) 
        

# convert DataFrame to input (not Numpy surprisingly) to feed into t-SNE
time_series_tsne = df_4_0_10.copy()     

# print summary statistics of the ateco groups used
print(criteria)

# save 'criteria' as .csv   
criteria.to_csv(os.path.sep.join([BASE_DIR, 
                                     'criteria.csv']), index= False)


##############################################################################    
## COMPUTING QUANTILES FOR EACH SPECIFIC ATECO MACROAREA (e.g. 47)    

# change names to meaningful ones 
df_complete = criteria_2.copy()    


# convert columns with int64 to string
df_complete['correct_ateco'] = df_complete['correct_ateco'].astype(str)

    
# append a new column with the first two Ateco digits.
df_complete['first_two_ateco'] =\
df_complete['correct_ateco'].apply(lambda ateco: ateco[0:2])
df_complete.reset_index(drop=True, inplace=True)     

# sub-sample 
df_6 = df_complete[df_complete['first_two_ateco'] == '47']  
  
# delete extreme outliers
df_4 = df_6.loc[(df_6['median_VolumeAffari'] > -0.9) & (df_6['median_VolumeAffari'] < 0.9)]
df_4.reset_index(drop=True, inplace=True) 

   
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
ateco_codes_quantiles = df_4[df_4['quantiles_groups'] == 8].correct_ateco
ateco_codes_quantiles.reset_index(drop=True, inplace=True) 


# sub-set dataset by taking only a small subset
df_4_0_10_quantile = pd.DataFrame() 

# for each 'correct_ateco' and for 4 years compute standard deviation. 
#correct_ateco_numeric_quantile = pd.to_numeric(ateco_codes_quantiles.correct_ateco, errors='coerce')
for ateco_120_quantile in ateco_codes_quantiles:
    ateco_quantile_1 = df_complete[df_complete['correct_ateco'] == ateco_120_quantile]
    
    if (ateco_quantile_1.loc[ateco_quantile_1['Frequency'] > 30]).empty:
        pass
    #if ateco_120 == int('472600'):  # exclude a particualry big ateco code 
    #    pass
    #elif ateco_120 == int('478201'):    
    #    pass
    else:
        ateco_first = df_3[df_3['correct_ateco']== str(ateco_120_quantile)]
        df_4_0_10_quantile = df_4_0_10_quantile.append(ateco_first, ignore_index=True) 

        
# convert DataFrame to input (not Numpy surprisingly) to feed into t-SNE
time_series_tsne = df_4_0_10_quantile.copy()     

# print summary statistics of the ateco groups used
print(ateco_codes_quantiles)

# save 'criteria' as .csv   
ateco_codes_quantiles.to_csv(os.path.sep.join([BASE_DIR, 
                                     'ateco_codes_quantiles.csv']), index= False)

###############################################################################
## 2.5. Principal Component Analysis (PCA) 

"""  
# import libraries 
from sklearn.decomposition import PCA


# initialize PCA parameters: 
n_components = 4
copy = True
whiten = True
svd_solver = 'full'
tol = [0.0]
iterated_power = 'auto'
random_state = None


## COMPUTE PCA (first time)
# initialize MDS model
pca = PCA(n_components = n_components, 
          copy = copy, 
          whiten = whiten, 
          svd_solver = svd_solver, 
          tol = tol, 
          iterated_power = iterated_power, 
          random_state = random_state)


# fit model
Y =[]
Y_pca = pca.fit_transform(time_series_tsne.loc[:, ['2014','2015','2016','2017']])
"""    
    
    
    
###############################################################################
## 2.6. Kernel Principal Component Analysis (PCA) 
 
# import libraries 
from sklearn.decomposition import KernelPCA
#from sklearn.decomposition import PCA
from multiprocessing import cpu_count


# initialize PCA parameters: 
n_components = 4
kernel = 'linear'
#kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed']
gamma = 1/4
degree = 1
coef0 = 1
kernel_params = None
alpha = 1.0
fit_inverse_transform = False
eigen_solver = 'dense'
#eigen_solver = ['auto', 'arpack','dense']
tol = 0
max_iter = 300000
remove_zero_eig = True
random_state = None
copy_X = True
n_jobs = int(round(((cpu_count()/4)-10),0))


## COMPUTE PCA (first time)
# initialize MDS model
kernel_pca = KernelPCA(n_components = n_components, 
          kernel = kernel, 
          gamma = gamma, 
          degree = degree, 
          coef0 = coef0, 
          kernel_params = kernel_params,
          alpha = alpha,
          fit_inverse_transform  = fit_inverse_transform ,
          eigen_solver = eigen_solver,
          tol = tol,
          max_iter  = max_iter,
          remove_zero_eig  = remove_zero_eig,
          random_state = random_state, 
          copy_X  = copy_X ,
          n_jobs = n_jobs,)


# fit model
#Y =[]
Y_pca = kernel_pca.fit_transform(time_series_tsne.loc[:, ['2014','2015','2016','2017']])

    
    
    
###############################################################################
## 2.7. Independent Component Analysis (ICA) 


## FIRST ROUND CLEANING: i) ICA  
# import libraries 
#from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
#from sklearn.manifold import LocallyLinearEmbedding
from multiprocessing import cpu_count


# initialize FastICA parameters: 
n_components = 4
algorithm  = 'parallel'
#algorithm  = ['parallel', 'deflation']
whiten = True
fun = 'logcosh'
#fun  = ['logcosh', 'exp', 'cube']
fun_args = {'alpha' : 1.0}
max_iter = 30000
tol = 1e-06
w_init = None
random_state = None


# initialize ICA model
fast_ICA = FastICA(n_components = n_components, 
          algorithm = algorithm, 
          whiten = whiten, 
          fun = fun, 
          fun_args = fun_args, 
          max_iter = max_iter,
          tol = tol, 
          w_init = w_init,
          random_state = random_state)


# fit model
#Y =[]
Y_ica = fast_ICA.fit_transform(Y_pca)
#Y = fast_ICA.fit_transform(time_series_tsne.loc[:, ['2014','2015','2016','2017']])    
    

###############################################################################
## 2.8. HDBSCAN 
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=criteria.shape[0],
                            metric = 'euclidean',
                            p = None, 
                            cluster_selection_method = 'eom', 
                            core_dist_n_jobs = int(round(((cpu_count()/4)-10),0)), 
                            approx_min_span_tree  = False).fit(Y_ica)


# plot outliers
#sns.distplot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], rug=True)


# delete outliers from original dataset
threshold = pd.Series(clusterer.outlier_scores_).quantile(0.95)
outliers = np.where(clusterer.outlier_scores_ > threshold)[0]




# drop outliers 
time_series_tsne = time_series_tsne.drop(index = outliers)

# reset index
time_series_tsne.reset_index(drop=True, inplace=True)

# re-run kernel PCA on the new dataset
Y_pca =[]
Y_pca = kernel_pca.fit_transform(time_series_tsne.loc[:, ['2014','2015','2016','2017']])

    
# re-run ICA on the new dataset    
# fit model
Y_ica =[]
Y_ica = fast_ICA.fit_transform(Y_pca)    
#Y = fast_ICA.fit_transform(time_series_tsne.loc[:, ['2014','2015','2016','2017']])    
   

###############################################################################
## 2.8. ISOLATION FOREST (IF) 

# import libraries    
from sklearn.ensemble import IsolationForest

    
# initialize Isolation Forest parameters: 
n_estimators = 100000
max_samples = 1.0
#max_samples ='auto'
contamination = 0.2
max_features = 1.0
bootstrap = True
n_jobs = int(round(((cpu_count()/2)-2),0))
behaviour ='new' 
random_state = None
verbose = 0 
warm_start = True    
  

# initialize classifier
clf = IsolationForest(n_estimators = n_estimators, 
                      max_samples = max_samples,
                      contamination = contamination,
                      max_features = max_features, 
                      bootstrap = bootstrap,
                      n_jobs = n_jobs,
                      behaviour = behaviour,                                      
                      random_state = random_state,
                      verbose = verbose, 
                      warm_start = warm_start)


# fit 
clf.fit(Y_ica)
y_pred = clf.predict(Y_ica)    
    

###############################################################################    
## 2.9. GETTING RID OF OUTLIERS from original dataset
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

  
# update 'time_series_tsne'
time_series_tsne = pd.DataFrame()  
time_series_tsne = dataset_no_outliers_01.copy()


# count frequency of 6-digits Ateco codes.    
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


# reset index
df_4_0_01.reset_index(drop=True, inplace=True) 


# convert DataFrame to input (not Numpy surprisingly) to feed into t-SNE
time_series_tsne = df_4_0_01.copy()  


# print summary statistics of the ateco groups used
print(criteria_01)


# save 'criteria_01' as .csv   
criteria_01.to_csv(os.path.sep.join([BASE_DIR, 
                                     'criteria_01.csv']), index= False)    
    
  
    
###############################################################################
# 2.10. MultiDimensional Scaling (MDS) 


# when no outlier cleaining is wanted.
#criteria_01 = criteria.copy()    


# re-run PCA on the new dataset
Y_pca =[]
#Y_pca = pca.fit_transform(time_series_tsne.loc[:, ['2014','2015','2016','2017']])
Y_pca = kernel_pca.fit_transform(time_series_tsne.loc[:, ['2014','2015','2016','2017']])

     
    
# re-run ICA on the new dataset    
# fit model
Y_ica =[]
Y_ica = fast_ICA.fit_transform(Y_pca)    
#Y = fast_ICA.fit_transform(time_series_tsne.loc[:, ['2014','2015','2016','2017']])    
    
    
# import libraries 
from sklearn.manifold import MDS  
from multiprocessing import cpu_count

# initialize MDS parameters: 
n_components = 2
metric = True
#n_init = 2
n_init = 64
#max_iter = 30
max_iter = 3000000
verbose = 1
eps = 0.001
n_jobs = int(round(((cpu_count()/4)-10),0))
#n_jobs = -4 # valid for MDS only. For Isolation Forest has been set to 1.
random_state = None # valid both for MDS and Isolation Forest
dissimilarity = 'euclidean'


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
Y_mds = mds.fit_transform(Y_ica)


# save Y as .csv 
Y_01 = pd.DataFrame(Y_mds)  
Y_01.to_csv(os.path.sep.join([BASE_DIR, 
                                     'Y_criteria_01.csv']), index= False)
    
# MDS stress metric
mds_stress = round(mds.stress_, 2)
print(mds_stress)

    
## PLOT FIRST ROUND RESULT 
time_series_tsne['MDS-2d-one'] = Y_mds[:,0] 
time_series_tsne['MDS-2d-two'] = Y_mds[:,1]


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


# save cleaned DataFrame as .csv file
time_series_tsne.to_csv(os.path.sep.join([BASE_DIR, 
                                     'data_criteria_01.csv']), index= False)




















































#%reset -f

