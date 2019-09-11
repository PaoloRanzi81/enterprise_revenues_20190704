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


# reduce the number of threads used by each CPU
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
                                     'clean_data_5.csv']))
  

time_series_tsne = df_2.copy()

# count the frequency of 6-digits Ateco codes.    
z_six_digits_sub = defaultdict(int)
for x in time_series_tsne['correct_ateco']:
    z_six_digits_sub[x] += 1
    
# converting dictionary to DataFrame
z_six_digits_criteria = pd.DataFrame.from_dict(z_six_digits_sub, orient='index',
                           columns =['Frequency'])
z_six_digits_criteria['correct_ateco'] = z_six_digits_criteria.index
z_six_digits_criteria.reset_index(drop=True, inplace=True) 

# build 'criteria'  
criteria = z_six_digits_criteria.copy()


###############################################################################
# 2.5. t-SNE GRID SEARCH
# import libraries for parallel computing +TSNE
from sklearn import manifold    
from joblib import Parallel
from joblib import delayed
from multiprocessing import cpu_count
import timeit

# TSNE parameters

param_grid = {'n_iter': 100000, 
              'method' : 'exact',
              'n_components' : 2,
              'init' : 'random',
              'verbose' : 0}

#parallel = False
#perplexities= [77,80]
#perplexity= 77
perplexities = np.linspace(5, 80, 30, dtype = int)
#learning_rates = [350, 450]
#learning_rate = 350
learning_rates = np.linspace(10, 1000, 50, dtype = int)

#init = 'random'
#init = 'pca'
#method = 'barnes_hut'
#method = 'exact'
#early_exag_coeff=12
# metric = metrics.pairwise.euclidean_distances


# function for computing TSNE
def TSNE_computation(param_grid, time_series_tsne, learning_rate, perplexity):
    tsne = manifold.TSNE(n_components = param_grid['n_components'], 
                         perplexity = perplexity, 
                         method = param_grid['method'], 
                         verbose = param_grid['verbose'], 
                         n_iter = param_grid['n_iter'],
                         init = param_grid['init'], 
                         learning_rate = learning_rate, 
                         random_state = None)
       
    # appropriate for Numpy array
    Y = tsne.fit_transform(time_series_tsne.loc[:,['2014','2015','2016',
                                               '2017']])
    # round KL divergence
    rounded_kl_divergence = round(tsne.kl_divergence_, 12)
    # generate a DataFrame where each row stores the output
    output_df =pd.DataFrame({'KL': rounded_kl_divergence,
                             'perplexity': [perplexity], 
                             'learning_rate': [learning_rate]}) 
    # AFTER GRID SEARCH
    #return (Y) 

    # DURING GRID SEARCH
    return (output_df) 



#initilize DataFrame
output_total = pd.DataFrame()
# first 'for loop' (perplexity)
for perplexity in perplexities:
# running either parallel or single-core computation. P.S.: GridsearchCV 
# for TSNE seems NOT to work.  
    if parallel:
            
            #timing  = """\
            
    		# execute configs in parallel
    		executor = Parallel(n_jobs= int(round(((cpu_count()/4)-4),0)), 
                                            backend='loky')
            # second 'for loop' (learning_rate)    
    		tasks = (delayed(TSNE_computation)(param_grid, time_series_tsne,
    		learning_rate, perplexity) for learning_rate in learning_rates)
    		output = executor(tasks)
            #"""
            #time = timeit.timeit(timing)
            
    else:
    		output = [TSNE_computation(param_grid, time_series_tsne, learning_rate, 
                                 perplexity) for learning_rate in learning_rates]
       
    # append output
    #print(time)
    print(output)
    output_total= output_total.append(output, ignore_index=True) 

    
# save DataFrame as a .csv file
output_total.to_csv(os.path.sep.join([BASE_DIR, 
                                     'output_tsne.csv']), index= False)


#load output_total
#output_total = pd.read_csv(os.path.sep.join([BASE_DIR, 
#                                     'output_tsne.csv']))   

    


###############################################################################
# 8. t-SNE BOOTSTRAPPING (when the range of optimal parameters has been 
# already found)

# import libraries for parallel computing +TSNE
from sklearn import manifold    
from joblib import Parallel
from joblib import delayed
from multiprocessing import cpu_count

# TSNE parameters

param_grid = {'n_iter': 1000000, 
              'method' : 'exact',
              'n_components' : 2,
              'init' : 'random',
              'verbose' : 0}

#perplexities = np.linspace(20, 50, 30, dtype = int) # optimal parameter
#learning_rates = [10, 36, 37] # optimal parameter
#method= 'barnes_hut'

perplexity = 100
learning_rate = 5000

# function for computing TSNE
def TSNE_computation(param_grid, time_series_tsne, learning_rate, perplexity):
    tsne = manifold.TSNE(n_components = param_grid['n_components'], 
                         perplexity = perplexity, 
                         method = param_grid['method'], 
                         verbose = param_grid['verbose'], 
                         n_iter = param_grid['n_iter'],
                         init = param_grid['init'], 
                         learning_rate = learning_rate, 
                         random_state = None)
       
    # appropriate for Numpy array
    Y = tsne.fit_transform(time_series_tsne.loc[:,['2014','2015','2016',
                                               '2017']])
    # appropriate for Pandas DataFrame (please further adapt the columns)
    #Y = tsne.fit_transform(time_series_tsne.loc[:,['revenue_median']])  
    #Y = tsne.fit_transform(time_series_tsne.loc[:,['revenue_median', 'Anno']])  
    
    # round KL divergence
    rounded_kl_divergence = round(tsne.kl_divergence_, 12)
    print(rounded_kl_divergence)
    # generate a DataFrame where each row stores the output
    output_df =pd.DataFrame({'KL': rounded_kl_divergence,
                             'perplexity': [perplexity], 
                             'learning_rate': [learning_rate]}) 

    # AFTER GRID SEARCH
    return (Y) 
    
    # DURING GRID SEARCH/BOOTSTRAPPING
    # return (output_df) 


# disable multi-core computation: 
parallel = False


#initilize DataFrame
#output_total = pd.DataFrame()
# first 'for loop' (perplexity)
for perplexity in perplexities:

# running either parallel or single-core computation. P.S.: GridsearchCV 
# for TSNE seems 
# NOT to work.  
    if parallel:
    		# execute configs in parallel
    		executor = Parallel(n_jobs= int(round(((cpu_count()/4)-2),0)),
                          backend='multiprocessing')
    		#executor = Parallel(n_jobs= -2, backend='loky')
       # second 'for loop' (learning_rate)    
    		tasks = (delayed(TSNE_computation)(param_grid, time_series_tsne,
    		learning_rate, perplexity) for learning_rate in learning_rates)
    		output = executor(tasks)
    else:
    		output = [TSNE_computation(param_grid, time_series_tsne, learning_rate, 
                                 perplexity) for learning_rate in learning_rates]
       
    # append output
    #print(output)
    #output_total= output_total.append(output, ignore_index=True)



    # 2-Dimensional/2 t-SNE components plot
    
    time_series_tsne['tsne-2d-one'] = Y[:,0]
    time_series_tsne['tsne-2d-two'] = Y[:,1]
    
    tsne_plot = plt.figure(figsize=(16,10))
    tsne_plot = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        #hue="Anno",
        hue="correct_ateco",
        palette=sns.color_palette("hls", criteria.shape[0]),
        #size = "2017",
        data=time_series_tsne,
        legend="full",
        alpha=0.9 
    )


    # add title                   
    tsne_plot = plt.title('perplexity = %.0f\n\
                          learning_rate = %.0f\n KL = %.8f' % \
                          (perplexity, learning_rate, 
                           rounded_kl_divergence)) 
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
         tsne_plot.text(time_series_tsne.loc[line,['tsne-2d-one']]+0.0001,
                        time_series_tsne.loc[line,['tsne-2d-two']]+0.0001, 
                        time_series_tsne.loc[line]['correct_ateco'], 
                        horizontalalignment='left', size='medium') 

"""

     
    

    
    
    
    
    
"""
    # 3-Dimensional/3 t-SNE components plot
    time_series_tsne['tsne-3d-one'] = Y[:,0]
    time_series_tsne['tsne-3d-two'] = Y[:,1]
    time_series_tsne['tsne-3d-three'] = Y[:,2]
    
    
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(
        xs=time_series_tsne['tsne-3d-one'], 
        ys=time_series_tsne['tsne-3d-two'], 
        zs=time_series_tsne['tsne-3d-three'], 
        c=time_series_tsne.loc[:,'first two ateco'] , 
        cmap='tab10'
    )
    ax.set_xlabel('tsne-3d-one')
    ax.set_ylabel('tsne-3d-two')
    ax.set_zlabel('tsne-3d-three')
    plt.title('perplexity = %.0f\n KL = %.4f' % (perplexity, rounded_kl_divergence)) 


    #date = str(datetime.datetime.now())
    #absolute_path = os.path.sep.join([BASE_REP, date[0:10]+ "_" + date[11:16]+".txt"])
    #plt.savefig(os.path.sep.join([BASE_REP, date[0:10]+ "_" + date[11:16]+".jpg"]))
    #plt.clf()
    plt.show()
"""























#%reset -f

