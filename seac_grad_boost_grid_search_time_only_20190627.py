"""
Analysese of the following data-set: 
- 23_seac_hr_ateco
- 6_seac_47_ateco
P.S.1.: please type in "analysis" at the command line. 
(e.g.'bootstrapping', 'grid_search'). E.g.: 
"python3 keras_multiple_gpus.py --analysis 'bootstrapping'"
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
   BASE_DIR = ('/home/paolo/Dropbox/analyses/python/tensorflow/seac_data/23_seac_hr_ateco')

else:
   BASE_DIR = '/home/Lan/paolo_scripts/exp_seasonality/seac_data_1'

BASE_REP = BASE_DIR




# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--parallel",  dest='parallel', action='store_true',
	help="# enable multi-core computation")
ap.add_argument("-no-p", "--no-parallel",  dest='parallel', action='store_false',
	help="# disable multi-core computation")
ap.add_argument("-a", "--analysis", type=str, default='bootstrapping',
	help="# type analysis's name") 
args = vars(ap.parse_args())


# grab the analysis you want to run. You have to write down the analysis name
# in the command line as done for all 'argparse' arguments.
analysis = args["analysis"] 
parallel = args["parallel"]
#analysis = 'bootstrapping'
#parallel = True


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





## 3. LOAD 
# load cleaned dataset    
df_2 = pd.read_csv(os.path.sep.join([BASE_DIR, 
                                     'data_criteria_02.csv']))









###############################################################################
## PREPARE DATA


# summary statistics
#df_2.info()
#summary_statistics = df_2.describe()

"""
# one-hot encoding  
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(categories='auto', sparse = False)
correct_ateco_1hot = encoder.fit_transform(df_2.loc[:, ['correct_ateco']])
"""

# pop out the MDS scaling 2-d columns. Please compare X_tsne_1 vs X_tsne_2 
# and see which one fits better the plot of decsion boundary
X_tsne_1 = df_2.loc[:, ['tsne-2d-one', 'tsne-2d-two']].copy()
correct_ateco = df_2.loc[:, ['correct_ateco']].copy()


# drop useless column
df_2 = df_2.drop(columns =['CodDitta', 'first two ateco', 'tsne-2d-one', 
                           'tsne-2d-two'])
    

# create arrays to be fed to sklearn 
X = X_tsne_1.to_numpy()


# It seems Gradient Boosting is NOT able to handle One-Hot Encoding    
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#y_labels = le.fit_transform(y_labels_1[0])
y = le.fit_transform(df_2.loc[:, ['correct_ateco']]).copy()






##############################################################################
## GRADIENT BOOSTING WITH BOOTSTRAPPING (when the range of optimal 
## parameters has been already found)
if analysis == 'bootstrapping':  

    from multiprocessing import cpu_count
    from joblib import Parallel
    from joblib import delayed
    import timeit
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import f1_score, matthews_corrcoef, log_loss 
    from sklearn.metrics import make_scorer
    #from sklearn.model_selection import  GroupShuffleSplit
    #from sklearn.model_selection import LeavePGroupsOut
    #from sklearn.ensemble import BaggingRegressor
    from sklearn.ensemble import GradientBoostingClassifier
    
    
    # configure bootstrap
    n_iterations = 30
    #n_size = int(len(data) * 0.50)

    
    
    # set grid search's parameters
    model = GradientBoostingClassifier()
    param_grid_1 = {'loss' : ['deviance'],
                  #'loss' : ['deviance', 'exponential'], # NOT WORKING
                  #'learning_rate' : np.linspace(0.1, 1, 10, dtype = float),
                  'learning_rate' : [0.2],                  
                  'n_estimators' : [50000],
                  #'n_estimators' : [50],
                  'subsample' : [0.9],
                  #'subsample' : np.linspace(0.1, 1, 10, dtype = float),
                  'criterion' : ['friedman_mse'],
                  #'min_samples_split' : np.linspace(10, 30, 5, dtype = int),
                  'min_samples_split' : [9], # insignificant difference BTW...?!?
                  #'min_samples_leaf' : [1, 2, 3, 4],
                  'min_samples_leaf' : [3],
                  #'min_weight_fraction_leaf' : np.linspace(0.1, 0.4, 4, dtype = float),
                  'min_weight_fraction_leaf' : [0.1], 
                  #'max_depth' : np.linspace(1, 110, 20, dtype = int),
                  'max_depth' : [29], # insignificant difference BTW...?!?
                  #'min_impurity_decrease' : np.linspace(0.1, 0.6, 6, dtype = float),
                  'min_impurity_decrease' : [0.6], 
                  'min_impurity_split' : [None],
                  'init' : [None],
                  'random_state' : [None],
                  'max_features' : ['log2'],
                  #'max_features' : ['auto', 'log2' ],
                  'verbose' : [0],
                  'max_leaf_nodes' : [None],
                  'warm_start' : [True],
                  #'warm_start' : [False, True],
                  #'presort ' : ['auto'],
                  #'presort ' : ['auto', True],
                  'validation_fraction' : [0.1],
                  'n_iter_no_change' : [None],
                  'tol' : [0.0001],
                  }
    
    # convert dictionary to DataFrame 
    param_grid = param_grid_1
    
    
    # cross-validation parameters function
    def cross_validation(X, y):
    
        # set cross-validation    
        #cv = StratifiedKFold(n_splits = 3, shuffle = True,                         
        #                            random_state = None)
        cv = StratifiedShuffleSplit(n_splits = 3, test_size=0.2, 
                                    random_state = None)
        #cv = GroupShuffleSplit(n_splits=3, test_size=labels_str.size, 
        #                       #groups =labels_str.size,
        #                      random_state= None)
        #cv = GroupShuffleSplit(n_splits=3, test_size=0.2, 
        #                            random_state= None)
        
        # 'iid': ['False', 'True'] # useless?
        
        # scoring methods
        scoring_LOSS = make_scorer(log_loss, greater_is_better=False,
                                             needs_proba=True,
                                             labels=sorted(np.unique(y)))

        scoring = {'f1_macro': 'f1_macro', 
                   'f1_micro': 'f1_micro', 
                   'log_loss': scoring_LOSS,
                   'matthews_corrcoef': make_scorer(matthews_corrcoef)}
        
        # NOT WORKING: in order to avoid the warning the following could be implemented 
        # 
        #scoring = f1_score(y_test, y_pred, average='weighted', 
        #                 labels=np.unique(y_pred))
        
        
        ## train_test_split
        """
        # TEST: trying to use GroupShufflSplit
        labels_str = pd.unique(df_2.loc[:, 'correct_ateco']) 
        """
            
        # split data-set with 'stratify' option
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=None,
                shuffle = True, stratify = y)
        
        # split data-set without 'stratify' option
        #X_train, X_test, y_train, y_test = train_test_split(
        #        X, y, test_size=0.2, random_state=None, shuffle = True) 
                
        # Outputs
        return X_train, X_test, y_train, y_test, scoring, cv 
    
    
    



    # function for computing grid search    
    def grid_search(model, param_grid, X, y):    
        
        # grid search
        if RELEASE == '4.18.0-24-generic': # Linux laptop
            # setting cross-validation parameters
            X_train, X_test, y_train, y_test, scoring, cv = cross_validation(X, y)
            # single-core computation
            grid = GridSearchCV(estimator = model, param_grid = param_grid,
                         iid = False, cv = cv, scoring=scoring, refit='matthews_corrcoef')
        
        else:
            # setting cross-validation parameters
            X_train, X_test, y_train, y_test, scoring, cv = cross_validation(X, y)
            # use multi-cores available    
            grid = GridSearchCV(estimator = model, param_grid = param_grid,
                         iid = False, cv = cv,
                         n_jobs = 1, # put positive number of CPUs to be used. 
                         # Nevertheless it seems the server does not stick to such
                         # a number!!!
                         #n_jobs = int(round(((cpu_count()/4)-36),0)),
                         scoring=scoring, refit='matthews_corrcoef')
        
        # fit model
        grid_result = grid.fit(X_train, y_train)
            
        # OUTPUT grid_search function
        return (grid_result)
    
    
    

    # running either parallel or single-core computation. 
    if parallel:
        #timing  = """\
    	# execute configs in parallel
        executor = Parallel(n_jobs= 32, backend='loky')
        #executor = Parallel(n_jobs= int(round(((cpu_count()/4)-4),0)), 
        #                                backend='loky')
          
    
        tasks = (delayed(grid_search)(model, param_grid, X, 
                 y) for i in range(n_iterations))
        output = executor(tasks)
            #"""
            #time = timeit.timeit(timing)
                
    else:
        output = [grid_search(model, param_grid, X, 
                              y) for i in range(n_iterations)]        
 
    
    # append output from 'joblib' in lists and DataFrames    
    stats = []
    best_score = []
    best_parameters = []
    results = pd.DataFrame() 
    
    # collect and save all CV results for plotting
    for counter in range(0,len(output)):
        # collect cross-validation results (e.g. multiple metrics etc.)
        results_1 = pd.DataFrame.from_dict(output[counter].cv_results_)
        results = results.append(results_1, ignore_index=True) 
        # collect best_scores    
        best_score.append(round(output[counter].best_score_, 3))
        best_parameters.append(output[counter].best_params_)



    # print
    print('Number iterations: %.0f' % (len(output)))

    


##############################################################################
## GRADIENT BOOSTING GRID SEARCH (when the range of optimal 
## parameters has to be found)
if analysis == 'grid_search':  

    from multiprocessing import cpu_count
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import f1_score, matthews_corrcoef, log_loss 
    from sklearn.metrics import make_scorer
    #from sklearn.model_selection import  GroupShuffleSplit
    #from sklearn.model_selection import LeavePGroupsOut
    #from sklearn.ensemble import BaggingRegressor
    from sklearn.ensemble import GradientBoostingClassifier
    
    
    # configure bootstrap
    n_iterations = 30
    #n_size = int(len(data) * 0.50)
    stats = []
    best_score = []
    best_parameters = []
    results = pd.DataFrame()
    
    
    for i in range(n_iterations):
        
        """
        # TEST: trying to use GroupShufflSplit
        labels_str = pd.unique(df_2.loc[:, 'correct_ateco']) 
        """
        
        
        # split data-set with 'stratify' option
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=None, stratify = y)
        
        
        ## split data-set without 'stratify' option
        #X_train, X_test, y_train, y_test = train_test_split(
        #        X, y, test_size=0.2, random_state=None)    
    
        
        
        # set grid search's parameters
        model = GradientBoostingClassifier()
        param_grid = {'loss' : ['deviance'],
                      #'loss' : ['deviance', 'exponential'], # NOT WORKING
                      #'learning_rate' : np.linspace(0.1, 1, 10, dtype = float),
                      'learning_rate' : [0.2],                  
                      'n_estimators' : [5000],
                      #'n_estimators' : [50],
                      'subsample' : [0.9],
                      #'subsample' : np.linspace(0.1, 1, 10, dtype = float),
                      'criterion' : ['friedman_mse'],
                      #'min_samples_split' : np.linspace(2, 30, 20, dtype = int),
                      'min_samples_split' : [9], # insignificant difference BTW...?!?
                      #'min_samples_leaf' : [1, 2, 3, 4],
                      'min_samples_leaf' : [3],
                      #'min_weight_fraction_leaf' : np.linspace(0.1, 0.4, 4, dtype = float),
                      'min_weight_fraction_leaf' : [0.1], 
                      #'max_depth' : np.linspace(1, 50, 20, dtype = int),
                      'max_depth' : [29], # insignificant difference BTW...?!?
                      #'min_impurity_decrease' : np.linspace(0.1, 0.6, 6, dtype = float),
                      'min_impurity_decrease' : [0.6], 
                      'min_impurity_split' : [None],
                      'init' : [None],
                      'random_state' : [None],
                      'max_features' : ['log2'],
                      #'max_features' : ['auto', 'log2' ],
                      'verbose' : [0],
                      'max_leaf_nodes' : [None],
                      'warm_start' : [True],
                      #'warm_start' : [False, True],
                      #'presort ' : ['auto'],
                      #'presort ' : ['auto', True],
                      'validation_fraction' : [0.1],
                      'n_iter_no_change' : [None],
                      'tol' : [0.0001],
                      }
           
         
        # set cross-validation    
        #cv = StratifiedKFold(n_splits = 3, shuffle = True,                         
        #                            random_state = None)
        cv = StratifiedShuffleSplit(n_splits = 3, test_size=0.2, 
                                    random_state = None)
        #cv = GroupShuffleSplit(n_splits=3, test_size=labels_str.size, 
        #                       #groups =labels_str.size,
        #                      random_state= None)
        #cv = GroupShuffleSplit(n_splits=3, test_size=0.2, 
        #                            random_state= None)
        
        # 'iid': ['False', 'True']
        
        
        # scoring methods
        scoring_LOSS = make_scorer(log_loss, greater_is_better=False,
                                             needs_proba=True,
                                             labels=sorted(np.unique(y)))

        scoring = {'f1_macro': 'f1_macro', 
                   'f1_micro': 'f1_micro', 
                   'log_loss': scoring_LOSS,
                   'matthews_corrcoef': make_scorer(matthews_corrcoef)}
        
        # NOT WORKING: in order to avoid the warning the following could be implemented 
        # 
        #scoring = f1_score(y_test, y_pred, average='weighted', 
        #                 labels=np.unique(y_pred))
        
        # grid search
        if RELEASE == '4.18.0-24-generic': # Linux laptop
            # single-core computation
        	grid = GridSearchCV(estimator = model, param_grid = param_grid,
                         iid = False, cv = cv, scoring=scoring, refit='matthews_corrcoef')
        
        else:
            # use multi-cores available    
            grid = GridSearchCV(estimator = model, param_grid = param_grid,
                         iid = False, cv = cv,
                         n_jobs = 31, # put positive number of CPUs to be used. 
                         # Nevertheless it seems the server does not stick to such
                         # a number!!!
                         #n_jobs = int(round(((cpu_count()/4)-36),0)),
                         scoring=scoring, refit='matthews_corrcoef')
        
        # fit model
        grid_result = grid.fit(X_train, y_train) 
    
    
        # collect and save all CV results for plotting
        results_1 = pd.DataFrame.from_dict(grid_result.cv_results_)
        results = results.append(results_1, ignore_index=True) 

        # collect best_scores    
        best_score.append(round(grid_result.best_score_, 3))
        best_parameters.append(grid_result.best_params_)
    
    
    
        # print
        print(grid_result.best_score_)
        print(grid_result.best_params_)
        print('Number iterations %.0f' % (i))


  
    
###############################################################################
## GENERATE OUTPUT (.csv files and .jpg pictures)
  


   
#feature_importances=grid_search.best_estimator_.feature_importances_
#cvres=grid_search.cv_results_   
#for mean_score,params in zip(cvres["mean_test_score"],cvres["params"]):    
#    print(np.sqrt(-mean_score),params)
    
    
 
    
###############################################################################
## SAVE BEST SCORES IN A PANDAS DATAFRAME AND PLOT THEIR BOOTSTRAPPING 
## DISTRIBUTION 

# save cleaned DataFrame as .csv file
results.to_csv(os.path.sep.join([BASE_DIR, 
                                     'cv_results.csv']), index= False) 

# concatenate results
best_score_df = pd.DataFrame(best_score, columns=['best_score'])
best_parameters_df = pd.DataFrame(best_parameters)
summary_table = pd.concat([best_score_df, best_parameters_df], axis = 1)

# save cleaned DataFrame as .csv file
summary_table.to_csv(os.path.sep.join([BASE_DIR, 
                                     'best_scores.csv']), index= False)

    
# in case you want to load the .csv with the best scores
#summary_table = pd.read_csv(os.path.sep.join([BASE_DIR, 
#                                     'best_scores.csv']))    

    
# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(summary_table.loc[:,'best_score'], p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(summary_table.loc[:,'best_score'], p))
median = np.median(summary_table.loc[:,'best_score'])
median_parameters = summary_table.loc[summary_table['best_score'] == (round(median, 2))]
print('%.1f confidence interval %.4f and %.4f' % (alpha*100, lower, 
                                                      upper))
print('Median %.4f' % (median))
print('Below best score (median) and parameters ')
print(median_parameters)
    

# plot scores and save plot
date = str(datetime.datetime.now())
sns_plot = sns.distplot(best_score_df, bins = 30)
#sns_plot = sns.distplot(best_score_df, bins = (len(best_score_df)/100))
fig = sns_plot.get_figure()
fig.savefig(os.path.sep.join([BASE_DIR, date[0:10]+ "_" + date[11:16]+".jpg"])) 
  
    







"""
###############################################################################
## PLOT MULTIPLE SCORES RESULTS
results = grid_result.cv_results_

plt.figure(figsize=(13, 13))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
          fontsize=16)

plt.xlabel("min_samples_split")
plt.ylabel("Score")

ax = plt.gca()
ax.set_xlim(0, 402)
ax.set_ylim(0.73, 1)

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_min_samples_split'].data, dtype=float)

for scorer, color in zip(sorted(scoring), ['g', 'k']):
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid(False)
plt.show()
"""

   

 
    











#%reset -f

