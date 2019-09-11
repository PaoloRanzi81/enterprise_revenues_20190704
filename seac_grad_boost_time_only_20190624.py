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
   BASE_DIR = '/home/paolo/Dropbox/analyses/python/tensorflow/seac_data/149_seac_hr_ateco'

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





## 3. LOAD 
# load cleaned dataset    
df_2 = pd.read_csv(os.path.sep.join([BASE_DIR, 
                                     'data_criteria_01.csv']))


    



###############################################################################
## PREPARE DATA

# summary statistics
#df_2.info()
#summary_statistics = df_2.describe()

"""
# one-hot encoding [NOT WORKING with gradient boosting]   
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(categories='auto', sparse = False)
correct_ateco_1hot = encoder.fit_transform(df_2.loc[:, ['correct_ateco']])
"""

# pop out the MDS scaling 2-d columns. 
X_tsne_1 = df_2.loc[:, ['MDS-2d-one', 'MDS-2d-two']].copy()


# create arrays to be fed to sklearn 
X = X_tsne_1.to_numpy()


# It seems Gradient Boosting is NOT able to handle One-Hot Encoding    
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df_2.loc[:, ['correct_ateco']]).copy()




"""
###############################################################################
## 2.6. Kernel Principal Component Analysis (PCA) 
 
# import libraries 
from sklearn.decomposition import KernelPCA
#from sklearn.decomposition import PCA
from multiprocessing import cpu_count


# initialize PCA parameters: 
n_components = 2
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
Y_pca = kernel_pca.fit_transform(X)

    
    
    
###############################################################################
## 2.7. Independent Component Analysis (ICA) 


## FIRST ROUND CLEANING: i) ICA  
# import libraries 
#from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
#from sklearn.manifold import LocallyLinearEmbedding
from multiprocessing import cpu_count


# initialize FastICA parameters: 
n_components = 2
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
X = []
X = fast_ICA.fit_transform(Y_pca)
"""






##############################################################################
## GRADIENT BOOSTING (when the range of optimal 
## parameters has been already found by grid search)
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer
#from sklearn.model_selection import  GroupShuffleSplit
#from sklearn.model_selection import LeavePGroupsOut
#from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingClassifier


# split data-set
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=None, shuffle = True, stratify = y)


# set grid search's parameters
param_grid_1 = {'loss' : ['deviance'],
              'learning_rate' : [0.2],                  
              'n_estimators' : [50000],
              'subsample' : [0.9],
              'criterion' : ['friedman_mse'],
              'min_samples_split' : [9],
              'min_samples_leaf' : [3],
              'min_weight_fraction_leaf' : [0.1], 
              'max_depth' : [29],
              'min_impurity_decrease' : [0.6], 
              'min_impurity_split' : [None],
              'init' : [None],
              'random_state' : [None],
              'max_features' : ['log2'],
              'verbose' : [0],
              'max_leaf_nodes' : [None],
              'warm_start' : [True],
              #'presort ' : ['auto'],
              #'presort ' : ['auto', True],
              'validation_fraction' : [0.1],
              'n_iter_no_change' : [None],
              'tol' : [0.0001],
              }


# convert dictionary to DataFrame   
param_grid = pd.DataFrame(param_grid_1)


# initialize classifier 
rf = GradientBoostingClassifier(loss =  param_grid.loc[:,'loss'][0],     
                            learning_rate =  param_grid.loc[:,'learning_rate'][0],
                            n_estimators = param_grid.loc[:,'n_estimators'][0],
                            subsample =  param_grid.loc[:,'subsample'][0],
                            criterion =  param_grid.loc[:,'criterion'][0],
                            min_samples_split =  param_grid.loc[:,'min_samples_split'][0],
                            min_samples_leaf =  param_grid.loc[:,'min_samples_leaf'][0],
                            min_weight_fraction_leaf =  param_grid.loc[:,'min_weight_fraction_leaf'][0],
                            max_depth =  param_grid.loc[:,'max_depth'][0],
                            min_impurity_decrease =  param_grid.loc[:,'min_impurity_decrease'][0],
                            min_impurity_split =  param_grid.loc[:,'min_impurity_split'][0],
                            init =  param_grid.loc[:,'init'][0],
                            random_state =  param_grid.loc[:,'random_state'][0],
                            max_features =  param_grid.loc[:,'max_features'][0],
                            verbose =  param_grid.loc[:,'verbose'][0],
                            max_leaf_nodes =  param_grid.loc[:,'max_leaf_nodes'][0],
                            warm_start =  param_grid.loc[:,'warm_start'][0],
                            #presort =  param_grid.loc[:,'presort'][0],
                            validation_fraction =  param_grid.loc[:,'validation_fraction'][0],
                            n_iter_no_change =  param_grid.loc[:,'n_iter_no_change'][0],
                            tol =  param_grid.loc[:,'tol'][0])

              
# fit model
rf.fit(X_train, y_train)


# check results: table
y_pred = rf.predict(X_test)
#pd.crosstab(y_test, y_pred, rownames=['actual'], 
#            colnames=['prediction'])


#pd.crosstab(y_test_labels.index, y_pred_labels.index, rownames=['actual'], 
#            colnames=['prediction'])


# inverse tranform from OneHot Encoding back to labels
y_train_labels = pd.DataFrame(le.inverse_transform(y_train))
y_test_labels = pd.DataFrame(le.inverse_transform(y_test))
y_pred_labels = pd.DataFrame(le.inverse_transform(y_pred))


# F1_score 'weighted'
scoring_weighted = f1_score(y_test_labels, y_pred_labels, average='weighted', 
                   labels=np.unique(y_pred_labels))
scoring_macro = f1_score(y_test_labels, y_pred_labels, average='macro')
scoring_micro = f1_score(y_test_labels, y_pred_labels, average='micro')
print(scoring_weighted)
print(scoring_macro)
print(scoring_micro)




# check results: feature_importances_
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

"""
# Print the feature_importances_ ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))


# Plot the feature_importances_ ranking
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]]);
"""



##############################################################################
## PLOTTING GRADIENT BOOSTNG'S DECISION BOUNDARIES (IT WORKS ONLY WITH 2-DIMENSIONS
## DATASETS). WARNING: using OneHotEncoding as y will fail to generate the 
## the mlxtend figure
from mlxtend.plotting import plot_decision_regions


# Eliminate the LabelEncoding used in the previous gradient boosting computation    
y_labels_1 = pd.DataFrame(le.inverse_transform(y))


# prepare Ateco code labels
labels_str = pd.unique(y_labels_1[0])
labels_str_df_1 = pd.DataFrame(labels_str, columns =['correct_ateco'], dtype=str)
labels_str_df = labels_str_df_1.sort_values(by= ['correct_ateco'], 
                                        ascending = True)


# create a list of strings (ateco code) to be put in the plot's legend
ateco_temp_df  =[]
for ateco_code in labels_str_df['correct_ateco']:
    #index = df_2[df_2['CodDitta'] == enterprise_id].index
    #ateco_code = df_2.iloc[index[0]]['correct_ateco']
    ateco_temp_df.append(str(ateco_code))



# Plotting decision regions
#ax = plot_decision_regions(X, y, clf = rf_plot, legend=0) # it does not show
# any difference?!?
ax = plot_decision_regions(X, y, clf = rf, legend=2)
# Adding axes annotations
plt.rcParams['figure.figsize'] = (26,10)
#plt.figure(figsize=(16,10)) # It does not work! 
#plt.figure(dpi=100) # It does not work!
plt.xlabel('MDS-2d-one')
plt.ylabel('MDS-2d-two')
plt.title('Gradient Boosting on 149_seac_hr_ateco, w/o PCA + w/o ICA')
# put the correct labels in the legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, 
          ateco_temp_df, 
           framealpha=0.3, scatterpoints=1)
ax.set_ylim([-0.2, 0.2])
ax.set_xlim([-0.2, 0.2])
# plot figure
plt.show()


# GENERATE AND SAVE PLOT 
date = str(datetime.datetime.now())
figure = plt.gcf() # get current figure
figure.set_size_inches(26,16)
plt.savefig(os.path.sep.join([BASE_REP, 
                          date[0:10]+ "_" + date[11:len(date)]+".jpg"]))













"""
from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)





plt.figure(figsize=(11,4))
plt.subplot(121)
plot_decision_boundary(tree_clf, X, y)
plt.title("Decision Tree", fontsize=14)
plt.subplot(122)
plot_decision_boundary(bag_clf, X, y)
plt.title("Decision Trees with Bagging", fontsize=14)
#save_fig("decision_tree_without_and_with_bagging_plot")
plt.show()
"""







   

 
    











#%reset -f

