# -*- coding: utf-8 -*-
"""

Templates for SEAR ML models

"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Sklearn tools
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GroupKFold

#Import models
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor

#Import scipy distributions (for random search)
from scipy.stats import loguniform, uniform


def develop_model(model_name, *args, **kwargs):
    if model_name == "null":
        return develop_null_model(*args, **kwargs)
    elif model_name == "linear":
        return develop_linear_model(*args, **kwargs)
    elif model_name == "nullanthro":
        return develop_linear_model(*args, **kwargs)
    elif model_name == "ridge":
        return develop_ridge_model(*args, **kwargs)
    elif model_name == "xgb":
        return develop_xgb_model(*args, **kwargs)
    else:
        raise TypeError("Hold up I havent implemented that model yet!")

#Fit a null model and return it
def develop_null_model(X_loso, y_loso, loso_sub_col, 
                       k_fold = 8, n_search = 1,
                       in_parallel = True):
    #Mostly for consistency with other model functions
    sub_ints = loso_sub_col.astype('category').cat.codes.values
    
    #Create crossvalidator
    group_cv = GroupKFold(n_splits = k_fold)
    
    
    #Need a dummy preprocessor
    from sklearn.base import BaseEstimator, TransformerMixin

    class NullFeatures(BaseEstimator, TransformerMixin):    
        def __init__(self):
            pass

        def fit(self, X, y=None):
            # Do nothing
            return self
        
        def transform(self, X):
            return np.ones((X.shape[0],1))
    
    #Kinda hacky solution to wrap
    null_model = LinearRegression()
    null_features = NullFeatures()
    
    #For consistency
    null_pipe = Pipeline(steps = [('null_features', null_features),
                                  ('null_regression', null_model)])
    null_grid = {'null_regression__fit_intercept': [False]}
    
    null_search = RandomizedSearchCV(null_pipe,
                                     param_distributions=null_grid,
                                     scoring = 'neg_mean_absolute_error',
                                     n_jobs = 1,
                                     cv = group_cv,
                                     verbose = 1,
                                     random_state = 1989,
                                     n_iter = 1,
                                     return_train_score = False,
                                     refit = True)
    print("Fitting null model...")
    #"search" for best params with k-fold
    null_search.fit(X_loso, y_loso, groups = sub_ints)
    print("Null model fit!")
    
    return null_search

#X_loso: design matrix with features (unscaled!)
#y_loso: outcome (speed in m/s)
#loso_sub_col: pd.Series of all subjects (except the loso'd one)
#   this has same number of rows as X_loso 

#should return a final model called model_i which has a .predict() method
#that takes as input X and y and outputs y_predict

# ------- Plain old fashioned linear regression ------------

def develop_linear_model(X_loso, y_loso, loso_sub_col, 
                         k_fold = 8, n_search = 1,
                         in_parallel = True):
    #Subject groupsings for cross-validator
    sub_ints = loso_sub_col.astype('category').cat.codes.values
    group_cv = GroupKFold(n_splits = k_fold)
    
    model = LinearRegression()
    scale = StandardScaler()
    
    pipe = Pipeline(steps = [('scale', scale),
                         ('regression', model)])

    param_grid = {'regression__fit_intercept': [True]}
    
    rand_search = RandomizedSearchCV(pipe,
                                     param_distributions=param_grid,
                                     scoring = 'neg_mean_absolute_error',
                                     n_jobs = 1,
                                     cv = group_cv,
                                     verbose = 1,
                                     random_state = 1989,
                                     n_iter = 1,
                                     return_train_score = False,
                                     refit = True)
    #Fit and time
    start = time.time() #tic;
    print("Fitting linear model...")
    #"search" for best params with k-fold (lol)
    rand_search.fit(X_loso, y_loso, groups = sub_ints)
    end = time.time() #toc;
    print("Linear model fit completed! Time elapsed: {} sec".format(end - start))
    
    return rand_search


# ------- Ridge regression  ------------
def develop_ridge_model(X_loso, y_loso, loso_sub_col, 
                        k_fold = 8, n_search = 500,
                        in_parallel = True):
    #For parallelism-  care, hard to abort
    if in_parallel:
        n_jobs = -1
    else:
        n_jobs = 1
    
    #Cross-validator
    sub_ints = loso_sub_col.astype('category').cat.codes.values
    group_cv = GroupKFold(n_splits = k_fold)
    
    model = Ridge(fit_intercept = True)
    scale = StandardScaler()
    
    pipe = Pipeline(steps = [('scale', scale),
                         ('regression', model)])
    
    param_grid = {'regression__alpha': loguniform(10e-5, 10e8)}
    
    rand_search = RandomizedSearchCV(pipe,
                                     param_distributions=param_grid,
                                     scoring = 'neg_mean_absolute_error',
                                     n_jobs = n_jobs,
                                     cv = group_cv,
                                     verbose = 1,
                                     random_state = 1989,
                                     n_iter = n_search,
                                     return_train_score = False,
                                     refit = True)
    start = time.time() #tic;
    print("Fitting ridge model...")
    #"search" for best params with k-fold
    rand_search.fit(X_loso, y_loso, groups = sub_ints)
    end = time.time() #toc;
    print("Ridge fit completed! Time elapsed: {} sec".format(end - start))
    
    return rand_search


def develop_xgb_model(X_loso, y_loso, loso_sub_col,
                      k_fold = 8, n_search = 100,
                      in_parallel = True):

    #For parallelism-  care, hard to abort
    # Hmmm sklearn is geting memory loeaks...
    if in_parallel:
        n_threads = 1 #use Sklearn parallelism not XGB
        n_jobs = -1 #Original was -2 to not crash --> try diff strat?
    else:
        n_threads = 1
        n_jobs = 1
                
    #Subject groupsings for cross-validator
    sub_ints = loso_sub_col.astype('category').cat.codes.values
    group_cv = GroupKFold(n_splits = k_fold)
    
    #Can add if GPU
    #tree_method = 'gpu_hist' for GPU usage
    setup_params = {'objective':'reg:squarederror',
                    'n_jobs':n_threads}

    model = XGBRegressor(**setup_params)
    scale = StandardScaler()
    pipe = Pipeline(steps = [('scale', scale),
                         ('regression', model)])
    
    #Not sure if needs improvement?
    param_grid = {"regression__n_estimators":np.logspace(1,3, 20).astype(int),
                  "regression__learning_rate":loguniform(1e-3, 3e-1),
                  "regression__max_depth":np.arange(1,12,2),
                  "regression__min_child_weight":[1, 10, 100],
                  "regression__gamma":[0,0.01,0.05],
                  "regression__subsample":[0.5, 0.75, 1.0],
                  "regression__colsample_bytree":[0.4, 0.6, 0.8, 1.0]}
    
    rand_search = RandomizedSearchCV(pipe,
                                     param_distributions=param_grid,
                                     scoring = 'neg_mean_absolute_error',
                                     cv = group_cv,
                                     verbose = 10,
                                     n_jobs = n_jobs,
                                     random_state = 1989,
                                     n_iter = n_search,
                                     return_train_score = False,
                                     refit = True)
    
    start = time.time() #tic;
    print("Fitting XGBoost model...")
    #"search" for best params with k-fold
    rand_search.fit(X_loso, y_loso, groups = sub_ints)
    end = time.time() #toc;
    print("XGBoost fit completed! Time elapsed: {} sec".format(end - start))
    
    return rand_search

