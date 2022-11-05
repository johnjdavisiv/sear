# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 01:06:44 2021

@author: John
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Custom functions
from sear_helpers import *
from develop_model_template import *

def crossval_sear_model(file_path, model_name, k_fold, n_search,
                        include_MIMS, include_anthro,
                        only_anthro, 
                        feature_flag = ""):
    
    #Error checking to make sure anthro only null models are fit right
    if model_name == "nullanthro":
        include_anthro = True
        only_anthro = True
        include_MIMS = False
        
    #Read in files, exclude outliers, choose to use anthro
    #maybe later choose to use slow features...
    X_features, y_speed, sub_col, feature_names = read_features(file_path, 
                                                                include_MIMS = include_MIMS,
                                                                include_anthro = include_anthro,
                                                                only_anthro = only_anthro)
    
    #sub_col is a pd.Series() object with the sub_code data
    #use it to build LOSO indices
    all_subs = sub_col.unique()
    
    results_dict = {"model_name":[],
                    "window_len":[],
                    "include_MIMS":[],
                    "include_anthro":[],
                    "test_subject":[],
                    "test_mae":[],
                    "test_rmse":[],
                    "test_mape":[]
                    }
    
    predict_vs_actual = {"test_subject":[],
                         "y_test":[],
                         "y_predict":[]
                         }
    
    #Danger! This is brittle!    
    w_len = int(file_path.split("_")[3][:2])
        
    full_start = time.time() #tic;
    
    # --------   For each subject -----------
    for i in range(len(all_subs)):
        #Leave out this subject by "popping off" their data
        print("**************************************************************")
        (X_loso, y_loso), (X_test, y_test), loso_sub_col = pop_subject(X_features, y_speed, 
                                                         sub_col, subject = all_subs[i])
        print("**************************************************************")
        
        #Develop a FULL learning algorithm on data from all except this subject
        model_i = develop_model(model_name, X_loso, y_loso, loso_sub_col,
                                k_fold = 8, n_search = n_search,
                                in_parallel = True)
        
        #Predict on left-out subject    
        (this_mae, this_rmse, this_mape) = evaluate_model(model_i, X_test, y_test)
        
        #Save results
        results_dict["model_name"].append(model_name)
        results_dict["window_len"].append(w_len)
        results_dict["include_MIMS"].append(include_MIMS)
        results_dict["include_anthro"].append(include_anthro)
        
        results_dict["test_subject"].append(all_subs[i])
        results_dict["test_mae"].append(this_mae)
        results_dict["test_rmse"].append(this_rmse)
        results_dict["test_mape"].append(this_mape)
        
        #Also save raw data for plotting and diagnostics
        predict_vs_actual["test_subject"].append(all_subs[i])
        predict_vs_actual["y_test"].append(y_test)
        predict_vs_actual["y_predict"].append(model_i.predict(X_test))
    #end!    
    
    #Save results to csv
    save_results(results_dict, predict_vs_actual, 
                 model_name, file_path,
                 include_MIMS,
                 include_anthro,
                 n_search,
                 feature_flag = feature_flag)
    
    full_end = time.time() #toc;
    
    print("**************************************************************")
    print("**************************************************************")
    print("Full cross-validation time elapsed: {} min".format((full_end - full_start)/60))
    print("**************************************************************")
    print("**************************************************************")
    print(" ")
    
    #Plot the results, check out MAE and MAPE
    print_and_plot_results(results_dict, predict_vs_actual, 
                           model_name, include_MIMS, 
                           include_anthro,
                           file_path)
