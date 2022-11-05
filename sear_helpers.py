# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 15:30:09 2021

@author: John
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings

def read_features(file_path, include_MIMS = True,
                  include_anthro = False, 
                  only_anthro = False):
    """
    
    Read csv file with saved features for accelerometer data

    Parameters
    ----------
    file_path : string
        full path to the csv we are reading
    include_MIMS : bool, optional
        Should we include data from the Qu et al MIMSunit dataset? 
    include_anthro : bool, optional
        Should we include anthropometrics (height & weight)? If so, we cannot include the MIMS data :(
    only_anthro : bool, optional
        Should we ONLY return height and weight? Only used for null models

    Returns
    -------
    X_features : 2D ndarray
        Numpy array of features. y-outliers (speed) are excluded
    y_speed : 1D ndarray
        Numpy array of speeds. y-outliers excluded
    sub_col : pd.Series
        Pandas series of subject identifiers. Same dim as y_speed
    feature_names : list
        List of strings, with names of each feature. Same number of columns as X_features

    """
    
    if include_anthro and include_MIMS:
        warnings.warn("You requested anthropometric data and MIMSunit data, but MIMS has no anthro data. MIMS data excluded...") 
    
    df_raw = pd.read_csv(file_path)
    
    #Need to exlcude gross outliers - get middle 99% of data
    mid_99 = np.quantile(df_raw["speed"], [0.005, 0.995])
    mid_99[1] = 6.0 #Manually set high end
    excl_ix = (df_raw["speed"].values < mid_99[0]) | (df_raw["speed"].values > 6.0)
    df_excl = df_raw.loc[~excl_ix,:]
    
    print("Excluded {} datapoints ({}%) for being outside of range {}-{} m/s".format(
    np.sum(excl_ix),
    np.round(np.sum(excl_ix)/df_raw.shape[0]*100,2),
    np.round(mid_99[0],2), 
    np.round(mid_99[1],2)))
    
    #For null models
    if only_anthro:
        #return only height and weight (for a null model)
        excl_mims = (df_excl["study"] == "MIMSunit")
        df = df_excl.loc[~excl_mims,:]
        
        print("Excluded {} datapoints ({}%) because MIMS has no height/weight data".format(
            np.sum(excl_mims),
            np.round(np.sum(excl_mims)/df_excl.shape[0]*100,2)))
        
        X_features = df.loc[:, ["height", "weight"]].values
        feature_names = list(df.loc[:, ["height", "weight"]].columns.values)
    elif include_anthro:
        excl_mims = (df_excl["study"] == "MIMSunit")
        df = df_excl.loc[~excl_mims,:]
        
        print("Excluded {} datapoints ({}%) because MIMS has no height/weight data".format(
            np.sum(excl_mims),
            np.round(np.sum(excl_mims)/df_excl.shape[0]*100,2)))
        
        accel_ft_X = df.loc[:, "R_mean":].values
        anthro_X = df.loc[:, ["height", "weight"]].values
        X_features = np.column_stack((anthro_X, accel_ft_X))
        
        feature_names_anthro = list(df.loc[:, ["height", "weight"]].columns.values)
        feature_names_accel = list(df.loc[:, "R_mean":].columns.values)
        feature_names = feature_names_accel + feature_names_anthro
    elif include_MIMS:
        df = df_excl
        X_features = df.loc[:, "R_mean":].values
        feature_names = list(df.loc[:, "R_mean":].columns.values)
    else:
        excl_mims = (df_excl["study"] == "MIMSunit")
        df = df_excl.loc[~excl_mims,:]
        
        print("Excluded {} datapoints ({}%) because you asked not to include MIMS  data".format(
            np.sum(excl_mims),
            np.round(np.sum(excl_mims)/df_excl.shape[0]*100,2)))
        
        X_features = df.loc[:, "R_mean":].values
        feature_names = list(df.loc[:, "R_mean":].columns.values)

    y_speed = df["speed"].values
    sub_col = df["sub_code"]
    
    #Drop HARE2_S009 --> is in Cohort 3! Don't "double dip"
    print("DROPPED S009 from HARE2")
    exl_ix = sub_col == "HARE2_S009"
    X_features = X_features[~exl_ix]
    y_speed = y_speed[~exl_ix]
    sub_col = sub_col[~exl_ix]
    # ------------------------------------

    print("Dataset has " + str(y_speed.shape[0]) 
      + " datapoints and " + str(sub_col.unique().shape[0]) 
      + " unique subjects")

    return X_features, y_speed, sub_col, feature_names
        
    #Return design matrix
                

def get_sample_weights(loso_sub_col):
    """
    Get a column of inverse proportion weights
    """
    #Get inverse proportion column
    df = pd.DataFrame({"subject":np.unique(loso_sub_col, return_counts=True)[0],
                        "count":np.unique(loso_sub_col, return_counts=True)[1]})
    
    df["prop_total"] = df["count"]/loso_sub_col.shape[0]
    df["raw_inverse_prop"] = 1.0/df["prop_total"]
    weight_sum = df["raw_inverse_prop"].sum()
    #Normalize, maybe? Don't let loss get too small, will break float precision
    df["inverse_prop"] = df["raw_inverse_prop"]
    df.drop(["count", "prop_total", "raw_inverse_prop"], axis=1, inplace=True)
    
    #Now joint with loso sub col
    #Danger, inner rearranges rows to totally screw everything up
    weight_df = pd.merge(pd.DataFrame({"subject":loso_sub_col}), 
                               df, on = "subject", how = "left")
    
    
    #Biggest difference is a factor of 76, can try log weights if its a big deal
    
    return weight_df["inverse_prop"].values



def pop_subject(X_features, y_speed, sub_col, subject):
    #"Pop" off data from one subject, return X and y matrices  
    left_out_ix = (sub_col == subject)
    print("Leaving out subject " + subject + "...")
    
    X_loso = X_features[~left_out_ix,:]
    y_loso = y_speed[~left_out_ix]
    
    X_test = X_features[left_out_ix]
    y_test = y_speed[left_out_ix]
    loso_sub_col = sub_col[~left_out_ix]
    
    return (X_loso, y_loso), (X_test, y_test), loso_sub_col


def evaluate_model(model_i, X_test, y_test):
    #Run predict on the model (it should have a predict function)
    #Then return scores    
    #Any model object with a predict method is fine
    y_test_pred = model_i.predict(X_test)
    #Note you will have to internally deal with scaling and centering
    (mae, rmse, mape) = score_metrics(y_test, y_test_pred)
    return mae, rmse, mape

def score_metrics(y_test, y_test_pred):
    mae = np.mean(np.abs(y_test - y_test_pred))
    mape = 100*np.mean(np.abs((y_test - y_test_pred)/y_test))
    rmse = np.sqrt(np.mean((y_test - y_test_pred)**2))
    return mae, rmse, mape

#STILL NEEDS TO INTERPRET DEVICE LOC AND WINDOW SIZE
def save_results(results_dict, predict_vs_actual, 
                 model_name, file_path,
                 include_MIMS,
                 include_anthro,
                 n_search,
                 feature_flag = ""):
    #Set up save
    w_len = file_path.split("_")[3][:2]
    dev_loc = file_path.split("_")[2]
    
    if include_MIMS:
        mims_flag = "_MIMS_1_"
    else:
        mims_flag = "_MIMS_0_"
        
    if include_anthro:
        anthro_flag = "anthro_1_"
    else:
        anthro_flag = "anthro_0_"
    
    #String formatting for save name
    today = datetime.date.today().strftime("%Y_%m_%d")
    right_now = datetime.datetime.now().strftime("%H_%M")
    tstamp = "date_" + today + "_time_" + right_now
    save_name = "cross-validation results/searraw_LOSO_cv_" \
        + dev_loc + "_" + w_len + "s_" + "model_" \
        + model_name + mims_flag + anthro_flag + tstamp \
            + "_nsearch_" + f'{n_search:04}' \
                + feature_flag +  ".csv"
    pred_save_name = "prediction results/searraw_LOSO_predict_" \
        + dev_loc + "_" + w_len + "s_" \
        + model_name + mims_flag + anthro_flag + tstamp \
            + "_nsearch_" + f'{n_search:04}' \
                + feature_flag +  ".csv"
    
    #Save cross validation results
    results_df = pd.DataFrame.from_dict(results_dict) 
    results_df.to_csv(save_name, index=False)
    
    #Save predicted vs actual (with some slick list comprehensions)
    n_samp = [len(i) for i in predict_vs_actual["y_predict"]]
    pred_sub_codes = np.concatenate(
        [np.repeat(n,n_samp[i]) for i, n 
         in enumerate(predict_vs_actual["test_subject"])]
        )
    
    all_predicted_y = np.concatenate(predict_vs_actual["y_predict"])
    all_actual_y = np.concatenate(predict_vs_actual["y_test"])
    
    pred_dict = {"test_subject":pred_sub_codes,
                 "model_predicted_speed":all_predicted_y,
                 "actual_speed":all_actual_y}
    
    pred_df = pd.DataFrame.from_dict(pred_dict)
    pred_df.to_csv(pred_save_name, index=False)
    return None


#Helper to plot results
def print_and_plot_results(results_dict, predict_vs_actual, 
                           model_name, include_MIMS, include_anthro,
                           file_path):
    
    w_len = file_path.split("_")[3][:2]
    dev_loc = file_path.split("_")[2]
    
    results_df = pd.DataFrame.from_dict(results_dict) 
    final_mae = np.mean(results_df["test_mae"])
    final_mape = np.mean(results_df["test_mape"])
    
    print("**************************************************************")
    print("LOSO cross-validated MAE: " + str(final_mae))
    print("LOSO cross-validated MAPE: " + str(final_mape))
    print("**************************************************************")
    
    all_predicted_y = np.concatenate(predict_vs_actual["y_predict"])
    all_actual_y = np.concatenate(predict_vs_actual["y_test"])
    
    if include_MIMS:
        mims_str = "_with_MIMS"
    else:
        mims_str = "_no_MIMS"
        
    if include_anthro:
        anthro_flag = "_with_anthro"
    else:
        anthro_flag = "_no_anthro"

    plt.figure()
    plt.scatter(all_predicted_y, all_actual_y, alpha = 0.1)
    plt.plot(np.linspace(1.8,6), np.linspace(1.8,6), color = "red")
    plt.axis('equal')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    title_str = "[" + model_name + mims_str + "" \
        + anthro_flag + "_" + dev_loc + "_" + w_len + "] - MAE: " \
            + str(round(final_mae,4)) + ", MAPE: " + str(round(final_mape,3))
    plt.title(title_str)
