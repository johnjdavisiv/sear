"""
Export features from raw accelerometer data (10sec windows)


"""

import os
import pandas as pd
import time
import sear_features

#tic
start = time.time()

all_files = [file for _, _, file in os.walk('raw data')][0]

for fname in all_files:
    print("Processing: " + fname)
    file_path = "raw data/" + fname
    this_df = pd.read_csv(file_path)
    
    #metadata from file name
    device_loc = fname[14:18]
    w_len = fname[19:21]
    save_name = 'SEAR_featurized_' + device_loc + '_' + w_len + 's_windows.csv' 
    
    accel_R = this_df.loc[:,"r0001":].to_numpy()
    
    #Get features - including slow entropy ones
    (R_features, col_names) = sear_features.extract_features(accel_R, slow_features = True)
    
    #To df, save
    feature_df = pd.DataFrame(R_features, columns = col_names)
    meta_df = this_df.loc[:, :"speed"]
    all_df = pd.concat([meta_df, feature_df], axis=1)
    all_df.to_csv('feature exports/' + save_name, index = False)

    
print("Finished!")
end = time.time()
print("Total time elapsed: {} sec".format(end - start))