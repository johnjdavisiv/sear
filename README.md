# SEAR: Speed estimation algorithm for running
Estimate running speed from raw wrist or waist-worn accelerometer data

Please cite:  
>Davis, J., Oeding, B., and Gruber, A., 2022. Estimating Running Speed From Wrist- or Waist-Worn Wearable Accelerometer Data: A Machine Learning Approach. Journal for the Measurement of Physical Behaviour, ahead of print. DOI: 10.1123/jmpb.2022-0011

## Downloading the data

To reproduce the paper results you will need to download the data from [FigShare DOI: 10.6084/m9.figshare.21507180](https://figshare.com/articles/dataset/SEAR_running_speed_estimation_data/21507180)  

The data should be unzipped to the `/raw data/` folder.   

## Reproducing the paper

First: run `extract_features.py` to extract features from raw 10sec windows. These are saved in `/feature exports/`

Second: run `main.py` to run nested subject-wise cross-validation. This writes result file to `/cross-validation results` and `prediction results`.  

For speed, `main.py` is set up to reproduce the ridge model results with fewer random searches of the hyperparameter space than the paper (here, 200; paper is 1000). Change `"ridge"` to `"xgb"` in the code, and `n_search_list` to be `[50]*2` to reproduce the xgboost model results (warning, this could take ~12hrs or more on a fast desktop computer).

## Using on new data  

*(Demo forthcoming)*

1. Resample to 100 Hz if necessary
2. Calculate resultant acceleration, in g-units.
3. Extract bouts of running ([perhaps using a running recognition algorithm](https://github.com/johnjdavisiv/carl))  
4. Window data into non-overlapping 10 second, 100 Hz windows and reshape into an n x 1000 numpy array. The helper function in `sear_features.py` might be useful here.   
5. Extract matrix of features using `sear_features.extract_features()`  
6. Feed into pre-trained algorithm using  `.predict()`` method  
