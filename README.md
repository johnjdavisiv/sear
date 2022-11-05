# SEAR: Speed estimation algorithm for running
Estimate running speed from raw wrist or waist-worn accelerometer data


>Citation goes here.

## Downloading the data

To reproduce the paper results you will need to download the data from DOI:[DOI_LINK](http://doi.org/)  

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
