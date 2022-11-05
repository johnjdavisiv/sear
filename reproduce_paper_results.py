# -*- coding: utf-8 -*-
"""
Reproduce results from SEAR paper
JJD

"""

from build_sear_model import crossval_sear_model

raw_file_list = []
base_path = 'feature exports/'

import re
import os
for dirname, _, filenames in os.walk(base_path):
    for filename in filenames:
        raw_file_list.append(filename)

r = re.compile(".*.csv")
sear_files = list(filter(r.match, raw_file_list)) 
print(sear_files)

# ---- Run through some models
include_MIMS = True      #<----- Should we include Qu et al MIMSunit data? n=10
include_anthro = False   #<----- Should height & weight be used as features? Precludes using MIMS
only_anthro = False      #<----- For testing null models 
k_fold = 8               #<----- Number of internal subjectwise-XV folds

file_list = [sear_files[0]]  + [sear_files[1]]  #Wrap in brackets if you are just doing one file
model_list = ["ridge"]*2 #"ridge" or "xgb"
n_search_list = [100]*2 #Number of random hyperparam searches
#Paper used 1000 for ridge and 50 for xgb

for this_file, this_model, this_n_search in zip(file_list, model_list, n_search_list):
    crossval_sear_model(base_path + this_file, this_model, k_fold, this_n_search,
                            include_MIMS, include_anthro,
                            only_anthro,
                            feature_flag = "")



