# -*- coding: utf-8 -*-
"""
Created on Fri Sep 09 14:22:29 2016
"""

#------------------------------------------------------------------------------
# Initiating

from __future__ import print_function, division

import pandas as pd
import numpy as np
from time import time

import os
os.chdir('D:\\Kaggle\\BoshProduction\\')
print('Current location:', os.getcwd())

#---------------------------------------------------------------------------
# My functions

# Function to print out structure of data (mimic str() in R)
def strdf(df):
    print(type(df), ':\t', df.shape[0], 'obs. of', df.shape[1], 'variables:')
    if df.shape[0] > 4:
        df = df.head(4) # Take first 4 obs.
        dots = '...' # Print ... for the rest values  
    else:
        dots = ''
    space = len(max(list(df), key=len))
    for c in list(df):
        print(' $', '{:{align}{width}}'.format(c, align='<', width=space),
              ':', df[c].dtype, str(df[c].values)[1:-1], dots)

# Function to print out NAN values and their data types
def nadf(df):
    print(type(df), ':\t', df.shape[0], 'obs. of', df.shape[1], 'variables:')
    df_type = df.dtypes    
    df_NA = df.isnull().sum()
    space = len(max(list(df), key=len))
    space_type = len(max([d.name for d in df.dtypes.values], key=len))
    space_NA = len(str(max(df_NA)))
    for c in list(df):
        print(' $', '{:{align}{width}}'.format(c, align='<', width=space),
              ':', '{:{align}{width}}'.format(df_type[c], align='<', width=space_type),
              '{:{align}{width}}'.format(df_NA[c], align='>', width=space_NA),
              'NAN value(s)')

# Function to convert categorical, boolean, datetime variables to integer
def var_to_int(df):
    X = df.copy(deep=True)
    vars_type = X.dtypes

    # Convert categorical vars
    categorical_list = list(X.columns[vars_type == 'object'].values)
    for c in categorical_list:
        X[c].fillna('NAN', inplace=True)
        X[c] = pd.factorize(X[c])[0]
    
    # Convert boolean vars
    bool_list = list(X.columns[vars_type == 'bool'].values)
    for c in bool_list: X[c] = X[c].astype(np.int8)
        
    # Convert datetime vars
    datetime_list = list(X.columns[vars_type == 'datetime64[ns]'].values)
    for c in datetime_list:
        X[str(c) + '_day'] = X[c].dt.day
        X[str(c) + '_month'] = X[c].dt.month
        X[str(c) + '_year'] = X[c].dt.year
        X[str(c) + '_isweekend'] = (X[c].dt.weekday >= 5).astype(np.int8)
        X = X.drop(c, axis=1)
    
    return X
        
#---------------------------------------------------------------------------
# Explore data

# Files are too big to read all in memory
# read some lines to identify the data type first

sample_submission = pd.read_csv('sample_submission.csv')
strdf(sample_submission)

train_numeric = pd.read_csv('train_numeric.csv', nrows=100)
strdf(train_numeric)

#---------------------------------------------------------------------------
# Import data

# Read full training set
var_train_type = {}
for v in list(train_numeric):
    var_train_type[v] = np.float16
var_train_type = {'Id':np.int64}
var_train_type['Response'] = np.int8

train = pd.read_csv('train_numeric.csv', dtype=var_train_type)
X = train.iloc[:, 1:-1]
y = train.iloc[:, -1]

del train # Save memory

# Dimensional reduction
from scipy.stats import pearsonr

X.shape
X.fillna(99, inplace=True) # Fill NA = 0

var_list = []
for c in list(X):
    if pearsonr(X[c], y)[1] <= 0.05: # p <= 0.05
        var_list.append(c)

X[var_list].shape
X = X[var_list] # Save memory

# Cross-validation
from sklearn.cross_validation import train_test_split

# Split data sets, prepare to train model
X_train_, _, y_train_, _ = train_test_split(X[var_list], y, train_size=0.3,
                                            random_state=123, stratify=y)

import xgboost as xgb

dtrain = xgb.DMatrix(X_train_, label=y_train_)

# Set xgboost params
param = {'max_depth':10, 'eta':0.02, 'silent':1, 'objective':'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['min_child_weight'] = 0
param['booster'] = "gblinear"
param['scale_pos_weight'] = 1176868/6879

# Run cross-validation to have an idea about the final result
watchlist = [(dtrain, 'train')]
num_round = 100
num_fold = 3
early_stopping_rounds = 10
xgb_cv = xgb.cv(param, dtrain, num_round, num_fold, watchlist,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=True, seed=0)

# del X_train_, y_train_ # Save memory
                
#---------------------------------------------------------------------------
# Actual train the xgboost model on all training set

# Re-train the model
# dtrain = xgb.DMatrix(X[var_list], label=y)
dtrain = xgb.DMatrix(X_train_, label=y_train_)

watchlist = [(dtrain, 'train')]
num_round = 500
early_stopping_rounds = 10
xgb_model = xgb.train(param, dtrain, num_round, watchlist,
                      early_stopping_rounds=early_stopping_rounds)

# Apply the model on test set and save the output
var_test_type = var_train_type.copy()
del var_test_type['Response']

test_reader = pd.read_csv('test_numeric.csv', dtype=var_test_type,
                          chunksize=10000)

test_Id = []
y_pred = []
count = 0
for chunk in test_reader:
    count += len(chunk)
    print('Read', count, 'lines')
    test_Id += list(chunk.Id.values)
    dtest = xgb.DMatrix(chunk[var_list].fillna(99))
    y_pred += list(xgb_model.predict(dtest))

output = pd.DataFrame({'Id':test_Id, 'Response':np.round(y_pred).astype(int)})
output.head() # Print out some output
output.to_csv('submission_xgboost_160916_3.csv', index=False)
