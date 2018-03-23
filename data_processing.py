# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:52:27 2016

functions usable to build models

@author: fanheng
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import xgboost as xgb
# from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import math

#one-hot-encoding
def factorToDummNoBase(data, varname, drop_origin = False):
    dummies = pd.get_dummies(data[varname],prefix=varname, prefix_sep='=',dummy_na=False)
    base_level = dummies.sum().idxmax()
    dummies.drop(base_level, inplace=True, axis=1)
    data = pd.concat([data, dummies], axis=1)
    if drop_origin:
        data.drop(varname, inplace=True, axis=1)
    return(data)

def factorToDumm(data, varname, drop_origin = False):
    dummies = pd.get_dummies(data[varname],prefix=varname, prefix_sep='=', dummy_na=False)
    data = pd.concat([data, dummies], axis=1)
    if drop_origin:
        data.drop(varname, inplace=True, axis=1)
    return(data)


def percentileScore(s, weighted=True):
    # s: pandas Series
    # weighted: default True. Whether to consider the values
    idx = s.index.copy()
    if weighted==True:
        pscore = (s.sort_values().cumsum() * 1.0 / s.sum()).reindex(idx)
    else:
        pscore = pd.Series(np.arange(1.0, len(s) + 1, 1) / len(s), index=idx)
    pscore.index.name = 'index'
    return(pscore)

def getLOO(train_x, label, categorical):
    train = train_x[[categorical, label]]
    cs = train.groupby(categorical)[label].sum()
    cc = train[categorical].value_counts()
    boolean = (cc == 1)
    index = boolean[boolean == True].index.values
    cc.loc[boolean] += 1
    cs.loc[index] *= 2
    train = train.join(cs.rename('sum'), on=[categorical])
    train = train.join(cc.rename('count'), on=[categorical])
    return ((train['sum'] - train[label])/(train['count'] - 1))
