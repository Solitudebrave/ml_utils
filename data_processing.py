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

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import math


def prefilter(df, conditions=[]):
    # df: pandas dataframe
    # groupby: if not None, apply conditions in each group
    # conditions: a list of conditions in string
    if conditions == []:
        return(df)
    else:
        conditions_str = " & ".join(conditions)
        return(df.query(conditions_str))

def prefilter_by_group(df, groupby=None, cond_func=None):
    # cond_func:
    # example: def cond_func(g):
    # ...     return(max(g['B']) >1)
    if groupby == None:
        print("No filter applied")
        return(df)
    else:
        gg = df.groupby(groupby)
        return(gg.filter(lambda g: cond_func(g)))


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


def defineEstimator(estimator):
    estimators = {'xgb': XGBRegressor()
                 ,'rf': RandomForestRegressor()}
    return(estimators[estimator])


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

def split_train_holdout(df, split_key):
    train_x, holdout_x = revenue.split_train_holdout_data(df, split_key=split_key)
    return train_x, holdout_x

def prefilter_split_train_holdout(df, split_key):
    def cond_func(g):
        return(len(g)>=100)
    df2 = (prefilter(prefilter_by_group(
                                    df, groupby='dim_user_market', cond_func=cond_func)
                                , conditions=['listing_host_payout_360d >= 1000'
                                              , 'listing_host_payout_180d > 0'
                                              ]))
    train_x, holdout_x = revenue.split_train_holdout_data(df2, split_key=split_key)
    return train_x, holdout_x

def split_train_holdout_data(df, split_key=None, method='random', validation='CV', **kwargs):
    # this is a pandas function. self is a pandas dataframe
    # validation: if the method is 'CV', it won't split train and test, but only separate the holdouot set
    if method == 'random':
        if split_key is None:
            split_key_group = pd.DataFrame(data=np.random.choice(['train', 'test', 'holdout'], size=len(df), p=[0.7,0.2,0.1]), index=df.index, columns=['group'])
            X = df.copy().merge(split_key_group, how='left', left_index=True, right_index=True)
        else:
            split_key_group = pd.DataFrame(data=np.random.choice(['train', 'test', 'holdout'], size=len(df[split_key].unique()), p=[0.7,0.2,0.1]), index=df[split_key].unique(), columns=['group'])
            X = df.copy().merge(split_key_group, how='left', left_on=split_key, right_index=True)
        if validation == 'CV':
            return(X[X['group'] != 'holdout'].reset_index(drop=True), X[X['group'] == 'holdout'].reset_index(drop=True))
        else:
            return(X[X['group'] == 'train'].reset_index(drop=True), X[X['group'] == 'test'].reset_index(drop=True), X[X['group'] == 'holdout'].reset_index(drop=True))
    elif method == 'prepick':
        return(df[df[split_key].isin(kwargs['prepick_list'])].reset_index(drop=True), df[~df[split_key].isin(kwargs['prepick_list'])].reset_index(drop=True))
    else:
        pass
