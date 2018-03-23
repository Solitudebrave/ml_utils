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

pd.to 

