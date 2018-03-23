
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def liftChart(data,actual, pred):
    a = data.ix[:,[actual, pred]].sort_values(pred)
    a.ix[:,'pred_g'] = pd.cut(np.arange(len(a)), 10, labels = np.arange(1,11,1))
    ax = a.groupby('pred_g').mean()[actual].plot(title='Lift Chart')
    ax.set_xlabel('predicted group')
    ax.set_ylabel('actual '+ actual+' rat')
    print("The lift is {}".format(a.groupby('pred_g').mean()[actual][-1]/a.groupby('pred_g').mean()[actual].mean()))