
# from numpy import loadtxt
import random
# from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_sample_index(data_size, sample_size):
    '''
    data_size: the length of the data frame
    sample_size: percentage (0, 100) 
    '''
    idx = random.sample(range(data_size), sample_size)
    return(idx)

def ramdom_sampling(df, sample_size, sample_by=None, **kwargs):
    '''
    df: pandas dataframe
    sample_size: fraction of rows to use
    sample_by: the level to sample from, default is None. If not None, random sampling on the variable level.
    **kwargs: any arguments applied to pandas.DataFrame.sample
    '''
    if sample_by is None:
        return(df.sample(frac=sample_size, **kwargs))
    else:
        if sample_by not in df.columns:
            raise ValueError("%s not in data frame columns" % sample_by)
        df.index = df[sample_by]
        sample_idx = random.sample(df.index, sample_size)
        df_sample = df.ix[sample_idx].reset_index()
        return(df_sample)
        