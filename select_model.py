
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
