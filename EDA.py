# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 20:39:23 2016

EDA functions

@author: fanheng
"""

'''
Sample Data:
raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'],
        'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'],
        'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'],
        'weight': [2, 4, 5, 3, 6, 3, 7, 3, 1, 7, 8, 4],
        'preTestScore': [4, 24, 31, 2, 3, 4, 24, np.nan, 2, 3, 2, 3],
        'postTestScore': [25, np.nan, 57, 62, 70, 25, 94, np.nan, 62, 70, 62, 70]}
df = pd.DataFrame(raw_data, columns = ['regiment', 'company', 'name', 'weight', 'preTestScore', 'postTestScore'])

'''


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def unique_count(df, var_names):
    '''
    df: pandas dataframe
    var_names: list of column feature names
    '''
    def get_unique_count(var):
        return("Unique count for" + var+ "is %s" %len(df[var].unique()))
    cnts = [get_unique_count(var) for var in var_names]
    for c in cnts:
        print(c)


def eda_hist(df, var_name):
    '''
    df: pandas dataframe
    var_name: numeric variable
    '''
    #remove NaN values
    myseries = df[var_name][df[var_name].notnull()]
    #histgram = plt.figure()
    bins = np.linspace(0.8*min(myseries), 1.2*max(myseries), 100)
    histgram = plt.figure()
    plt.hist(myseries, bins, alpha=0.5)
    quantiles = myseries.quantile([.0, .01, .05, .25, .5, .75, .9, .95, .99, 1])
    values = np.array(quantiles)
    #indexs = np.array(quantiles.index)
    indexs = np.array(['min', '1%', '5%', '25%', '50%', '75%', '90%', '95%', '99%', 'max'])
    table_vals = np.vstack((indexs, values)).transpose()
    the_table = plt.table(cellText=table_vals,
                  colWidths = [0.1]*3,
                  loc='center right')
    plt.title(var_name)
    plt.show(histgram)
    plt.show(the_table)

'''
fig,axs = plt.subplots(1,2)
axs = axs.ravel()
varnn = ['tenure_length', 'tenure_length']
for idx,ax in enumerate(axs):
    #print(idx)
    eda_hist2(df, varnn[idx], ax)

'''

def eda_hist2(df, var_name, ax):
    '''
    df: pandas dataframe
    var_name: numeric variable
    '''
    #remove NaN values
    myseries = df[var_name][df[var_name].notnull()]
    #histgram = plt.figure()
    bins = np.linspace(0.8*min(myseries), 1.2*max(myseries), 100)
    ax.hist(myseries, bins, alpha=0.5)
    quantiles = myseries.quantile([.0, .01, .05, .25, .5, .75, .9, .95, .99, 1])
    values = np.array(quantiles)
    #indexs = np.array(quantiles.index)
    indexs = np.array(['min', '1%', '5%', '25%', '50%', '75%', '90%', '95%', '99%', 'max'])
    table_vals = np.vstack((indexs, values)).transpose()
    ax.table(cellText=table_vals,
                  colWidths = [0.1]*3,
                  loc='center right')
    ax.set_title(var_name)
    #plt.show(the_hist)
    #plt.show(the_table)


def eda_char(df, var_name, output=True):
    '''
    df: pandas dataframe
    var_name: categorical variable
    '''
    df2 = pd.DataFrame(df[var_name])
    df2 = df2.fillna('missing')
    grp_size = df2.groupby(var_name).size()
    index = np.arange(len(grp_size))
    grp_ptcg = grp_size/len(df)
    grp_df = pd.DataFrame({'grp_size': grp_size, 'grp_ptcg': grp_ptcg})

    # plot bar chart
    fig, ax1 = plt.subplots(figsize=(20, 4))
    ax2 = ax1.twinx()
    width=0.5
    ax1.bar(index, grp_size,width,
                color='blue',
                alpha=0.5)
    ax2.plot(index+0.5*width, grp_ptcg,
                         color='blue')
    # axes and labels
    ax1.set_ylabel('counts')
    ax2.set_ylabel('percentage')
    ax1.set_xlim(-width,len(index)+width)
    ax1.set_xticks(index+0.5*width)
    xTickMarks = grp_size.index.tolist()
    xtickNames = ax1.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=12)
    plt.title(var_name)
    if output:
        return(grp_df)

def eda_char2(df, var_name, ax1):
    '''
    df: pandas dataframe
    var_name: categorical variable
    '''
    df2 = pd.DataFrame(df[var_name])
    df2[var_name] = df2[var_name].cat.add_categories(['missing'])
    df2 = df2.fillna('missing')
    grp_size = df2.groupby(var_name).size()
    index = np.arange(len(grp_size))
    grp_ptcg = grp_size/len(df)
    #grp_df = pd.DataFrame({'grp_size': grp_size, 'grp_ptcg': grp_ptcg})

    # plot bar chart
    ax2 = ax1.twinx()
    width=0.5
    ax1.bar(index, grp_size,width,
                color='blue',
                alpha=0.5)
    ax2.plot(index+0.5*width, grp_ptcg,
                         color='blue')
    # axes and labels
    ax1.set_ylabel('counts')
    ax2.set_ylabel('percentage')
    ax1.set_xlim(-width,len(index)+width)
    ax1.set_xticks(index+0.5*width)
    xTickMarks = grp_size.index.tolist()
    xtickNames = ax1.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=10)
    ax1.set_title(var_name)

def uni_actual(df, var_name, target, weight=None, save=False, plot=True, output=True):
    '''
    df: pandas dataframe
    var_name: categorical variable
    target: target variable
    '''
    df2 = df.ix[:, [var_name, weight, target]]
    #if df[var_name].isnull().any():
     #   df[var_name] = df[var_name].cat.add_categories(['missing'])
    #df = df.fillna('missing')

    if weight is None:
        df2 = df.ix[:, [var_name, target]]
    else:
        df2 = df.ix[:, [var_name, weight, target]]

    if df2[var_name].isnull().any():
        df2[var_name] = df2[var_name].astype('category')
        df2[var_name] = df2[var_name].cat.add_categories(['missing'])
        df2[var_name] = df2[var_name].fillna('missing')

    if weight is None:
        grp_size = df2.groupby(var_name).size()
        index = np.arange(len(grp_size))
        grp_act = df2[[var_name, target]].groupby(var_name).mean().ix[:,0]
        grp_df = pd.DataFrame({'grp_size': grp_size, 'grp_actual': grp_act})

        if plot:
            # plot bar chart
            fig, ax1 = plt.subplots(figsize=(20, 6))
            ax2 = ax1.twinx()
            width=0.5
            ax1.bar(index, grp_size,width,
                        color='grey',
                        alpha=0.5)
            ax2.plot(index+0.5*width, grp_act,
                             color='blue')
            # axes and labels
            ax1.set_ylabel('counts')
            ax2.set_ylabel('actual' + target + 'rate')
            ax1.set_xlim(-width,len(index)+width)
            ax1.set_xticks(index+0.5*width)
            xTickMarks = grp_size.index.tolist()
            xtickNames = ax1.set_xticklabels(xTickMarks)
            for i in np.arange(len(index)):
                ax2.text(index[i], grp_act[i], round(grp_act[i],2), horizontalalignment='center', verticalalignment='top')
            plt.setp(xtickNames, rotation=45, fontsize=10)
            plt.title(var_name)
            img_name = "actual" + target + "by" + var_name + '.png'
            if save:
                plt.savefig(img_name)
    else:
        grp_size = df2.groupby(var_name).sum()
        index = np.arange(len(grp_size))
        grp_act = df2[[var_name, target]].groupby(var_name).mean()
        grp_df = pd.merge(grp_size, grp_act, left_index=True, right_index=True)

        if plot:
            # plot bar chart
            fig, ax1 = plt.subplots(figsize=(20, 6))
            ax2 = ax1.twinx()
            width=0.5
            ax1.bar(index, grp_df[weight],width,
                       color='grey',
                       alpha=0.5)
            ax2.plot(index+0.5*width, grp_df[target],
                             color='blue')
            # axes and labels
            ax1.set_ylabel(weight)
            ax2.set_ylabel('actual' + target + 'rate')
            ax1.set_xlim(-width,len(index)+width)
            ax1.set_xticks(index+0.5*width)
            xTickMarks = grp_size.index.tolist()
            xtickNames = ax1.set_xticklabels(xTickMarks)
            plt.setp(xtickNames, rotation=45, fontsize=10)
            plt.title(var_name)
            img_name = "actual" + target + "by" + var_name + '.png'
            if save:
                plt.savefig(img_name)

    if output:
        return(grp_df)

def uni_bin_actual(df, var_name, target, bins=None, weight=None, save=False):
    '''
    df: pandas dataframe
    var_name: numerical variable
    target: target variable
    bins: list, given bin range, if None, using quantiles
    weight: if None, return the counts for each bins, otherwise the sum of weight
    save: flag to save image
    '''

    if weight is None:
        dff = df.ix[:, [var_name, target]]
    else:
        if df[weight].isnull().any():
            print("Can't have missing value in the weight variable")
            raise
        dff = df.ix[:, [var_name, target, weight]]

    var_gp = var_name+'cut'
    if bins is not None:
        dff[var_gp] = pd.cut(dff[var_name], bins)
    else:
        dff[var_gp] = pd.qcut(dff[var_name], [.01, .1, .2, .3,.4, .5, .6,.7, .8, .9, .99])

    dff[var_gp] = dff[var_gp].astype('category')
    if dff[var_gp].isnull().any():
        dff[var_gp] = dff[var_gp].cat.add_categories(['missing'])
        dff[var_gp] = dff[var_gp].fillna('missing')

    #heritage from uni_actual()
    if weight is None:
        df2 = pd.DataFrame(dff[var_gp])
    else:
        df2 = dff.ix[:, [var_gp, weight]]

    if weight is None:
        grp_size = df2.groupby(var_gp).size()
        index = np.arange(len(grp_size))
        grp_act = dff[[var_gp, target]].groupby(var_gp).mean().ix[:,0]
        grp_df = pd.DataFrame({'grp_size': grp_size, 'grp_actual': grp_act})
        # plot bar chart
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        width=0.5
        ax1.bar(index, grp_size,width,
                    color='grey',
                    alpha=0.5)
        ax2.plot(index+0.5*width, grp_act,
                         color='blue')
        # axes and labels
        ax1.set_ylabel('counts')
        ax2.set_ylabel('actual' + target + 'rate')
        ax1.set_xlim(-width,len(index)+width)
        ax1.set_xticks(index+0.5*width)
        xTickMarks = grp_size.index.tolist()
        xtickNames = ax1.set_xticklabels(xTickMarks)
        plt.setp(xtickNames, rotation=45, fontsize=10)
        plt.title(var_name)
        img_name = "actual" + target + "by" + var_name + '.png'
        plt.savefig(img_name)
    else:
        grp_size = df2.groupby(var_gp).sum()
        index = np.arange(len(grp_size))
        grp_act = dff[[var_gp, target]].groupby(var_gp).mean()
        grp_df = pd.merge(grp_size, grp_act, left_index=True, right_index=True)
        #grp_df[weight] = grp_df[weight].fillna(0.000001)

        # plot bar chart
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        width=0.5
        ax1.bar(index, grp_df[weight],width,
                   color='grey',
                   alpha=0.5)
        ax2.plot(index+0.5*width, grp_df[target],
                         color='blue',
                         linestyle='-', marker='o')
        # axes and labels
        ax1.set_ylabel(weight)
        ax2.set_ylabel('actual' + target + 'rate')
        ax1.set_xlim(-width,len(index)+width)
        ax1.set_xticks(index+0.5*width)
        xTickMarks = grp_df.index.tolist()
        xtickNames = ax1.set_xticklabels(xTickMarks)
        plt.setp(xtickNames, rotation=45, fontsize=10)
        plt.title(var_name)
        img_name = "actual" + target + "by" + var_name + '.png'
        plt.savefig(img_name)
        del((grp_size, grp_act, grp_df, df2, dff))
    #return(grp_df)
