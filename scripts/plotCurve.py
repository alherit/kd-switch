# -*- coding: utf-8 -*-
"""
Created on Fri May 10 22:54:09 2019

@author: alheritier
"""

import matplotlib
matplotlib.use('Agg')


import os

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns



def plotCurveTime(basedir,n_trials,resolution, max_time=None , log_x = False):
    print("n_trials",n_trials)
    print("resolution: " , resolution)

    dataset =  os.path.basename(os.path.normpath(basedir))
    print(dataset)

    algorithms = next(os.walk(basedir))[1]
    algorithms.sort()
    print("algorithms: " ,algorithms)
    
    dfs = []
    for trial in range(n_trials):
        fname = "probs_seq_" + str(trial) + ".dat"

        for algo in algorithms:
            if not algo.startswith("_"):
                try:
    
                    df = pd.read_csv(basedir +"/" + algo + "/" +  fname, sep = " ", header=None, names=["time","prob","NLL"])
                    df["algorithm"] = algo
                    df["trial"] = trial
            
            
                    #create timestamps in nanoseconds
                    df.index = pd.DatetimeIndex(df["time"]*(10**9))
        
                    #change resolution, take last value for each interval
                    df = df.resample(str(resolution)+"s").last()
                    # fillin nans with linear interpolation
                    df["NLL"] = df["NLL"].interpolate(method="linear")
                    del df["time"]
                    df = df.reset_index()
                    #get back to seconds
                    df["elapsed_time (s)"] = (df["time"] - df.iloc[0]["time"]).dt.seconds
                    
                    if max_time is not None:
                        df = df[df["elapsed_time (s)"] <= max_time]
                    
                    dfs.append(df)    
                
                except Exception as e:
                    print("error: " + str(e))

            
    df = pd.concat(dfs)
    
    print(df["algorithm"].unique())
    
    ax = sns.lineplot(x="elapsed_time (s)", y="NLL",
                 hue="algorithm", ci="sd",
                 data=df)

    if log_x:
        ax.set(xscale="log")

    ax.set_title(dataset+ " n_trials=" + str(n_trials))

    opt = "" if max_time is None else "max-x"+str(max_time) 
    plt.savefig(dataset+opt+"-time.pdf", bbox_inches = 'tight')


def plotCurveNVsTime(basedir,n_trials,resolution, max_time=None ,min_y = None ,  max_y = None, log_x = False):
    print("n_trials",n_trials)
    print("resolution: " , resolution)

    dataset =  os.path.basename(os.path.normpath(basedir))
    print(dataset)

    algorithms = next(os.walk(basedir))[1]
    algorithms.sort()
    print("algorithms: " ,algorithms)
    
    dfs = []
    for trial in range(n_trials):
        fname = "probs_seq_" + str(trial) + ".dat"

        for algo in algorithms:
            if not algo.startswith("_") and not algo == "true":
                try:
    
                    df = pd.read_csv(basedir +"/" + algo + "/" +  fname, sep = " ", header=None, names=["time","prob","NLL"])
                    df["algorithm"] = algo
                    df["trial"] = trial

                    df.reset_index(inplace=True)
                    df = df[df["index"]%resolution == 0]       
                    df.rename(columns={'index':'n'}, 
                         inplace=True)

            
                    #create timestamps in nanoseconds
                    df.index = pd.DatetimeIndex(df["time"]*(10**9))
        
                    #change resolution, take last value for each interval
                    df = df.resample(str(resolution)+"s").last()
                    # fillin nans with linear interpolation
                    df["NLL"] = df["NLL"].interpolate(method="linear")
                    del df["time"]
                    df = df.reset_index()
                    #get back to seconds
                    df["elapsed_time (s)"] = (df["time"] - df.iloc[0]["time"]).dt.seconds
                    
                    if max_time is not None:
                        df = df[df["elapsed_time (s)"] <= max_time]

                    
                    dfs.append(df)    
                
                except Exception as e:
                    print("error: " + str(e))

            
    df = pd.concat(dfs)
    
    print(df["algorithm"].unique())
    
    ax = sns.lineplot(y="n", x="elapsed_time (s)",
                 hue="algorithm", ci="sd",
                 data=df)

    if log_x:
        ax.set(xscale="log")
        
    if min_y is not None:
        ax.set_ylim(bottom=min_y)
    if max_y is not None:
        ax.set_ylim(top=max_y)


    ax.set_title(dataset+ " n_trials=" + str(n_trials))

    opt = "" if max_time is None else "max-x"+str(max_time) 
    plt.savefig(dataset+opt+"-n-vs-time.pdf", bbox_inches = 'tight')


def plotCurve(basedir,n_trials,resolution, max_time=None , min_y = None , max_y = None ,log_x = False, entropy = None):
    print("n_trials",n_trials)
    print("resolution: " , resolution)

    dataset =  os.path.basename(os.path.normpath(basedir))
    print(dataset)

    algorithms = next(os.walk(basedir))[1]
    algorithms.sort()
    print("algorithms: " ,algorithms)
    
    dfs = []
    for trial in range(n_trials):
        fname = "probs_seq_" + str(trial) + ".dat"
    
        for algo in algorithms:
            if not algo.startswith("_"):
                try:
                    df = pd.read_csv(basedir +"/" + algo + "/" +  fname, sep = " ", header=None, names=["time","prob","NLL"])
                    df["algorithm"] = algo
                    df["trial"] = trial
                    df.reset_index(inplace=True)
                    
            
                    df = df[df["index"]%resolution == 0]       
        
                    df.rename(columns={'index':'n'}, 
                         inplace=True)
                    
                    if max_time is not None:
                        df = df[df["n"] <= max_time]
    
                    
                    dfs.append(df)    
    
                except Exception as e:
                    print("error: " + str(e))
            
    df = pd.concat(dfs)
    
    
    ax = sns.lineplot(x="n", y="NLL",
                 hue="algorithm", ci="sd",
                 data=df)

    if log_x:
        ax.set(xscale="log")

    if min_y is not None:
        ax.set_ylim(bottom=min_y)
    if max_y is not None:
        ax.set_ylim(top=max_y)

    if entropy is not None:
        plt.axhline(y=entropy, c='red', linestyle='dashed', label="H(L|Z)")
        plt.legend()

    ax.set_title(dataset+ " n_trials=" + str(n_trials))
    
    opt = "" if max_time is None else "max-x"+str(max_time) 
    plt.savefig(dataset+opt+"-nll-vs-n.pdf", bbox_inches = 'tight')


def plotCurveTimeVsN(basedir,n_trials,resolution, max_time=None , min_y = None , max_y = None ,log_x = False):
    print("n_trials",n_trials)
    print("resolution: " , resolution)

    dataset =  os.path.basename(os.path.normpath(basedir))
    print(dataset)

    algorithms = next(os.walk(basedir))[1]
    algorithms.sort()
    print("algorithms: " ,algorithms)
    
    dfs = []
    for trial in range(n_trials):
        fname = "probs_seq_" + str(trial) + ".dat"
    
        for algo in algorithms:
            if not algo.startswith("_") and algo!="true":
                try:
                    df = pd.read_csv(basedir +"/" + algo + "/" +  fname, sep = " ", header=None, names=["time","prob","NLL"])
                    df["algorithm"] = algo
                    df["trial"] = trial
                    df.reset_index(inplace=True)
                    
            
                    df = df[df["index"]%resolution == 0]       

                    df.rename(columns={'index':'n'}, 
                         inplace=True)
                    #print("filtering by max_time: ",max_time)
                    if max_time is not None:
                        df = df[df["n"] <= max_time]
    
                    
                    #create timestamps in nanoseconds
                    df.index = pd.DatetimeIndex(df["time"]*(10**9))
                    del df["time"]
                    df = df.reset_index()
                    #get back to seconds
                    df["elapsed_time (s)"] = (df["time"] - df.iloc[0]["time"]).dt.seconds
                    
                    
                    
                    dfs.append(df)    
    
                except Exception as e:
                    print("error: " + str(e))
            
    df = pd.concat(dfs)
    
    
    ax = sns.lineplot(x="n", y="elapsed_time (s)",
                 hue="algorithm", ci="sd",
                 data=df)

    if log_x:
        ax.set(xscale="log")

    if min_y is not None:
        ax.set_ylim(bottom=min_y)
    if max_y is not None:
        ax.set_ylim(top=max_y)

    ax.set_title(dataset+ " n_trials=" + str(n_trials))
    
    opt = "" if max_time is None else "max-x"+str(max_time) 
    plt.savefig(dataset+opt+"-time-vs-n.pdf", bbox_inches = 'tight')



def foo_callback2(option, opt, value, parser):
  setattr(parser.values, option.dest,  value.split(',') )



if __name__ == "__main__":
    
    from optparse import OptionParser

    parser = OptionParser()

    parser.add_option('-d','--basedir', dest='basedir', type='string',
                      help='basedir [default: %default]', default=".")

    parser.add_option('-r', '--resolution', dest='resolution', type='int',
                      help='index/time resolution [default: %default]', default=1000)


    parser.add_option('-n', '--nTrials', dest='nTrials', type='int',
                      help='n trials to consider [default: %default]', default=10)
    
    parser.add_option('-t','--time-plot', action="store_true", dest='time_plot', 
                      help='time plot [default: %default]', default=False)

    parser.add_option('-T', '--max-time', dest='max_time', type='float',
                      help='max_time [default: %default]', default=None)



    parser.add_option('--min-y', dest='min_y', type='float',
                      help='min_y [default: %default]', default=None)
    parser.add_option('--max-y', dest='max_y', type='float',
                      help='max_y [default: %default]', default=None)


    parser.add_option('-c','--time-vs-n', action="store_true", dest='time_vs_n', 
                      help='time vs n [default: %default]', default=False)


    parser.add_option('-g','--log-x', action="store_true", dest='log_x', 
                      help='log x axis [default: %default]', default=False)

    parser.add_option('--entropy', dest='entropy', type='float',
                      help='entropy [default: %default]', default=None)
    

    (options, args) = parser.parse_args()


    if options.time_plot:
        plotCurveTime(options.basedir, options.nTrials, options.resolution, options.max_time, options.log_x)
    elif options.time_vs_n:
        plotCurveTimeVsN(options.basedir, options.nTrials, options.resolution, options.max_time, options.min_y, options.max_y, options.log_x)
    else:
        plotCurve(options.basedir, options.nTrials, options.resolution, options.max_time, options.min_y, options.max_y, options.log_x, options.entropy)