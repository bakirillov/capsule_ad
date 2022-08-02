"""The module with auxilliary functions for reading the result files
This file provides the definitions for all functions that read results.
This file is to be imported as a module and exports the following:
    proportions - computes proportions of normal and anomalous cases in the file
    read_ABCNL - read the results for ABC and NL studies
    read_CN - read the results for CN studies
"""

import re
import numpy as np
import pandas as pd


def proportions(fn):
    """
    Compute proportions of normal and anomalous cases in the file
    
    fn is the filename
    
    Outputs a numpy array of floats
    """
    df = pd.read_csv(fn)
    df = df[df["real"] != -1]
    return(np.unique(df["real"], return_counts=True)[1]/df.shape)


def read_ABCNL(fn, fun):
    """
    Read the results of ABC and NL studies, compute the performance
    
    fn is the filename
    fun is the performance measure to compute
    
    Outputs the output of fun
    """
    df = pd.read_csv(fn)
    df = df[df["real"] != -1]
    return(fun(y_true=df["real"], y_score=df["score"]))


def read_CN(fn, what, fun):
    """
    Read the results of CN studies, compute the performance
    
    fn is the filename
    what is the name of a column with the results
    fun is the performance measure to compute
    
    Outputs the output of fun
    """
    df = pd.read_csv(fn)
    df = df[df["real"] != -1]
    if what not in ["plain_probs", "gini", "palma"]:
        if what not in ["n_pp", "n_re"]:
            return(
                fun(y_true=df["real"], y_score=df[what])
            )
        else:
            return(
                #n_pp and n_re measure normality
                fun(y_true=df["real"], y_score=1/(df[what]+10**-5))
            )
    else:
        return(
            fun(
                y_true=df["real"], 
                y_score=df[what].apply(
                    #plain_probs, gini and palma measure anomality
                    lambda x: float(
                        re.split("\s+", x.replace("[", "").replace(
                            "]", ""
                        ))[1]
                    )
                )
            )
        )

