# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:36:45 2019

@author: Lionel Massoulard
"""

import matplotlib.pylab as plt
import seaborn as sns

import pandas as pd
import numpy as np

from aikit.ml_machine.ml_machine_guider import transfo_quantile


def conditional_density_plot(df, var, explaining_var, f=None, ax=None):
    """ draw the conditional density of 'var' for all the modalities of 'explaining var' 
    
    Parameters
    ----------
    df : pd.DataFrame
        contains the data
        
    var : str
        the name of the variable to analyze (continuous variable)
        
    explaining_var : str
        the name of the variable to condition on (discret variable)
        
    f : None or function
        if not None will apply the function before plotting the density
        
    ax : matplotlib axes

    Returns
    -------
    ax
    """
    if ax is None:
        ax = plt.gca()

    if f is None:
        f = lambda x: x

    if df is None:
        df = pd.DataFrame({"var": var, "explaining_var": explaining_var})
        var = "var"
        explaining_var = "explaining_var"

    sns.distplot(f(df[var].values), label="--overall--", ax=ax)
    for m, sub_df in df.groupby(explaining_var):
        sns.distplot(f(sub_df[var]), label=m, ax=ax)
    ax.legend()
    ax.set_xlabel(var)

    return ax


def conditional_repartition_plot(df, var, explaining_var, ax=None, normalize=False):
    """ draw the conditional repartition plot

    Parameters
    ----------
    df : pd.DataFrame
        contains the data
        
    var : str
        the name of the variable to analyze (continuous variable)
        
    explaining_var : str
        the name of the variable to condition on (discret variable)
        
    normalize : boolean, default=False
        if True will normalize the number of observations per modalities
        
    ax : matplotlib axes
    
    
    Returns
    -------
    ax
    """

    if ax is None:
        ax = plt.gca()

    if df is None:
        df = pd.DataFrame({"var": var, "explaining_var": explaining_var})
        var = "var"
        explaining_var = "explaining_var"

    for m, sub_df in df.groupby(explaining_var):
        xs = np.arange(len(sub_df))
        if normalize:
            xs = xs / len(xs) + 1 / (2 * len(xs))
        ax.plot(xs, sub_df[var].sort_values(ascending=False).values, ".-", label=m)
    ax.legend()

    ax.set_ylabel(var)
    if normalize:
        ax.set_xlabel("perc of obs.")
    else:
        ax.set_xlabel("nb of obs.")

    return ax


def conditional_boxplot(df, var, explaining_var, nb_quantiles=10, use_rank=True, marker_size=0.1, ax=None):
    """ draw the condtional boxplot 
    Parameters
    ----------
    df : pd.DataFrame
        contains the data
        
    var : str
        the name of the variable to analyze (continuous variable)
        
    explaining_var : str
        the name of the variable to condition on (continuous)
        
    nb_quantiles: int, default=10
        the number of groups to split the 'explaining_var'
        
    use_rank: boolean, default=True
        if True will use the rank of each variable to plot
        
    ax : matplotlib axes
    
    
    Returns
    -------
    ax
    """
    if ax is None:
        ax = plt.gca()

    if df is None:
        df = pd.DataFrame({"var": var, "explaining_var": explaining_var})
        var = "var"
        explaining_var = "explaining_var"

    df_copy = pd.DataFrame({var: df[var], explaining_var: df[explaining_var]})
    if use_rank:

        var_bis = var + "__q"
        explaining_var_bis = explaining_var + "__q"

        df_copy[var_bis] = transfo_quantile(df_copy[var].values)
        df_copy[explaining_var_bis] = transfo_quantile(df_copy[explaining_var].values)
    else:
        var_bis = var
        explaining_var_bis = explaining_var

    df_copy["_quantile"] = pd.qcut(df_copy[explaining_var_bis], q=nb_quantiles)

    positions = [np.median(sub_df[explaining_var_bis].values) for _, sub_df in df_copy.groupby("_quantile")]
    if use_rank:
        real_positions = [np.median(sub_df[explaining_var].values) for _, sub_df in df_copy.groupby("_quantile")]
    else:
        real_positions = positions

    real_conditional_means = np.array([np.mean(sub_df[var].values) for _, sub_df in df_copy.groupby("_quantile")])
    quantile_means = np.array([np.mean(df[var] <= m) for m in real_conditional_means])

    widths = np.diff(positions).min() * 0.90

    ax.scatter(df_copy[explaining_var_bis], df_copy[var_bis], s=marker_size)

    for p, q in zip(positions, quantile_means):
        ax.plot([p - widths / 2, p + widths / 2], [q, q], color="g")

    ax.boxplot(
        [sub_df[var_bis].values for _, sub_df in df_copy.groupby("_quantile")],
        positions=positions,
        widths=widths,
        manage_xticks=False,
    )

    plt.xticks(positions, labels=["%2.2f" % p for p in real_positions])

    percentiles = [100 / (2 * nb_quantiles) + i * 100 / nb_quantiles for i in range(nb_quantiles)]
    vper = np.percentile(df_copy[var], percentiles)

    if use_rank:
        plt.yticks(np.array(percentiles) / 100, ["%2.2f" % p for p in vper])

    if use_rank:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        ax.set_xlim(df_copy[explaining_var_bis].min(), df_copy[explaining_var_bis].max())
        ax.set_ylim(df_copy[var_bis].min(), df_copy[var_bis].max())

    ax.set_xlabel(explaining_var)
    ax.set_ylabel(var)

    return ax
