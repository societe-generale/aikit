# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:38:05 2019

@author: Lionel Massoulard
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt


from aikit.plot import conditional_density_plot, conditional_repartition_plot, conditional_boxplot

np.random.seed(123)
x = np.random.randn(1000)
y = np.random.randn(1000)
c = np.random.randint(0,3,size=1000)

df = pd.DataFrame({"x":x,"y":y,"c":c})

def test_conditional_density_plot():

    
    ax = conditional_density_plot(df, "x","c")
    assert ax is plt.gca()
    
    ax2 = conditional_density_plot(df, "x","c",f=np.exp,ax=ax)
    assert ax2 is ax
    
    ax3 = conditional_density_plot(None, x,c)
    assert ax3 is plt.gca()
    

def test_conditional_boxplot():
    ax = conditional_boxplot(df, "x","y")
    assert ax is plt.gca()

    ax2 = conditional_boxplot(None, x,y, ax=ax)
    assert ax is ax2
    

def test_conditional_repartition_plot():

    ax = conditional_repartition_plot(df, "x","c")
    assert ax is plt.gca()
    
    ax2 = conditional_repartition_plot(None, x,c, ax=ax)
    assert ax is ax2