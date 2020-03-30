# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:46:47 2020

@author: Lionel Massoulard
"""
import numpy as np

from aikit.ml_machine.default import get_default_pipeline
from aikit.datasets import load_titanic

from sklearn.tree import DecisionTreeClassifier



def test_get_default_pipeline():

    dfX, y , *arg = load_titanic()

    model = get_default_pipeline(dfX, y)
    model.graphviz
    assert hasattr(model, "fit")

    model = get_default_pipeline(dfX.loc[:,["sex","age","sibsp","parch"]], y)
    assert hasattr(model, "fit")

    model = get_default_pipeline(dfX, y, final_model=DecisionTreeClassifier())
    assert isinstance(model.models["DecisionTreeClassifier"], DecisionTreeClassifier)    


    X = np.random.randn(100,10)
    y = np.random.randn(100)
    
    model = get_default_pipeline(X,1*(y>0))
    
    assert hasattr(model, "fit")
    
    X[0,0] = np.nan
    model = get_default_pipeline(X, 1*(y>0))
    assert hasattr(model, "fit")
    
    
