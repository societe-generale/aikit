# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:31:13 2020

@author: LionelMassoulard
"""

import pytest
import itertools

import pandas as pd
import numpy as np

from aikit.models.ordinal import ClassifierFromRegressor, OrdinalClassifier, RegressorFromClassifier

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

@pytest.mark.parametrize("string_classes, change_order, multi_target", list(itertools.product((True, False), (True, False), (True, False))))
def test_ClassifierFromRegressor(string_classes, change_order, multi_target):
    
    model = DecisionTreeRegressor(random_state=123, max_depth=None)
    klass = ClassifierFromRegressor
    generic_classifier_testing(klass,
                               model, 
                               string_classes, 
                               change_order, 
                               multi_target)
                               
    
@pytest.mark.parametrize("string_classes, change_order, multi_target", list(itertools.product((True, False), (True, False), (True, False))))
def test_OrdinalClassifier(string_classes, change_order, multi_target):
    
    model = DecisionTreeClassifier(random_state=123, max_depth=None)
    klass = OrdinalClassifier
    generic_classifier_testing(klass,
                               model, 
                               string_classes, 
                               change_order, 
                               multi_target)
    

def generic_classifier_testing(klass,
                               model,
                               string_classes,
                               change_order,
                               multi_target):
    
    if change_order and not string_classes:
        return # this is not a valid test
    
    np.random.seed(123)
    
    X = np.random.randn(9, 2)
    y_int = np.array([2,0,1]*3)
    
    if string_classes:
        y = np.array(["a","b","c"])[y_int]
    else:
        y = y_int
                
    if change_order:
        if string_classes:
            classes = ["c", "a", "b"]
            expected_classes = classes
    else:
        if string_classes:
            classes = "auto"
            expected_classes = ["a", "b", "c"]
        else:
            classes = "auto"
            expected_classes = [0, 1, 2]

    if multi_target:
        y = np.repeat(y[:,np.newaxis], 2, axis=1)
        expected_classes = [expected_classes, expected_classes]

        if classes != "auto":
            classes = [classes, classes]
            

    classifier = klass(model, classes=classes)
    classifier.fit(X, y)
    
    proba = classifier.predict_proba(X)
    
    yhat = classifier.predict(X)
    
    assert (yhat == y).all() # DecisionTree should be able to overfit easily : so that I can at least check that 
    assert type(yhat) == type(y)
    assert yhat.dtype == y.dtype
    assert yhat.shape == y.shape

    if multi_target:

        assert isinstance(classifier.classes_, list)
        assert len(classifier.classes_) == 2
        for c, e in zip(classifier.classes_, expected_classes):
            assert list(c) == list(e)
        
        assert isinstance(proba, list)
        assert len(proba) == 2
        for j, p in enumerate(proba):
            assert p.shape == (y.shape[0] , 3)
            
            assert (classifier.classes_[j][p.argmax(axis=1)] == yhat[:,j]).all()

            assert p.min() >= 0
            assert p.max() <= 1
            
            assert not pd.isnull(p).any()
            assert np.abs(p.sum(axis=1) - 1).max() <= 0.0001

    else:
        assert list(classifier.classes_) == expected_classes
        assert proba.shape == (y.shape[0] , 3)
        assert (classifier.classes_[proba.argmax(axis=1)] == yhat).all()
        assert proba.min() >= 0
        assert proba.max() <= 1
        

        assert not pd.isnull(proba).any()
        assert np.abs(proba.sum(axis=1) - 1).max() <= 0.0001


def test_RegressorFromClassifier(multi_target):
    np.random.seed(123)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    
    if multi_target:
        y = np.concatenate((y[:, np.newaxis],y[:, np.newaxis]),axis=1)
    
    regressor = RegressorFromClassifier(DecisionTreeClassifier())
    regressor.fit(X, y)

    yhat = regressor.predict(X)
    
    assert y.shape == yhat.shape
    assert type(y) == type(yhat)
    assert y.dtype == yhat.dtype
    
    from sklearn.cluster import KMeans

    
    regressor = RegressorFromClassifier(DecisionTreeClassifier(), y_clusterer=KMeans(10, random_state=123))
    regressor.fit(X, y)

    yhat = regressor.predict(X)

    assert y.shape == yhat.shape
    assert type(y) == type(yhat)
    assert y.dtype == yhat.dtype

