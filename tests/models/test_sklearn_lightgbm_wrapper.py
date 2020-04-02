# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 09:26:34 2020

@author: Lionel Massoulard
"""
import pytest

import itertools

import numpy as np
import pandas as pd

try:
    import lightgbm
except ImportError:
    lightgbm = None

if lightgbm is not None:
    from aikit.models.sklearn_lightgbm_wrapper import LGBMClassifier, LGBMRegressor


@pytest.mark.skipif(lightgbm is None, reason="lightgbm isn't installed")
@pytest.mark.parametrize("do_crossvalidation, third_class, classes_as_string, target_is_numpy, data_is_pandas", itertools.product([True, False],[True, False], [True, False], [True, False], [True, False]))
def test_LGBMClassifier(do_crossvalidation, third_class, classes_as_string, target_is_numpy, data_is_pandas):
    np.random.seed(123)
    X = np.random.randn(100,10)
    Xtest = np.random.randn(50,10)
    y = np.random.randint(0,2 + 1*third_class, size=100)
    if classes_as_string:
        y = np.array(["a","b","c"])[y]
        
    if data_is_pandas:
        X = pd.DataFrame(X, columns=[f"COL_{j}" for j in range(X.shape[1])])
        Xtest = pd.DataFrame(Xtest, columns=[f"COL_{j}" for j in range(X.shape[1])])
        
    if not target_is_numpy:
        y = pd.Series(y)
    
    model = LGBMClassifier(do_crossvalidation=do_crossvalidation)
    model.fit(X, y)

    yhat = model.predict(X)
    yhat_test = model.predict(Xtest)

    assert yhat.shape == y.shape
    assert yhat_test.shape == (Xtest.shape[0], )

    uy = set(y)
    assert len(set(yhat).difference(uy)) == 0
    assert len(set(yhat_test).difference(uy)) == 0

    yhat_proba = model.predict_proba(X)
    yhat_proba_test = model.predict_proba(Xtest)

    assert yhat_proba.shape == (y.shape[0], len(uy))
    assert yhat_proba.min() >= 0
    assert yhat_proba.max() <= 1
    
    assert np.abs(yhat_proba.sum(axis=1) - 1).max() <= 10**(-5)
    
    assert yhat_proba_test.shape == (Xtest.shape[0], len(uy))
    assert yhat_proba_test.min() >= 0
    assert yhat_proba_test.max() <= 1
    
    assert np.abs(yhat_proba.sum(axis=1) - 1).max() <= 10**(-5)

    assert list(model.classes_) == sorted(list(uy))


@pytest.mark.skipif(lightgbm is None, reason="lightgbm isn't installed")
@pytest.mark.parametrize("do_crossvalidation, target_is_numpy, data_is_pandas", itertools.product([True, False],[True, False], [True, False]))
def test_LGBMRegressor(do_crossvalidation, target_is_numpy, data_is_pandas):
    np.random.seed(123)

    X = np.random.randn(100,10)
    Xtest = np.random.randn(50,10)

    y = np.random.randn(100)
    
    if data_is_pandas:
        X = pd.DataFrame(X, columns=[f"COL_{j}" for j in range(X.shape[1])])
        Xtest = pd.DataFrame(Xtest, columns=[f"COL_{j}" for j in range(X.shape[1])])

    if not target_is_numpy:
        y = pd.Series(y)

    model = LGBMRegressor(do_crossvalidation=do_crossvalidation)
    model.fit(X, y)

    yhat = model.predict(X)
    yhat_test = model.predict(Xtest)

    assert yhat.shape == y.shape
    assert yhat_test.shape == (Xtest.shape[0], )
