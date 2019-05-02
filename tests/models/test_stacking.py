# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:49:10 2018

@author: Lionel Massoulard
"""

import pytest

import numpy as np
import pandas as pd

from sklearn.base import is_regressor, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.dummy import DummyRegressor


from aikit.models.stacking import OutSamplerTransformer, StackerClassifier, StackerRegressor


def test_OutSamplerTransformer_classifier():

    np.random.seed(123)
    X = np.random.randn(100, 10)
    y = 1 * (np.random.randn(100) > 0)

    model = OutSamplerTransformer(RandomForestClassifier(n_estimators=10, random_state=123))
    model.fit(X, y)

    p1 = model.model.predict_proba(X)
    p2 = model.transform(X)

    assert not is_classifier(model)
    assert not is_regressor(model)

    assert np.abs(p1[:, 1] - p2[:, 0]).max() <= 10 ** (-10)
    assert p2.shape == (100, 1)

    assert model.get_feature_names() == ["RandomForestClassifier__1"]

    y = np.array(["a", "b", "c"])[np.random.randint(0, 3, 100)]

    model = OutSamplerTransformer(RandomForestClassifier(n_estimators=10, random_state=123))
    model.fit(X, y)

    p1 = model.model.predict_proba(X)
    p2 = model.transform(X)

    assert p1.shape == (100, 3)
    assert p2.shape == (100, 3)

    assert np.abs(p1 - p2).max() <= 10 ** (-10)

    assert model.get_feature_names() == [
        "RandomForestClassifier__a",
        "RandomForestClassifier__b",
        "RandomForestClassifier__c",
    ]


def test_OutSampleTransformer_classifier_unbalanced():
    np.random.seed(123)
    X = np.random.randn(100, 2)
    y = np.array(["AA"] * 33 + ["BB"] * 33 + ["CC"] * 33 + ["DD"])

    model = OutSamplerTransformer(RandomForestClassifier(n_estimators=10, random_state=123))

    p3 = model.fit_transform(X, y)

    assert (p3.max(axis=1) > 0).all()


def test_OutSamplerTransformer_classifier_fit_transform():

    X = np.random.randn(100, 10)
    y = 1 * (np.random.randn(100) > 0)

    cv = KFold(n_splits=10, shuffle=True, random_state=123)

    model = OutSamplerTransformer(LogisticRegression(C=1,random_state=123), cv=cv)
    model.fit(X, y)
    y1 = model.transform(X)

    model = OutSamplerTransformer(LogisticRegression(C=1,random_state=123), cv=cv)
    y2 = model.fit_transform(X, y)

    assert np.abs(y1 - y2).flatten().max() >= 0.01  # vector should be different


def test_OutSamplerTransformer_regressor():

    np.random.seed(123)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)

    model = OutSamplerTransformer(RandomForestRegressor(n_estimators=10,random_state=123), cv=10)
    model.fit(X, y)

    y1 = model.model.predict(X)
    y2 = model.transform(X)

    assert not is_classifier(model)
    assert not is_regressor(model)

    assert np.abs(y1 - y2[:, 0]).max() <= 10 ** (-10)
    assert y2.shape == (100, 1)

    assert model.get_feature_names() == ["RandomForestRegressor__target"]


def test_OutSamplerTransformer_regressor_fit_transform():

    np.random.seed(123)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)

    cv = KFold(n_splits=10, shuffle=True, random_state=123)

    model = OutSamplerTransformer(DummyRegressor(), cv=cv)
    model.fit(X, y)
    y1 = model.transform(X)

    model = OutSamplerTransformer(DummyRegressor(), cv=cv)
    y2 = model.fit_transform(X, y)

    assert np.abs(y1 - y2).flatten().max() >= 0.01  # vector should be different


def test_approx_cross_validation_OutSamplerTransformer_regressor():

    np.random.seed(123)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)

    model = OutSamplerTransformer(RandomForestRegressor(random_state=123), cv=10)

    cv_res, yhat = model.approx_cross_validation(X, y, cv=10, method="transform", no_scoring=True)

    assert cv_res is None
    assert yhat.ndim == 2
    assert yhat.shape == (y.shape[0], 1)

    with pytest.raises(NotFittedError):
        model.transform(X)

    cv = KFold(n_splits=10, shuffle=True, random_state=123)

    model = OutSamplerTransformer(DummyRegressor(), cv=cv)
    yhat1 = model.fit_transform(X, y)

    cv_res, yhat2 = model.approx_cross_validation(X, y, cv=cv, method="transform", no_scoring=True, return_predict=True)
    # Approx cross val and fit transform should return the same thing here
    assert np.abs((yhat1 - yhat2).flatten()).max() <= 10 ** (-5)

    yhat3 = np.zeros((y.shape[0], 1))

    for train, test in cv.split(X, y):
        model = DummyRegressor()
        model.fit(X[train, :], y[train])

        yhat3[test, 0] = model.predict(X[test, :])

    assert np.abs((yhat1 - yhat3).flatten()).max() <= 10 ** (-5)
    assert np.abs((yhat1 - yhat2).flatten()).max() <= 10 ** (-5)


def test_approx_cross_validation_OutSamplerTransformer_classifier():

    np.random.seed(123)
    X = np.random.randn(100, 10)
    y = 1 * (np.random.randn(100) > 0)

    model = OutSamplerTransformer(RandomForestClassifier(random_state=123), cv=10)

    cv_res, yhat = model.approx_cross_validation(X, y, cv=10, method="transform", no_scoring=True)

    assert cv_res is None
    assert yhat.ndim == 2
    assert yhat.shape == (y.shape[0], 1)

    with pytest.raises(NotFittedError):
        model.transform(X)

    with pytest.raises(NotFittedError):
        model.model.predict(X)

    cv = KFold(n_splits=10, shuffle=True, random_state=123)
    model = OutSamplerTransformer(LogisticRegression(C=1,random_state=123), cv=cv)
    yhat1 = model.fit_transform(X, y)

    model = OutSamplerTransformer(LogisticRegression(C=1,random_state=123), cv=cv)
    cv_res, yhat2 = model.approx_cross_validation(X, y, cv=cv, method="transform", no_scoring=True, return_predict=True)

    # Approx cross val and fit transform should return the same thing here
    assert np.abs((yhat1 - yhat2).flatten()).max() <= 10 ** (-5)

    yhat3 = np.zeros((y.shape[0], 1))

    for train, test in cv.split(X, y):
        model = LogisticRegression()
        model.fit(X[train, :], y[train])

        yhat3[test, 0] = model.predict_proba(X[test, :])[:, 1]

    assert np.abs((yhat1 - yhat3).flatten()).max() <= 10 ** (-5)
    assert np.abs((yhat1 - yhat2).flatten()).max() <= 10 ** (-5)


def test_StackerRegressor():

    np.random.seed(123)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)

    stacker = StackerRegressor(models=[RandomForestRegressor(n_estimators=10,random_state=123), Ridge(random_state=123)], cv=10, blender=Ridge(random_state=123))

    stacker.fit(X, y)

    yhat = stacker.predict(X)

    assert yhat.ndim == 1
    assert yhat.shape[0] == X.shape[0]

    assert is_regressor(stacker)
    assert not is_classifier(stacker)

    with pytest.raises(AttributeError):
        stacker.predict_proba(X)

    with pytest.raises(AttributeError):
        stacker.classes_


def test_StackerClassifier():

    np.random.seed(123)
    X = np.random.randn(100, 10)
    y = 1 * (np.random.randn(100) > 0)

    stacker = StackerClassifier(
        models=[RandomForestClassifier(random_state=123), LogisticRegression(C=1,random_state=123)], cv=10, blender=LogisticRegression(C=1,random_state=123)
    )

    stacker.fit(X, y)

    yhat = stacker.predict(X)

    assert yhat.ndim == 1
    assert yhat.shape[0] == X.shape[0]

    assert list(set(yhat)) == [0, 1]
    assert list(stacker.classes_) == [0, 1]

    yhat_proba = stacker.predict_proba(X)

    assert yhat_proba.shape == (y.shape[0], 2)

    assert not is_regressor(stacker)
    assert is_classifier(stacker)


def test_approx_cross_validation_StackerRegressor():

    np.random.seed(123)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)

    stacker = StackerRegressor(models=[RandomForestRegressor(n_estimators=10,random_state=123), Ridge(random_state=123)], cv=10, blender=Ridge(random_state=123))

    cv_res, yhat = stacker.approx_cross_validation(
        X, y, cv=10, method="predict", scoring=["neg_mean_squared_error"], return_predict=True, verbose=False
    )

    assert cv_res is not None
    assert isinstance(cv_res, pd.DataFrame)
    assert cv_res.shape[0] == 10
    assert "test_neg_mean_squared_error" in cv_res
    assert "train_neg_mean_squared_error" in cv_res

    assert yhat.ndim == 1
    assert yhat.shape[0] == y.shape[0]

    with pytest.raises(NotFittedError):
        stacker.predict(X)

    for m in stacker.models:
        with pytest.raises(NotFittedError):
            m.predict(X)


def test_approx_cross_validation_StackerClassifier():

    np.random.seed(123)
    X = np.random.randn(100, 10)
    y = 1 * (np.random.randn(100) > 0)

    stacker = StackerClassifier(
        models=[RandomForestClassifier(n_estimators=10,random_state=123), LogisticRegression(C=1,random_state=123)], cv=10, blender=LogisticRegression(C=1,random_state=123)
    )

    cv_res, yhat = stacker.approx_cross_validation(
        X, y, cv=10, method="predict_proba", scoring=["accuracy"], return_predict=True, verbose=False
    )

    assert cv_res is not None
    assert isinstance(cv_res, pd.DataFrame)
    assert cv_res.shape[0] == 10
    assert "test_accuracy" in cv_res
    assert "train_accuracy" in cv_res

    assert yhat.ndim == 2
    assert yhat.shape == (y.shape[0], 2)

    with pytest.raises(NotFittedError):
        stacker.predict(X)

    for m in stacker.models:
        with pytest.raises(NotFittedError):
            m.predict(X)


def _verif_all():
    test_OutSamplerTransformer_classifier()
    test_OutSamplerTransformer_regressor()

    test_OutSamplerTransformer_classifier_fit_transform()
    test_OutSamplerTransformer_regressor_fit_transform()

    test_approx_cross_validation_OutSamplerTransformer_regressor()
    test_approx_cross_validation_OutSamplerTransformer_classifier()

    test_StackerClassifier()
    test_StackerRegressor()

    test_approx_cross_validation_StackerClassifier()
    test_approx_cross_validation_StackerRegressor()
