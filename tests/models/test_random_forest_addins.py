# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:51:02 2018

@author: Lionel Massoulard
"""

import numpy as np

import itertools
import pytest

from aikit.models.random_forest_addins import (
    RandomForestRidge,
    RandomForestLogit,
    RandomForestClassifierTransformer,
    RandomForestRegressorTransformer,
)


def test_RandomForestRidge():
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000)

    rf_ridge = RandomForestRidge()
    rf_ridge.fit(X, y)

    yhat = rf_ridge.predict(X)
    assert yhat.shape == y.shape
    assert rf_ridge.C == 1


@pytest.mark.parametrize("C, do_svd, nodes_to_keep", list(itertools.product((1, 10), (True, False), (None, 0.9))))
def test_RandomForestRidge_with_args(C, do_svd, nodes_to_keep):

    X = np.random.randn(1000, 10)
    y = np.random.randn(1000)

    rf_ridge = RandomForestRidge(C=C, do_svd=do_svd, n_estimators=10, nodes_to_keep=nodes_to_keep)
    rf_ridge.fit(X, y)
    yhat = rf_ridge.predict(X)

    assert yhat.shape == y.shape


def test_RandomForestLogit():
    X = np.random.randn(1000, 10)
    y = 1 * (np.random.randn(1000) > 0)

    rf_ridge = RandomForestLogit()
    rf_ridge.fit(X, y)

    yhat = rf_ridge.predict(X)
    yhat_proba = rf_ridge.predict_proba(X)

    assert yhat_proba.min() >= 0
    assert yhat_proba.max() <= 1
    assert yhat_proba.shape == (1000, 2)

    assert yhat.shape == y.shape
    assert rf_ridge.C == 1

    assert list(rf_ridge.classes_) == [0, 1]


@pytest.mark.parametrize("C, do_svd, nodes_to_keep", list(itertools.product((1, 10), (True, False), (None, 0.9))))
def test_RandomForestLogit_with_args(C, do_svd, nodes_to_keep):

    X = np.random.randn(1000, 10)
    y = 1 * (np.random.randn(1000) > 0)

    rf_ridge = RandomForestLogit(C=C, do_svd=do_svd, n_estimators=10, nodes_to_keep=nodes_to_keep)
    rf_ridge.fit(X, y)

    yhat = rf_ridge.predict(X)
    yhat_proba = rf_ridge.predict_proba(X)

    assert yhat_proba.min() >= 0
    assert yhat_proba.max() <= 1
    assert yhat_proba.shape == (1000, 2)

    assert yhat.shape == y.shape
    assert rf_ridge.C == C

    assert list(rf_ridge.classes_) == [0, 1]


@pytest.mark.parametrize("do_svd, nodes_to_keep", list(itertools.product((True, False), (None, 0.9))))
def test_RandomForestClassifierTransformer(do_svd, nodes_to_keep):
    X = np.random.randn(100, 10)
    y = 1 * (np.random.randn(100) > 0)

    rf_transfo = RandomForestClassifierTransformer(
        n_estimators=10, do_svd=do_svd, svd_n_components=10, nodes_to_keep=nodes_to_keep
    )

    rf_transfo.fit(X, y)  # try to fit,
    Xres = rf_transfo.transform(X)  # try to transform

    assert Xres.shape[0] == X.shape[0]

    if do_svd:
        assert Xres.shape[1] == 10
        assert rf_transfo.get_feature_names() == ["RFNODE_SVD_%d" % i for i in range(Xres.shape[1])]
    else:
        assert rf_transfo.get_feature_names() == ["RFNODE_%d" % i for i in range(Xres.shape[1])]


def _verif_RandomForestClassifierTransformer():
    for do_svd, nodes_to_keep in itertools.product((True, False), (None, 0.9)):
        test_RandomForestClassifierTransformer(do_svd=do_svd, nodes_to_keep=nodes_to_keep)


@pytest.mark.parametrize("do_svd, nodes_to_keep", list(itertools.product((True, False), (None, 0.9))))
def test_RandomForestRegressorTransformer(do_svd, nodes_to_keep):
    X = np.random.randn(100, 10)
    y = np.random.randn(100)

    rf_transfo = RandomForestRegressorTransformer(
        n_estimators=10, do_svd=do_svd, svd_n_components=10, nodes_to_keep=nodes_to_keep
    )

    rf_transfo.fit(X, y)  # try to fit,
    Xres = rf_transfo.transform(X)  # try to transform

    assert Xres.shape[0] == X.shape[0]

    if do_svd:
        assert Xres.shape[1] == 10
        assert rf_transfo.get_feature_names() == ["RFNODE_SVD_%d" % i for i in range(Xres.shape[1])]
    else:
        assert rf_transfo.get_feature_names() == ["RFNODE_%d" % i for i in range(Xres.shape[1])]


def _verif_RandomForestRegressorTransformer():
    for do_svd, nodes_to_keep in itertools.product((True, False), (None, 0.9)):
        test_RandomForestRegressorTransformer(do_svd=do_svd, nodes_to_keep=nodes_to_keep)
