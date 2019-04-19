# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:29:33 2019

@author: Lionel Massoulard
"""

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

import pytest

import aikit.scorer  # import this will add to the list of scorer

assert aikit.scorer  # to remove python warning 'imported but unused'


@pytest.mark.xfail
def test_log_loss_scorer_sklearn():
    np.random.seed(123)
    X = np.random.randn(100, 2)

    y = np.array(["AA"] * 33 + ["BB"] * 33 + ["CC"] * 33 + ["DD"])

    cv = StratifiedKFold(n_splits=10)

    logit = LogisticRegression()
    cv_res = cross_val_score(logit, X, y, cv=cv, scoring="neg_log_loss")  # Value Error here

    assert cv_res.shape == (10,)
    assert not pd.isnull(cv_res).any()


def test_log_loss_patched_scorer_aikit():
    np.random.seed(123)
    X = np.random.randn(100, 2)

    y = np.array(["AA"] * 33 + ["BB"] * 33 + ["CC"] * 33 + ["DD"])

    cv = StratifiedKFold(n_splits=10)

    logit = LogisticRegression()

    cv_res1 = cross_val_score(logit, X, y, cv=cv, scoring="log_loss_patched")
    assert cv_res1.shape == (10,)
    assert not pd.isnull(cv_res1).any()

    cv_res2 = cross_val_score(logit, X, y, cv=cv, scoring=aikit.scorer.log_loss_scorer_patched())
    assert cv_res2.shape == (10,)
    assert not pd.isnull(cv_res2).any()

    assert np.abs(cv_res1 - cv_res2).max() <= 10 ** (-5)


def test_log_loss_patched_same_result_sklearn():
    np.random.seed(123)
    X = np.random.randn(100, 2)

    y = np.array(["AA"] * 33 + ["BB"] * 33 + ["CC"] * 34)

    logit = LogisticRegression()
    cv = StratifiedKFold(n_splits=10)

    cv_res1 = cross_val_score(logit, X, y, cv=cv, scoring="log_loss_patched")
    cv_res2 = cross_val_score(logit, X, y, cv=cv, scoring="neg_log_loss")

    assert np.abs(cv_res1 - cv_res2).max() <= 10 ** (-5)


# check that score is between -1 and 1 and that a nan is returned if nclust==1
# or nclust == nsamples
def test_silhouette_score():
    np.random.seed(123)

    X = np.random.randn(100, 5)

    estimator1 = KMeans(n_clusters=5)
    estimator2 = KMeans(n_clusters=1)
    estimator3 = KMeans(n_clusters=100)

    estimator1.fit(X)
    estimator2.fit(X)
    estimator3.fit(X)

    score1 = aikit.scorer.silhouette_scorer(estimator1, X)
    score2 = aikit.scorer.silhouette_scorer(estimator2, X)
    score3 = aikit.scorer.silhouette_scorer(estimator3, X)

    assert (score1 >= -1) and (score1 <= 1)
    assert math.isnan(score2)
    assert math.isnan(score3)


# check that score is greater than 0 and that a nan is returned if nclust==1
# or nclust == nsamples
def test_calinski_harabaz_score():
    np.random.seed(123)

    X = np.random.randn(100, 5)

    estimator1 = KMeans(n_clusters=5)
    estimator2 = KMeans(n_clusters=1)
    estimator3 = KMeans(n_clusters=100)

    estimator1.fit(X)
    estimator2.fit(X)
    estimator3.fit(X)

    score1 = aikit.scorer.calinski_harabaz_scorer(estimator1, X)
    score2 = aikit.scorer.calinski_harabaz_scorer(estimator2, X)
    score3 = aikit.scorer.calinski_harabaz_scorer(estimator3, X)

    assert score1 >= 0
    assert math.isnan(score2)
    assert math.isnan(score3)


# check that score is smaller than 0 and that a nan is returned if nclust==1
# or nclust == nsamples
def test_davies_bouldin_score():
    np.random.seed(123)

    X = np.random.randn(100, 5)

    estimator1 = KMeans(n_clusters=5)
    estimator2 = KMeans(n_clusters=1)
    estimator3 = KMeans(n_clusters=100)

    estimator1.fit(X)
    estimator2.fit(X)
    estimator3.fit(X)

    score1 = aikit.scorer.davies_bouldin_scorer(estimator1, X)
    score2 = aikit.scorer.davies_bouldin_scorer(estimator2, X)
    score3 = aikit.scorer.davies_bouldin_scorer(estimator3, X)

    assert score1 <= 0
    assert math.isnan(score2)
    assert math.isnan(score3)
