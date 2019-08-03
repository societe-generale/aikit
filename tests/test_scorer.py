# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:29:33 2019

@author: Lionel Massoulard
"""
import pytest

import math
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import log_loss


import aikit.scorer  # import this will add to the list of scorer
from aikit.scorer import (_GroupProbaScorer,
                          max_proba_group_accuracy,
                          log_loss_scorer_patched)



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


def test_avg_roc_auc_scorer_aikit():
    np.random.seed(123)
    X = np.random.randn(100, 2)

    y = np.array(["AA"] * 33 + ["BB"] * 33 + ["CC"] * 33 + ["DD"])

    cv = StratifiedKFold(n_splits=10)

    logit = LogisticRegression()

    cv_res1 = cross_val_score(logit, X, y, cv=cv, scoring="avg_roc_auc")
    assert cv_res1.shape == (10,)
    assert not pd.isnull(cv_res1).any()

    cv_res2 = cross_val_score(logit, X, y, cv=cv, scoring=aikit.scorer.avg_roc_auc_score())
    assert cv_res2.shape == (10,)
    assert not pd.isnull(cv_res2).any()

    assert np.abs(cv_res1 - cv_res2).max() <= 10 ** (-5)

    with pytest.raises(ValueError):
        cross_val_score(logit, X, y, cv=cv, scoring="roc_auc") # sklearn doesn't handle that
        
    cv_res_aikit   = cross_val_score(logit, X, 1*(y=="AA"), cv=cv, scoring="avg_roc_auc")
    cv_res_sklearn = cross_val_score(logit, X, 1*(y=="AA"), cv=cv, scoring="roc_auc")

    assert np.abs(cv_res_aikit - cv_res_sklearn).max() <= 10 **(-5)

def test_average_precision_scorer_aikit():
    np.random.seed(123)
    X = np.random.randn(100, 2)

    y = np.array(["AA"] * 33 + ["BB"] * 33 + ["CC"] * 33 + ["DD"])

    cv = StratifiedKFold(n_splits=10)

    logit = LogisticRegression()

    cv_res1 = cross_val_score(logit, X, y, cv=cv, scoring="avg_average_precision")
    assert cv_res1.shape == (10,)
    assert not pd.isnull(cv_res1).any()

    cv_res2 = cross_val_score(logit, X, y, cv=cv, scoring=aikit.scorer.avg_average_precision())
    assert cv_res2.shape == (10,)
    assert not pd.isnull(cv_res2).any()

    assert np.abs(cv_res1 - cv_res2).max() <= 10 ** (-5)

    with pytest.raises(ValueError):
        cross_val_score(logit, X, y, cv=cv, scoring="average_precision") # sklearn doesn't handle that

    
def test_log_loss_patched_multioutput():
    np.random.seed(123)
    X = np.random.randn(100, 2)

    y1 = np.array(["AA"] * 33 +  ["BB"] * 33 + ["CC"] * 33 + ["DD"])
    y2 = np.array(["aaa"] * 50+ ["bbb"] * 40 + ["ccc"]* 9 + ["ddd"])
    y2d = np.concatenate((y1[:,np.newaxis],y2[:,np.newaxis]),axis=1)
    
    clf = RandomForestClassifier(n_estimators=10,random_state=123)
    clf.fit(X,y2d)
    
    scorer = log_loss_scorer_patched()
    
    s = scorer(clf,X,y2d)
    assert isinstance(s, float) # verify that the scorer works
    
    y_pred = clf.predict_proba(X)
    s2 = -0.5*log_loss(y2d[:,0],y_pred[0]) -0.5*log_loss(y2d[:,1],y_pred[1])
    assert s == s2


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


def test_max_proba_group_accuracy():
    y = np.array([
         1,0,0,0,
         1,0,0,0,
         1,0,0,0,
         1,0,0,0])
    groups = np.array([
            0,0,0,0,
            1,1,1,1,
            2,2,2,2,
            3,3,3,3])
    
    p = np.array([
         0.25,0.1,0.1,0.1,     # max proba is True
         0.25,0.5,0.1,0.1,     # max proba is False
         0,0.1,0,0.1,          # max proba is False
         0.75,0.1, 0.2,0.1])   # max proba is True
         
    r = max_proba_group_accuracy(y,p,groups)

    assert r == 0.5
    
# In[]
    
def test__GroupProbaScorer():
    np.random.seed(123)
    X = np.random.randn(100,10)
    y = 1*(np.random.randn(100)>0)
    groups = np.array([0]*25 + [1] * 25 + [2] * 25 + [3]*25)
    
    
    logit = LogisticRegression(solver="lbfgs", random_state=123)
    logit.fit(X,y)
    
    
    scorer = _GroupProbaScorer(score_func=max_proba_group_accuracy, sign=1, kwargs={})
    
    res = scorer(logit, X, y, groups)
    
    assert isinstance(res, float)
    assert 0 <= res <= 1
    assert not pd.isnull(res)
    
    
    with pytest.raises(TypeError):
        res = scorer(logit, X, y) # should not work because group is missing
        
