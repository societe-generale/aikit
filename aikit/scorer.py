# -*- coding: utf-8 -*-
"""
Created on Wed May  9 13:56:11 2018

@author: Lionel Massoulard
"""

import sklearn.metrics
from sklearn.metrics.regression import _check_reg_targets, r2_score
from sklearn.metrics import silhouette_score, calinski_harabaz_score, davies_bouldin_score

from sklearn.metrics.scorer import SCORERS, _BaseScorer

import numpy as np
import pandas as pd

from aikit.pipeline import GraphPipeline
import logging

logger = logging.getLogger(__name__)


class log_loss_scorer_patched(object):
    """ Log Loss scorer, correcting a small issue in sklearn (labels not used) """

    def __init__(self):
        self._deprecation_msg = None

    def __call__(self, clf, X, y, sample_weight=None):
        y_pred = clf.predict_proba(X)
        if not hasattr(clf, "classes_"):
            raise ValueError("estimator should have a 'classes_' attribute")
        return -1.0 * sklearn.metrics.log_loss(y, y_pred, sample_weight=sample_weight, labels=clf.classes_)


class avg_roc_auc_score(object):
    """ Average Roc Auc scorer """

    def __init__(self, average="macro"):
        self.average = average
        self._deprecation_msg = None

    def __call__(self, clf, X, y, sample_weight=None):

        y_pred = clf.predict_proba(X)
        if not hasattr(clf, "classes_"):
            raise ValueError("estimator should have a 'classes_' attribute")
        if not y_pred.shape[1] == len(clf.classes_):
            raise ValueError("estimator.classes_ isn't the same shape as predict_proba")

        y2 = np.zeros(y_pred.shape)
        for i, cl in enumerate(clf.classes_):
            y2[:, i] = 1 * (y == cl)

        classes_present = y2.sum(axis=0) > 0
        return sklearn.metrics.roc_auc_score(
            y2[:, classes_present], y_pred[:, classes_present], sample_weight=sample_weight, average=self.average
        )
        # return classes_present.sum() / len(classes_present) * score


class confidence_score(object):
    """ Mesure howmuch 'maxproba' helps discriminate between mistaken and correct instance
    If the maximum probability is high, the model is confident in its prediction otherwise the model esitates.
    We'd like error to be less present when maximum proba is high.

    roc_auc_score mesures how much 'maxproba' discrimite betweeen mistaken and correct instance

    Remark : if we note p(X,c) := Proba(Y = c | X) for a given class c
    then we have Proba( Y = Ypredict | X ) = Max(  p(X,c) for c in classes )

    """

    def __init__(self):
        self._deprecation_msg = None

    def __call__(self, clf, X, y, sample_weight=None):
        yhat = clf.predict(X)
        yhat_proba = clf.predict_proba(X)

        if isinstance(yhat_proba, pd.DataFrame):
            yhat_proba = yhat_proba.values

        yhat_maxproba = yhat_proba.max(axis=1)

        is_correct = 1 * (yhat == y)
        if (is_correct == 1).all():
            return np.nan
        else:
            return sklearn.metrics.roc_auc_score(y_true=is_correct, y_score=yhat_maxproba, sample_weight=sample_weight)


def log_r2_score(y_true, y_pred, sample_weight=None, multioutput="uniform_average"):
    """ r squared on log of prediction

    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape = (n_samples), optional
        Sample weights.

    multioutput : string in ['raw_values', 'uniform_average'] \
            or array-like of shape = (n_outputs)

        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.

        'raw_values' :
            Returns a full set of errors when the input is of multioutput
            format.

        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)

    if not (y_true >= 0).all() and not (y_pred >= 0).all():
        raise ValueError("Mean Squared Logarithmic Error cannot be used when " "targets contain negative values.")

    return r2_score(np.log(y_true + 1), np.log(y_pred + 1), sample_weight, multioutput)


class _CustomPredictScorer(_BaseScorer):
    def __init__(self, score_func, sign, kwargs):
        super().__init__(score_func, sign, kwargs)

    def __call__(self, estimator, X):
        """
        Unsupervised evaluation metric for cluster analysis results which
        mesures the quality of the model itself.
        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring.
            Must have a labels_ attribute
        X : array-like or sparse matrix
            data that will be fed to score function
        Returns
        -------
        score : float
            Score function applied to labels cluster from
            the estimator fitted on X.
        """
        if isinstance(estimator, GraphPipeline):
            terminal_node = estimator._terminal_node
            y_pred = estimator.models[terminal_node].labels_
        else:
            y_pred = estimator.labels_

        try:
            return self._sign * self._score_func(X, y_pred, **self._kwargs)
        except Exception as e:
            logger.warning(str(e) + ": NaN will be return")
            return np.nan


def make_scorer_clustering(score_func, greater_is_better, **kwargs):
    sign = 1 if greater_is_better else -1
    return _CustomPredictScorer(score_func, sign, kwargs)


log_r2_scorer = sklearn.metrics.make_scorer(log_r2_score)
silhouette_scorer = make_scorer_clustering(silhouette_score, metric="euclidean", greater_is_better=True)
calinski_harabaz_scorer = make_scorer_clustering(calinski_harabaz_score, greater_is_better=True)

davies_bouldin_scorer = make_scorer_clustering(davies_bouldin_score, greater_is_better=False)


SCORERS["avg_roc_auc"] = avg_roc_auc_score()
SCORERS["log_loss_patched"] = log_loss_scorer_patched()
SCORERS["confidence_score"] = confidence_score()
SCORERS["log_r2"] = log_r2_scorer
SCORERS["silhouette"] = silhouette_scorer
SCORERS["calinski_harabaz"] = calinski_harabaz_scorer
SCORERS["davies_bouldin"] = davies_bouldin_scorer
