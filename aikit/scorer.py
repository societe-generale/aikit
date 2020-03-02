# -*- coding: utf-8 -*-
"""
Created on Wed May  9 13:56:11 2018

@author: Lionel Massoulard
"""
import logging
logger = logging.getLogger(__name__)

import sklearn.metrics
from sklearn.metrics.regression import _check_reg_targets, r2_score
from sklearn.metrics import silhouette_score, calinski_harabaz_score, davies_bouldin_score

from sklearn.metrics.scorer import SCORERS, _BaseScorer, type_of_target


import numpy as np
import pandas as pd

from functools import partial

class log_loss_scorer_patched(object):
    """ Log Loss scorer, correcting a small issue in sklearn (labels not used) """

    def __init__(self):
        self._deprecation_msg = None

    def __call__(self, clf, X, y, sample_weight=None):
        y_pred = clf.predict_proba(X)
        if not hasattr(clf, "classes_"):
            raise ValueError("estimator should have a 'classes_' attribute")
        if isinstance(y_pred, list):
            # this means that this is a multi-target prediction
            all_log_losses = [
                -1.0 * sklearn.metrics.log_loss(y[:, j], y_pred[j], sample_weight=sample_weight, labels=clf.classes_[j])
                for j in range(len(y_pred))
            ]

            # Avg of all log-loss
            # TODO : we could also returns everythings
            return np.mean(all_log_losses)

        else:
            return -1.0 * sklearn.metrics.log_loss(y, y_pred, sample_weight=sample_weight, labels=clf.classes_)


class avg_roc_auc_score(object):
    """ Average Roc Auc scorer, make sklearn roc auc scorer works with multi-class """

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


class avg_average_precision(object):
    """ Average of Average Precision, make sklearn average precision scorer works with multi-class """

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
        return sklearn.metrics.average_precision_score(
            y2[:, classes_present], y_pred[:, classes_present], sample_weight=sample_weight, average=self.average
        )


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


def _cached_call(cache, estimator, method, *args, **kwargs):
    """Call estimator with method and args and kwargs."""
    # Remark : copy of sk22 code
    if cache is None:
        return getattr(estimator, method)(*args, **kwargs)

    try:
        return cache[method]
    except KeyError:
        result = getattr(estimator, method)(*args, **kwargs)
        cache[method] = result
        return result


class _CustomPredictScorer(_BaseScorer):
    def __init__(self, score_func, sign, kwargs):
        super().__init__(score_func, sign, kwargs)


    def __call__(self, estimator, X, y_true=None, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        # Remark : copy of sk22 code TODO
        return self._score(partial(_cached_call, None), estimator, X, y_true,
                           sample_weight=sample_weight)

    def _score(self, method_caller, estimator, X, y_true=None, sample_weight=None):
        
        y_pred = method_caller(estimator, "predict", X)
        
        try:
            return self._sign * self._score_func(X, y_pred, **self._kwargs)
        except Exception as e:
            logger.warning(str(e) + ": NaN will be return")
            return np.nan



def make_scorer_clustering(score_func, greater_is_better, **kwargs):
    sign = 1 if greater_is_better else -1
    return _CustomPredictScorer(score_func, sign, kwargs)


class _GroupProbaScorer(_BaseScorer):
    def __call__(self, clf, X, y, groups, sample_weight=None):
        """Evaluate predicted probabilities for X relative to y_true.

        Parameters
        ----------
        clf : object
            Trained classifier to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to clf.predict_proba.

        y : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.
        
        groups : array-like
            The groups to use for the scoring

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        y_type = type_of_target(y)
        y_pred = clf.predict_proba(X)
        if y_type == "binary":
            if y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]
            else:
                raise ValueError(
                    "got predict_proba of shape {},"
                    " but need classifier with two"
                    " classes for {} scoring".format(y_pred.shape, self._score_func.__name__)
                )
        if sample_weight is not None:
            return self._sign * self._score_func(y, y_pred, groups, sample_weight=sample_weight, **self._kwargs)
        else:
            return self._sign * self._score_func(y, y_pred, groups, **self._kwargs)

    def _factory_args(self):
        return ", needs_proba=True"


def max_proba_group_accuracy(y, y_pred, groups):
    """ group by group average of 'True' if prediction with highest probability is True """
    if y_pred.ndim != 1:
        raise ValueError("this function is for binary classification only")

    df = pd.DataFrame({"proba": y_pred, "groups": groups, "y": y})

    def _max_proba_is_true(sub_group):
        return sub_group.sort_values(by="proba", ascending=False)["y"].iloc[0]

    return df.groupby("groups").apply(_max_proba_is_true).mean()


log_r2_scorer = sklearn.metrics.make_scorer(log_r2_score)
silhouette_scorer = make_scorer_clustering(silhouette_score, metric="euclidean", greater_is_better=True)
calinski_harabaz_scorer = make_scorer_clustering(calinski_harabaz_score, greater_is_better=True)

davies_bouldin_scorer = make_scorer_clustering(davies_bouldin_score, greater_is_better=False)


SCORERS["avg_roc_auc"] = avg_roc_auc_score()
SCORERS["avg_average_precision"] = avg_average_precision()
SCORERS["log_loss_patched"] = log_loss_scorer_patched()
SCORERS["confidence_score"] = confidence_score()
SCORERS["log_r2"] = log_r2_scorer
SCORERS["silhouette"] = silhouette_scorer
SCORERS["calinski_harabaz"] = calinski_harabaz_scorer
SCORERS["davies_bouldin"] = davies_bouldin_scorer
