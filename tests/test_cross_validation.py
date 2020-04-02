# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 09:59:02 2018

@author: Lionel Massoulard
"""

import pytest

import pandas as pd
import numpy as np

import itertools
from collections import OrderedDict
import numbers

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.base import is_classifier, is_regressor, RegressorMixin, TransformerMixin, BaseEstimator

from sklearn.datasets import make_classification, make_regression
import sklearn.model_selection
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit, GroupKFold, cross_val_predict

from sklearn.model_selection._validation import _score#, _multimetric_score # TODO : fix test
from sklearn.exceptions import NotFittedError

from aikit.tools.data_structure_helper import convert_generic
from aikit.enums import DataTypes
from aikit.transformers.model_wrapper import DebugPassThrough
from aikit.pipeline import GraphPipeline

from aikit.cross_validation import (
    cross_validation,
    create_scoring,
    create_cv,
    score_from_params_clustering,
    is_clusterer,
    _score_with_group,
    _multimetric_score_with_group,
    IndexTrainTestCv,
    RandomTrainTestCv,
    SpecialGroupCV,
    _check_fit_params
)

from aikit.scorer import SCORERS, _GroupProbaScorer, max_proba_group_accuracy

# In[] : verification of sklearn behavior

def test___check_fit_params():
    X= np.zeros((20,2))
    train = np.arange(10)
    fit_params = {"param":"value"}
    r1 = _check_fit_params(X, fit_params, train)
    assert r1 == {'param': 'value'}
    
    
    r2 = _check_fit_params(X, {"weight":np.arange(20),"value":"param"}, train)
    assert r2.keys() == {"weight","value"}
    assert (r2["weight"] == np.arange(10)).all()
    assert r2["value"] == "param"


def test_is_classifier_is_regressor_is_clusterer():
    """ verif behavior of is_classifier and is_regressor """
    rf_c = RandomForestClassifier(n_estimators=10, random_state=123)
    assert is_classifier(rf_c)
    assert not is_regressor(rf_c)
    assert not is_clusterer(rf_c)

    rf_r = RandomForestRegressor()
    assert not is_classifier(rf_r)
    assert is_regressor(rf_r)
    assert not is_clusterer(rf_r)

    kmeans = KMeans()
    assert not is_classifier(kmeans)
    assert not is_regressor(kmeans)
    assert is_clusterer(kmeans)

    sc = StandardScaler()
    assert not is_classifier(sc)
    assert not is_regressor(sc)
    assert not is_clusterer(sc)

    pipe_c = Pipeline([("s", StandardScaler()), ("r", RandomForestClassifier(n_estimators=10, random_state=123))])
    assert is_classifier(pipe_c)
    assert not is_regressor(pipe_c)
    assert not is_clusterer(pipe_c)

    pipe_r = Pipeline([("s", StandardScaler()), ("r", RandomForestRegressor(n_estimators=10, random_state=123))])
    assert not is_classifier(pipe_r)
    assert is_regressor(pipe_r)
    assert not is_clusterer(pipe_r)

    pipe_t = Pipeline([("s", StandardScaler()), ("r", StandardScaler())])
    assert not is_classifier(pipe_t)
    assert not is_regressor(pipe_t)
    assert not is_clusterer(pipe_t)

    pipe_cluster = Pipeline([("s", StandardScaler()), ("r", KMeans())])
    assert is_clusterer(pipe_cluster)
    assert not is_regressor(pipe_cluster)
    assert not is_classifier(pipe_cluster)


def test_fit_and_predict_transfrom():
    X, y = make_classification(n_samples=100)
    X = pd.DataFrame(X, columns=["col_%d" % i for i in range(X.shape[1])])

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

    for train, test in cv.split(X, y):

        pt = DebugPassThrough()
        predictions, _ = sklearn.model_selection._validation._fit_and_predict(
            pt, X, y, train, test, verbose=1, fit_params=None, method="transform"
        )

        assert predictions.shape[0] == test.shape[0]
        assert predictions.shape[1] == X.shape[1]

        assert type(predictions) == type(X)


def test_fit_and_predict_predict():
    X, y = make_classification(n_samples=100)
    X = pd.DataFrame(X, columns=["col_%d" % i for i in range(X.shape[1])])

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

    for train, test in cv.split(X, y):

        logit = LogisticRegression()
        predictions, _ = sklearn.model_selection._validation._fit_and_predict(
            logit, X, y, train, test, verbose=1, fit_params=None, method="predict"
        )

        assert predictions.shape[0] == test.shape[0]
        assert len(predictions.shape) == 1


def test_fit_and_predict_predict_proba():
    X, y = make_classification(n_samples=100)
    X = pd.DataFrame(X, columns=["col_%d" % i for i in range(X.shape[1])])

    y = np.array(["CL_%d" % i for i in y])

    cv = KFold(n_splits=10, shuffle=False)

    for train, test in cv.split(X, y):

        logit = LogisticRegression()
        predictions, _ = sklearn.model_selection._validation._fit_and_predict(
            logit, X, y, train, test, verbose=1, fit_params=None, method="predict_proba"
        )

        assert predictions.shape[0] == test.shape[0]
        assert predictions.shape[1] == 2


@pytest.mark.xfail
def test_cross_val_predict():
    X, y = make_classification(n_samples=100)
    X = pd.DataFrame(X, columns=["col_%d" % i for i in range(X.shape[1])])

    ii = np.arange(X.shape[0])
    np.random.seed(123)
    np.random.shuffle(ii)

    pt = DebugPassThrough()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

    Xhat = cross_val_predict(pt, X, y, cv=cv, method="transform")
    assert type(Xhat) == type(X)  # Fail : cross_val_predict change the type


# In[] cv and scoring
def test_create_scoring():
    classifier = RandomForestClassifier()
    regressor = RandomForestRegressor()

    res = create_scoring(classifier, "accuracy")
    assert isinstance(res, OrderedDict)
    assert "accuracy" in res
    for k, v in res.items():
        assert callable(v)

    res = create_scoring(regressor, "neg_mean_squared_error")
    assert isinstance(res, OrderedDict)
    assert "neg_mean_squared_error" in res
    for k, v in res.items():
        assert callable(v)

    res = create_scoring(regressor, ["neg_mean_squared_error", "neg_median_absolute_error"])
    assert isinstance(res, OrderedDict)
    assert "neg_mean_squared_error" in res
    assert "neg_median_absolute_error" in res
    for k, v in res.items():
        assert callable(v)

    res = create_scoring(regressor, res)
    assert isinstance(res, OrderedDict)
    assert "neg_mean_squared_error" in res
    assert "neg_median_absolute_error" in res
    for k, v in res.items():
        assert callable(v)
        assert type(v) == type(res[k])

    res = create_scoring(regressor, {"scorer1": "accuracy"})
    assert isinstance(res, OrderedDict)
    assert "scorer1" in res
    for k, v in res.items():
        assert callable(v)
        assert type(v) == type(res[k])

    res = create_scoring(regressor, {"scorer1": SCORERS["accuracy"]})
    assert isinstance(res, OrderedDict)
    assert "scorer1" in res
    for k, v in res.items():
        assert callable(v)
        assert type(v) == type(res[k])

    res = create_scoring(regressor, SCORERS["accuracy"])
    assert isinstance(res, OrderedDict)
    for k, v in res.items():
        assert callable(v)
        assert type(v) == type(res[k])

    res = create_scoring(regressor, None)
    assert "default_score" in res
    assert isinstance(res, OrderedDict)
    for k, v in res.items():
        assert callable(v)
        assert type(v) == type(res[k])


def test_create_cv():
    y = np.array([0] * 10 + [1] * 10)
    X = np.random.randn(20, 3)

    cv1 = create_cv(cv=10, y=y, classifier=True)
    assert cv1.__class__.__name__ == "StratifiedKFold"
    assert len(list(cv1.split(X, y))) == 10
    cv1b = create_cv(cv1)
    assert cv1b is cv1

    y2 = np.random.randn(20)
    cv2 = create_cv(cv=10, y=y2)
    assert cv2.__class__.__name__ == "KFold"
    assert len(list(cv2.split(X, y))) == 10

    class PersonalizedCV(object):
        def __init__(self):
            pass

        def split(self, X, y, groups=None):
            pass

    cv = PersonalizedCV()
    cv_res = create_cv(cv)
    assert cv is cv_res


@pytest.mark.parametrize("with_groups", [True, False])
def test_cross_validation0(with_groups):
    np.random.seed(123)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)

    if with_groups:
        groups = np.array([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25)
    else:
        groups = None

    forest = RandomForestRegressor(n_estimators=10)
    result = cross_validation(forest, X, y, groups=groups, scoring=["neg_mean_squared_error", "r2"], cv=10)

    with pytest.raises(sklearn.exceptions.NotFittedError):
        forest.predict(X)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "test_neg_mean_squared_error",
        "test_r2",
        "train_neg_mean_squared_error",
        "train_r2",
        "fit_time",
        "score_time",
        "n_test_samples",
        "fold_nb",
    ]
    assert len(result) == 10

    forest = RandomForestRegressor(n_estimators=10, random_state=123)
    result, yhat = cross_validation(
        forest, X, y, groups, scoring=["neg_mean_squared_error", "r2"], cv=10, return_predict=True
    )
    with pytest.raises(sklearn.exceptions.NotFittedError):
        forest.predict(X)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "test_neg_mean_squared_error",
        "test_r2",
        "train_neg_mean_squared_error",
        "train_r2",
        "fit_time",
        "score_time",
        "n_test_samples",
        "fold_nb",
    ]

    assert len(result) == 10
    assert yhat.shape == (100,)

    X = np.random.randn(100, 10)
    y = np.array(["A"] * 33 + ["B"] * 33 + ["C"] * 34)
    forest = RandomForestClassifier(n_estimators=10, random_state=123)

    result = cross_validation(forest, X, y, groups, scoring=["accuracy", "neg_log_loss"], cv=10)
    with pytest.raises(sklearn.exceptions.NotFittedError):
        forest.predict(X)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "test_accuracy",
        "test_neg_log_loss",
        "train_accuracy",
        "train_neg_log_loss",
        "fit_time",
        "score_time",
        "n_test_samples",
        "fold_nb",
    ]

    assert len(result) == 10

    forest = RandomForestClassifier(random_state=123, n_estimators=10)
    result, yhat = cross_validation(
        forest, X, y, groups, scoring=["accuracy", "neg_log_loss"], cv=10, return_predict=True, method="predict"
    )
    with pytest.raises(sklearn.exceptions.NotFittedError):
        forest.predict(X)

    assert yhat.shape == (100,)
    assert set(np.unique(yhat)) == set(("A", "B", "C"))

    forest = RandomForestClassifier(random_state=123, n_estimators=10)
    result, yhat = cross_validation(
        forest, X, y, groups, scoring=["accuracy", "neg_log_loss"], cv=10, return_predict=True, method="predict_proba"
    )

    with pytest.raises(sklearn.exceptions.NotFittedError):
        forest.predict(X)

    assert yhat.shape == (100, 3)
    assert isinstance(yhat, pd.DataFrame)
    assert list(yhat.columns) == ["A", "B", "C"]


class TransformerFailNoGroups(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y, groups=None):
        if groups is None:
            raise ValueError("I need a groups")

        assert X.shape[0] == groups.shape[0]
        return self

    def fit_transform(self, X, y, groups):
        if groups is None:
            raise ValueError("I need a groups")

        assert X.shape[0] == groups.shape[0]

        return X

    def transform(self, X):
        return X


def test_cross_validation_passing_of_groups():
    np.random.seed(123)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    groups = np.random.randint(0, 20, size=100)

    estimator = TransformerFailNoGroups()

    cv_res, yhat = cross_validation(estimator, X, y, groups, cv=10, no_scoring=True, return_predict=True)
    # Check that it doesn't fail : meaning the estimator has access to the groups

    assert cv_res is None
    assert (yhat == X).all()


def test_cross_validation_with_scorer_object_regressor():
    np.random.seed(123)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)

    forest = RandomForestRegressor(n_estimators=10, random_state=123)
    result1 = cross_validation(forest, X, y, scoring=SCORERS["neg_mean_absolute_error"], cv=10)
    assert result1.shape[0] == 10
    assert isinstance(result1, pd.DataFrame)

    forest = RandomForestRegressor(n_estimators=10, random_state=123)
    result2 = cross_validation(forest, X, y, scoring="neg_mean_absolute_error", cv=10)
    assert result2.shape[0] == 10
    assert isinstance(result2, pd.DataFrame)

    assert np.abs(result1.iloc[:, 0] - result2.iloc[:, 0]).max() <= 10 ** (-5)
    assert np.abs(result1.iloc[:, 1] - result2.iloc[:, 1]).max() <= 10 ** (-5)


def test_cross_validation_with_scorer_object_classifier():
    X = np.random.randn(100, 10)
    y = np.array(["A"] * 33 + ["B"] * 33 + ["C"] * 34)
    forest = RandomForestClassifier(n_estimators=10, random_state=123)

    result1 = cross_validation(forest, X, y, scoring=SCORERS["accuracy"], cv=10)
    assert result1.shape[0] == 10
    assert isinstance(result1, pd.DataFrame)

    result2 = cross_validation(forest, X, y, scoring="accuracy", cv=10)
    assert result2.shape[0] == 10
    assert isinstance(result2, pd.DataFrame)

    assert np.abs(result1.iloc[:, 0] - result2.iloc[:, 0]).max() <= 10 ** (-5)
    assert np.abs(result1.iloc[:, 1] - result2.iloc[:, 1]).max() <= 10 ** (-5)

    result1 = cross_validation(forest, X, y, scoring=SCORERS["neg_log_loss"], cv=10)
    assert result1.shape[0] == 10
    assert isinstance(result1, pd.DataFrame)

    result2 = cross_validation(forest, X, y, scoring="neg_log_loss", cv=10)
    assert result2.shape[0] == 10
    assert isinstance(result2, pd.DataFrame)

    assert np.abs(result1.iloc[:, 0] - result2.iloc[:, 0]).max() <= 10 ** (-5)
    assert np.abs(result1.iloc[:, 1] - result2.iloc[:, 1]).max() <= 10 ** (-5)


# In[] : verification of approx_cross_validation function


@pytest.mark.parametrize(
    "add_third_class, x_data_type, y_string_class, shuffle, graph_pipeline, with_groups",
    list(
        itertools.product(
            (True, False),
            (DataTypes.DataFrame, DataTypes.NumpyArray, DataTypes.SparseArray),
            (True, False),
            (True, False),
            (True, False),
            (True, False),
        )
    ),
)
def test_cross_validation(add_third_class, x_data_type, y_string_class, shuffle, graph_pipeline, with_groups):

    X, y = make_classification(n_samples=100, random_state=123)
    if with_groups:
        groups = np.array([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25)
    else:
        groups = None

    X = convert_generic(X, output_type=x_data_type)
    if x_data_type == DataTypes.DataFrame:
        X.columns = ["col_%d" % i for i in range(X.shape[1])]

    if add_third_class:
        y[0:2] = 2

    if shuffle:
        np.random.seed(123)
        ii = np.arange(X.shape[0])
        np.random.shuffle(ii)
        y = y[ii]

        if isinstance(X, pd.DataFrame):
            X = X.loc[ii, :]
        else:
            X = X[ii, :]

    if y_string_class:
        y = np.array(["CL_%d" % i for i in y])

    if add_third_class:
        scoring = ["accuracy"]
    else:
        scoring = ["accuracy", "neg_log_loss"]

    if graph_pipeline:
        estimator = GraphPipeline({"pt": DebugPassThrough(), "lg": LogisticRegression()}, edges=[("pt", "lg")])
    else:
        estimator = LogisticRegression()

    ##################
    ### Only score ###
    ##################

    cv_res = cross_validation(estimator, X, y, groups, cv=10, scoring=scoring, verbose=0)

    assert isinstance(cv_res, pd.DataFrame)
    assert cv_res.shape[0] == 10
    for s in scoring:
        assert ("test_" + s) in set(cv_res.columns)
        assert ("train_" + s) in set(cv_res.columns)

    with pytest.raises(NotFittedError):
        estimator.predict(X)

    #####################
    ### Score + Proba ###
    #####################
    cv_res, yhat_proba = cross_validation(
        estimator, X, y, groups, cv=10, scoring=scoring, verbose=0, return_predict=True
    )

    assert isinstance(cv_res, pd.DataFrame)
    assert cv_res.shape[0] == 10
    for s in scoring:
        assert ("test_" + s) in set(cv_res.columns)
        assert ("train_" + s) in set(cv_res.columns)

    assert isinstance(yhat_proba, pd.DataFrame)
    if isinstance(X, pd.DataFrame):
        assert (yhat_proba.index == X.index).all()

    assert yhat_proba.shape == (y.shape[0], 2 + 1 * add_third_class)
    assert yhat_proba.min().min() >= 0
    assert yhat_proba.max().max() <= 1
    assert list(yhat_proba.columns) == list(np.sort(np.unique(y)))

    with pytest.raises(NotFittedError):
        estimator.predict(X)

    #######################
    ### Score + Predict ###
    #######################
    cv_res, yhat = cross_validation(
        estimator, X, y, groups, cv=10, scoring=scoring, verbose=0, return_predict=True, method="predict"
    )

    assert isinstance(cv_res, pd.DataFrame)
    assert cv_res.shape[0] == 10
    for s in scoring:
        assert ("test_" + s) in set(cv_res.columns)
        assert ("train_" + s) in set(cv_res.columns)

    assert yhat.ndim == 1
    assert len(np.setdiff1d(yhat, y)) == 0

    assert yhat.shape[0] == y.shape[0]

    with pytest.raises(NotFittedError):
        estimator.predict(X)

    ####################
    ### Predict only ###
    ####################
    cv_res, yhat = cross_validation(
        estimator,
        X,
        y,
        groups,
        cv=10,
        scoring=scoring,
        verbose=0,
        return_predict=True,
        method="predict",
        no_scoring=True,
    )

    assert yhat.shape[0] == y.shape[0]

    assert cv_res is None
    assert yhat.ndim == 1
    assert len(np.setdiff1d(yhat, y)) == 0

    with pytest.raises(NotFittedError):
        estimator.predict(X)


@pytest.mark.parametrize(
    "add_third_class, x_data_type, y_string_class, shuffle, graph_pipeline, with_groups",
    list(
        itertools.product(
            (True, False),
            (DataTypes.DataFrame, DataTypes.NumpyArray, DataTypes.SparseArray),
            (True, False),
            (True, False),
            (True, False),
            (True, False),
        )
    ),
)
def test_approx_cross_validation_early_stop(
    add_third_class, x_data_type, y_string_class, shuffle, graph_pipeline, with_groups
):

    X, y = make_classification(n_samples=100, random_state=123)

    if with_groups:
        groups = np.array([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25)
    else:
        groups = None

    if add_third_class:
        y[0:2] = 2

    X = convert_generic(X, output_type=x_data_type)
    if x_data_type == DataTypes.DataFrame:
        X.columns = ["col_%d" % i for i in range(X.shape[1])]

    if shuffle:
        np.random.seed(123)
        ii = np.arange(X.shape[0])
        np.random.shuffle(ii)
        y = y[ii]

        if isinstance(X, pd.DataFrame):
            X = X.loc[ii, :]
        else:
            X = X[ii, :]

    if y_string_class:
        y = np.array(["CL_%d" % i for i in y])

    if add_third_class:
        scoring = ["accuracy"]
    else:
        scoring = ["accuracy", "neg_log_loss"]

    if graph_pipeline:
        estimator = GraphPipeline(
            {"pt": DebugPassThrough(), "lg": LogisticRegression(C=1, random_state=123)}, edges=[("pt", "lg")]
        )
    else:
        estimator = LogisticRegression(C=1, random_state=123)

    cv_res, yhat = cross_validation(
        estimator,
        X,
        y,
        groups,
        cv=10,
        scoring=scoring,
        verbose=0,
        return_predict=True,
        method="predict",
        stopping_round=1,
        stopping_threshold=1.01,  # So that accuracy is sure to be bellow
    )

    assert isinstance(cv_res, pd.DataFrame)
    assert cv_res.shape[0] == 2
    for s in scoring:
        assert ("test_" + s) in set(cv_res.columns)
        assert ("train_" + s) in set(cv_res.columns)

    assert yhat is None

    cv_res, yhat = cross_validation(
        estimator,
        X,
        y,
        groups,
        cv=10,
        scoring=scoring,
        verbose=0,
        return_predict=True,
        method="predict",
        stopping_round=1,
        stopping_threshold=0.0,
    )

    assert isinstance(cv_res, pd.DataFrame)
    assert cv_res.shape[0] == 10
    for s in scoring:
        assert ("test_" + s) in set(cv_res.columns)
        assert ("train_" + s) in set(cv_res.columns)

    assert yhat.ndim == 1
    assert len(np.setdiff1d(yhat, y)) == 0


@pytest.mark.parametrize(
    "x_data_type, shuffle, graph_pipeline, with_groups",
    list(
        itertools.product(
            (DataTypes.DataFrame, DataTypes.NumpyArray, DataTypes.SparseArray),
            (True, False),
            (True, False),
            (True, False),
        )
    ),
)
def test_approx_cross_validation_transformer(x_data_type, shuffle, graph_pipeline, with_groups):

    if graph_pipeline:
        estimator = GraphPipeline({"ptA": DebugPassThrough(), "ptB": DebugPassThrough()}, edges=[("ptA", "ptB")])
    else:
        estimator = DebugPassThrough()

    X, y = make_classification(n_samples=100, random_state=123)
    if with_groups:
        groups = np.array([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25)
    else:
        groups = None

    X = convert_generic(X, output_type=x_data_type)
    if x_data_type == DataTypes.DataFrame:
        X.columns = ["col_%d" % i for i in range(X.shape[1])]

    if shuffle:
        np.random.seed(123)
        ii = np.arange(X.shape[0])
        np.random.shuffle(ii)
        y = y[ii]

        if isinstance(X, pd.DataFrame):
            X = X.loc[ii, :]
        else:
            X = X[ii, :]

    scoring = ["accuracy", "neg_log_loss"]

    ##################
    ### Score only ###
    ##################
    with pytest.raises(Exception):
        cross_validation(estimator, X, y, groups, cv=10, scoring=scoring, verbose=0)
        # shouldn't work since DebugPassThrough can't be scored

    #################
    ### Transform ###
    #################
    cv_res, Xhat = cross_validation(
        estimator, X, y, groups, cv=10, scoring=scoring, verbose=0, return_predict=True, no_scoring=True
    )

    assert type(Xhat) == type(X)
    assert cv_res is None
    assert Xhat.shape == X.shape

    if isinstance(X, pd.DataFrame):
        assert (Xhat.index == X.index).all()
        assert (Xhat.columns == X.columns).all()

    if isinstance(X, pd.DataFrame):
        assert np.abs(Xhat - X).max().max() <= 10 ** (10 - 10)
    else:
        assert np.max(np.abs(Xhat - X)) <= 10 ** (-10)


def test_cross_validation_time_serie_split():
    X, y = make_classification(n_samples=100, random_state=123)

    cv = TimeSeriesSplit(n_splits=10)

    model = RandomForestClassifier(n_estimators=10, random_state=123)
    cv_res, yhat = cross_validation(model, X, y, cv=cv, return_predict=True)

    assert yhat is None  # because I can't return predictions
    assert len(cv_res) == 10
    assert isinstance(cv_res, pd.DataFrame)


def test_score_from_params_clustering():
    np.random.seed(123)
    X = np.random.randn(100, 10)

    kmeans = KMeans(n_clusters=3, random_state=123)
    result1 = score_from_params_clustering(kmeans, X, scoring=["silhouette", "davies_bouldin"])

    with pytest.raises(sklearn.exceptions.NotFittedError):
        kmeans.predict(X)

    assert isinstance(result1, pd.DataFrame)
    assert list(result1.columns) == ["test_silhouette", "test_davies_bouldin", "fit_time", "score_time"]
    assert len(result1) == 1

    kmeans = KMeans(n_clusters=3, random_state=123)
    result2, yhat = score_from_params_clustering(
        kmeans, X, scoring=["silhouette", "davies_bouldin"], return_predict=True
    )

    with pytest.raises(sklearn.exceptions.NotFittedError):
        kmeans.predict(X)

    assert isinstance(result2, pd.DataFrame)
    assert list(result2.columns) == ["test_silhouette", "test_davies_bouldin", "fit_time", "score_time"]

    assert len(result2) == 1
    assert yhat.shape == (100,)
    assert len(np.unique(yhat)) == 3

    assert np.abs(result1.iloc[:, 0] - result2.iloc[:, 0]).max() <= 10 ** (-5)
    assert np.abs(result1.iloc[:, 1] - result2.iloc[:, 1]).max() <= 10 ** (-5)


def test_score_from_params_clustering_with_scorer_object():
    X = np.random.randn(100, 10)

    kmeans = KMeans(n_clusters=3, random_state=123)
    result1 = score_from_params_clustering(kmeans, X, scoring=SCORERS["silhouette"])
    assert result1.shape[0] == 1
    assert isinstance(result1, pd.DataFrame)

    result2 = score_from_params_clustering(kmeans, X, scoring="silhouette")
    assert result2.shape[0] == 1
    assert isinstance(result2, pd.DataFrame)

    assert np.abs(result1.iloc[:, 0] - result2.iloc[:, 0]).max() <= 10 ** (-5)

    result1 = score_from_params_clustering(kmeans, X, scoring=SCORERS["calinski_harabaz"])
    assert result1.shape[0] == 1
    assert isinstance(result1, pd.DataFrame)

    result2 = score_from_params_clustering(kmeans, X, scoring="calinski_harabaz")
    assert result2.shape[0] == 1
    assert isinstance(result2, pd.DataFrame)

    assert np.abs(result1.iloc[:, 0] - result2.iloc[:, 0]).max() <= 10 ** (-5)

    result1 = score_from_params_clustering(kmeans, X, scoring=SCORERS["davies_bouldin"])
    assert result1.shape[0] == 1
    assert isinstance(result1, pd.DataFrame)

    result2 = score_from_params_clustering(kmeans, X, scoring="davies_bouldin")
    assert result2.shape[0] == 1
    assert isinstance(result2, pd.DataFrame)

    assert np.abs(result1.iloc[:, 0] - result2.iloc[:, 0]).max() <= 10 ** (-5)


@pytest.mark.parametrize(
    "x_data_type, shuffle, graph_pipeline",
    list(
        itertools.product(
            (DataTypes.DataFrame, DataTypes.NumpyArray, DataTypes.SparseArray), (True, False), (True, False)
        )
    ),
)
def test_score_from_params(x_data_type, shuffle, graph_pipeline):
    np.random.seed(123)
    X = np.random.randn(100, 10)

    X = convert_generic(X, output_type=x_data_type)

    if x_data_type == DataTypes.DataFrame:
        X.columns = ["col_%d" % i for i in range(X.shape[1])]

    if shuffle:
        ii = np.arange(X.shape[0])
        np.random.shuffle(ii)

        if isinstance(X, pd.DataFrame):
            X = X.loc[ii, :]
        else:
            X = X[ii, :]

    scoring = ["silhouette", "davies_bouldin", "calinski_harabaz"]

    if graph_pipeline:
        estimator = GraphPipeline(
            {"pt": DebugPassThrough(), "lg": KMeans(n_clusters=3, random_state=123)}, edges=[("pt", "lg")]
        )
    else:
        estimator = KMeans(n_clusters=3, random_state=123)

    ##################
    ### Only score ###
    ##################

    res = score_from_params_clustering(estimator, X, scoring=scoring, verbose=0)

    assert isinstance(res, pd.DataFrame)
    assert res.shape[0] == 1
    for s in scoring:
        assert ("test_" + s) in set(res.columns)

    with pytest.raises(NotFittedError):
        estimator.predict(X)

    ##########################
    ### Score + Prediction ###
    ##########################
    res, label = score_from_params_clustering(estimator, X, scoring=scoring, verbose=0, return_predict=True)

    assert isinstance(res, pd.DataFrame)
    assert res.shape[0] == 1
    for s in scoring:
        assert ("test_" + s) in set(res.columns)

    assert isinstance(label, np.ndarray)

    assert len(np.unique(label)) == 3

    with pytest.raises(NotFittedError):
        estimator.predict(X)

    ####################
    ### Predict only ###
    ####################
    res, label = score_from_params_clustering(
        estimator, X, scoring=scoring, verbose=0, return_predict=True, no_scoring=True
    )

    assert len(np.unique(label)) == 3
    assert res is None

    with pytest.raises(NotFittedError):
        estimator.predict(X)


class DummyModel(RegressorMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[:, 0]


class DummyModelCheckFitParams(RegressorMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y, **fit_params):
        assert "param" in fit_params
        assert fit_params["param"] == "value"

        return self

    def predict(self, X):
        return X[:, 0]

class DummyModelCheckSampleWeight(RegressorMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            assert X.shape[0] == sample_weight.shape[0] 
        return self

    def predict(self, X):
        return X[:, 0]


class DummyModelWithApprox(RegressorMixin, BaseEstimator):
    def __init__(self, check_kwargs=False):
        self.check_kwargs = check_kwargs

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[:, 0]

    def approx_cross_validation(
        self,
        X,
        y,
        groups=None,
        scoring=None,
        cv=10,
        verbose=0,
        fit_params=None,
        return_predict=False,
        method="predict",
        no_scoring=False,
        stopping_round=None,
        stopping_threshold=None,
        **kwargs
    ):

        if self.check_kwargs:
            assert "kwargs_param" in kwargs
            assert kwargs["kwargs_param"] == "kwargs_value"

        if no_scoring:
            cv_res = None
        else:
            cv_res = {"scoring": scoring}

        if return_predict:
            return cv_res, X[:, 1]
        else:
            return cv_res
        
def test_cross_validation_sample_weight():
    X, y = make_classification(n_samples=100, random_state=123)
    sample_weight = np.ones(y.shape[0])
    
    estimator = DummyModelCheckSampleWeight()
    estimator.fit(X, y, sample_weight=sample_weight)

    cv_res, yhat = cross_validation(
        estimator,
        X,
        y,
        cv=10,
        no_scoring=True,
        return_predict=True,
        method="predict",
        fit_params={"sample_weight":sample_weight}
    )
    
    # I just need to check that it works
    assert yhat.shape[0] == y.shape[0]
    
    
    estimator = DummyModelCheckSampleWeight()
    estimator.fit(X, y)

    cv_res, yhat = cross_validation(
        estimator,
        X,
        y,
        cv=10,
        no_scoring=True,
        return_predict=True,
        method="predict"
    )
    
    # I just need to check that it works
    assert yhat.shape[0] == y.shape[0]


@pytest.mark.parametrize("approximate_cv", [True, False])
def test_approx_cross_validation_fit_params(approximate_cv):
    X, y = make_classification(n_samples=100, random_state=123)

    estimator = DummyModelCheckFitParams()
    with pytest.raises(AssertionError):
        cv_res, yhat = cross_validation(
            estimator,
            X,
            y,
            cv=10,
            no_scoring=True,
            return_predict=True,
            method="predict",
            approximate_cv=approximate_cv,
        )

    cv_res, yhat = cross_validation(
        estimator,
        X,
        y,
        cv=10,
        no_scoring=True,
        return_predict=True,
        method="predict",
        fit_params={"param": "value"},
        approximate_cv=approximate_cv,
    )


def test_approx_cross_validation_pass_kwargs():
    X, y = make_classification(n_samples=100, random_state=123)

    estimator = DummyModelWithApprox(check_kwargs=True)

    with pytest.raises(AssertionError):
        cv_res, yhat = cross_validation(
            estimator,
            X,
            y,
            cv=10,
            no_scoring=True,
            return_predict=True,
            method="predict",
            fit_params={"param": "value"},
            approximate_cv=True,
        )
        # error because kwargs not passed

    cv_res, yhat = cross_validation(
        estimator,
        X,
        y,
        cv=10,
        no_scoring=True,
        return_predict=True,
        method="predict",
        fit_params={"param": "value"},
        kwargs_param="kwargs_value",
        approximate_cv=True,
    )


@pytest.mark.parametrize("approximate_cv", [True, False])
def test_approx_cross_validation_dummy(approximate_cv):

    X, y = make_classification(n_samples=100, random_state=123)

    estimator = DummyModel()
    cv_res, yhat = cross_validation(
        estimator, X, y, cv=10, no_scoring=True, return_predict=True, method="predict", approximate_cv=approximate_cv
    )

    assert yhat.ndim == 1
    assert np.abs(yhat - X[:, 0]).max() <= 10 ** (-5)

    estimator = DummyModel()
    cv_res, yhat = cross_validation(
        estimator, X, y, cv=10, no_scoring=False, return_predict=True, method="predict", approximate_cv=approximate_cv
    )

    assert yhat.ndim == 1
    assert np.abs(yhat - X[:, 0]).max() <= 10 ** (-5)


@pytest.mark.parametrize("approximate_cv", [True, False])
def test_approx_cross_validation_raise_error(approximate_cv):

    X, y = make_classification(n_samples=100, random_state=123)

    estimator = DummyModel()
    with pytest.raises(ValueError):
        cv_res, yhat = cross_validation(
            estimator,
            X,
            y,
            cv=10,
            no_scoring=True,
            return_predict=False,
            method="predict",
            approximate_cv=approximate_cv,
        )

    # no_scoring = True AND return_predict = False => Nothing to do ... error
    estimator = DummyModel()
    with pytest.raises(AttributeError):
        cv_res, yhat = cross_validation(
            estimator,
            X,
            y,
            cv=10,
            no_scoring=True,
            return_predict=True,
            method="transform",
            approximate_cv=approximate_cv,
        )


def test_approx_cross_validation_pass_to_method():
    X, y = make_classification(n_samples=100, random_state=123)

    estimator = DummyModelWithApprox()
    cv_res, yhat = cross_validation(
        estimator, X, y, cv=10, no_scoring=True, return_predict=True, method="predict", approximate_cv=True
    )

    assert cv_res is None
    assert yhat.ndim == 1
    assert np.abs(yhat - X[:, 1]).max() <= 10 ** (-5)

    estimator = DummyModelWithApprox()
    cv_res, yhat = cross_validation(
        estimator, X, y, cv=10, no_scoring=False, return_predict=True, method="predict", approximate_cv=True
    )
    assert cv_res is not None
    assert "scoring" in cv_res

    assert yhat.ndim == 1
    assert np.abs(yhat - X[:, 1]).max() <= 10 ** (-5)

    estimator = DummyModelWithApprox()
    cv_res = cross_validation(
        estimator, X, y, cv=10, no_scoring=False, return_predict=False, method="predict", approximate_cv=True
    )
    assert cv_res is not None
    assert "scoring" in cv_res

    assert yhat.ndim == 1
    assert np.abs(yhat - X[:, 1]).max() <= 10 ** (-5)

    estimator = DummyModelWithApprox()
    cv_res = cross_validation(
        estimator,
        X,
        y,
        cv=10,
        scoring=["neg_mean_squared_error"],
        no_scoring=False,
        return_predict=False,
        method="predict",
        approximate_cv=True,
    )
    assert cv_res is not None
    assert "scoring" in cv_res
    assert cv_res["scoring"] == ["neg_mean_squared_error"]


@pytest.mark.parametrize("approximate_cv", [True, False])
def test_approx_cross_validation_cv(approximate_cv):
    X, y = make_classification()

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

    estimator = DebugPassThrough()

    cv_res, yhat = cross_validation(
        estimator,
        X,
        y,
        groups=None,
        cv=cv,
        verbose=1,
        fit_params={},
        return_predict=True,
        method="transform",
        no_scoring=True,
        stopping_round=None,
        stopping_threshold=None,
        approximate_cv=approximate_cv,
    )
    assert cv_res is None
    assert yhat.ndim == 2
    assert yhat.shape == X.shape


@pytest.mark.skipif(sklearn.__version__ >= "0.21", reason="bug fixed in 0.21")
@pytest.mark.xfail
def test_cross_val_predict_sklearn_few_sample_per_classes():
    np.random.seed(123)
    X = np.random.randn(100, 2)

    y = np.array(["AA"] * 33 + ["BB"] * 33 + ["CC"] * 33 + ["DD"])

    cv = StratifiedKFold(n_splits=10)

    logit = LogisticRegression()
    yhat_proba = cross_val_predict(logit, X, y, cv=cv, method="predict_proba")

    assert (yhat_proba.max(axis=1) > 0).all()


@pytest.mark.parametrize("with_groups", [True, False])
def test_cross_validation_few_sample_per_classes(with_groups):
    np.random.seed(123)
    X = np.random.randn(100, 2)

    y = np.array(["AA"] * 33 + ["BB"] * 33 + ["CC"] * 33 + ["DD"])
    if with_groups:
        groups = np.array([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25)
    else:
        groups = None

    cv = StratifiedKFold(n_splits=10)

    logit = LogisticRegression()

    _, yhat_proba = cross_validation(logit, X, y, groups=groups, cv=cv, return_predict=True, no_scoring=True)
    assert (yhat_proba.max(axis=1) > 0).all()

    assert yhat_proba.shape == (100, 4)
    assert list(yhat_proba.columns) == ["AA", "BB", "CC", "DD"]


def test_IndexTrainTestCv():
    np.random.seed(123)
    X = np.random.randn(100, 10)

    test_index = [0, 1, 10]
    cv = IndexTrainTestCv(test_index=test_index)

    assert hasattr(cv, "split")
    assert hasattr(cv, "get_n_splits")

    assert cv.get_n_splits(X) == 1

    splits = list(cv.split(X))
    assert len(splits) == 1
    assert len(splits[0]) == 2
    train, test = splits[0]
    assert (test == test_index).all()
    assert len(np.intersect1d(train, test)) == 0
    assert (np.sort(np.union1d(train, test)) == np.arange(100)).all()


def test_RandomTrainTestCv():
    np.random.seed(123)
    X = np.random.randn(100, 10)

    cv = RandomTrainTestCv(test_size=0.1, random_state=123)

    assert hasattr(cv, "split")
    assert hasattr(cv, "get_n_splits")

    assert cv.get_n_splits(X) == 1

    splits = list(cv.split(X))
    assert len(splits) == 1
    assert len(splits[0]) == 2
    train, test = splits[0]
    assert len(test) == 10

    assert len(np.intersect1d(train, test)) == 0

    assert (np.sort(np.union1d(train, test)) == np.arange(100)).all()

    cv = RandomTrainTestCv(test_size=0.1, random_state=123)
    splits = list(cv.split(X))
    assert len(splits) == 1
    train2, test2 = splits[0]

    assert (test2 == test).all()
    assert (train2 == train).all()  # same result when seed is the the same

    cv = RandomTrainTestCv(test_size=0.1, random_state=456)
    splits = list(cv.split(X))
    assert len(splits) == 1
    train3, test3 = splits[0]

    assert not (test3 == test).all()
    assert not (train3 == train).all()  # different result when seed is the the same


def test_RandomTrainTestCv_fail_with_cross_val_predict():
    np.random.seed(123)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    
    cv = RandomTrainTestCv(test_size=0.1, random_state=123)
    
    estimator = DecisionTreeRegressor(max_depth=2, random_state=123)
    
    with pytest.raises(ValueError):
        cross_val_predict(estimator, X, y, cv=cv)
        
    res = cross_validation(estimator, X, y, cv=cv, no_scoring=True, return_predict=True)
    assert res == (None, None)
    

def test_SpecialGroupCV():
    np.random.seed(123)
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000)
    groups = np.random.randint(0, 50, size=1000)

    cv = SpecialGroupCV(KFold(n_splits=10, shuffle=True, random_state=123))

    assert hasattr(cv, "split")
    assert hasattr(cv, "get_n_splits")

    assert cv.get_n_splits(X, y, groups=groups) == 10
    splits = list(cv.split(X, y, groups=groups))
    assert len(splits) == 10

    all_index = np.zeros(X.shape[0], dtype=np.int32)
    indexes = np.arange(X.shape[0], dtype=np.int32)
    for train, test in splits:
        groups_train = groups[train]
        groups_test = groups[test]

        index_train = indexes[train]
        index_test = indexes[test]

        assert len(np.intersect1d(groups_train, groups_test)) == 0  # no groups in both
        assert len(np.intersect1d(index_train, index_test)) == 0  # no index in both

        assert np.array_equal(np.sort(np.concatenate((index_train, index_test))), indexes)  # train + test = everything

        all_index[test] += 1

    assert (all_index == 1).all()  # all things taken once and only once in test

    # check : same split if we use the same seed
    cv2 = SpecialGroupCV(KFold(n_splits=10, shuffle=True, random_state=123))
    splits2 = list(cv2.split(X, y=None, groups=groups))
    for (train1, test1), (train2, test2) in zip(splits, splits2):
        assert np.array_equal(train1, train2)
        assert np.array_equal(test1, test2)

    cv3 = SpecialGroupCV(KFold(n_splits=10, shuffle=True, random_state=456))
    splits = list(cv3.split(X, groups=groups))
    with pytest.raises(AssertionError):
        for (train1, test1), (train2, test2) in zip(splits, splits2):
            assert np.array_equal(train1, train2)
            assert np.array_equal(test1, test2)
        # some things should be different if different seed

    cv = SpecialGroupCV(KFold(n_splits=10, shuffle=True, random_state=123))
    with pytest.raises(ValueError):
        list(cv.split(X))  # doesn't work because groups isn't setted


def test__score_with_group__multimetric_score_with_group():
    roc_auc_scorer = SCORERS["roc_auc"]

    np.random.seed(123)
    X_test = np.random.randn(100, 10)
    y_test = 1 * (np.random.randn(100) > 0)
    group_test = np.array([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25)

    estimator = LogisticRegression(solver="lbfgs", random_state=123)
    estimator.fit(X_test, y_test)

    #######################################################
    ###   Test with a scorer that doesn't accept group  ###
    #######################################################
    for i in range(2):
        if i == 0:
            result1 = _score_with_group(estimator, X_test, y_test, None, roc_auc_scorer)
        else:
            result1 = _score_with_group(estimator, X_test, y_test, group_test, roc_auc_scorer)
        result2 = _score(estimator, X_test, y_test, roc_auc_scorer)

        assert not pd.isnull(result1)
        assert isinstance(result1, numbers.Number)
        assert abs(result1 - result2) <= 10 ** (-10)

    for i in range(2):
        if i == 0:
            result1 = _multimetric_score_with_group(estimator, X_test, y_test, None, {"auc": roc_auc_scorer})
        else:
            result1 = _multimetric_score_with_group(estimator, X_test, y_test, group_test, {"auc": roc_auc_scorer})
        
        #result2 = _multimetric_score(estimator, X_test, y_test, {"auc": roc_auc_scorer}) TODO : fix test

        assert isinstance(result1, dict)
        assert set(result1.keys()) == {"auc"}
        assert not pd.isnull(result1["auc"])
        assert isinstance(result1["auc"], numbers.Number)
        
        # assert abs(result1["auc"] - result2["auc"]) <= 10 ** (-10) # TODO : fix test

    ##############################################
    ### test with a scorer that accepts group  ###
    ##############################################

    max_proba_scorer = _GroupProbaScorer(score_func=max_proba_group_accuracy, sign=1, kwargs={})

    result1 = _score_with_group(estimator, X_test, y_test, group_test, max_proba_scorer)
    assert not pd.isnull(result1)
    assert isinstance(result1, numbers.Number)
    assert 0 <= result1 <= 1

    with pytest.raises(TypeError):
        result1 = _score_with_group(estimator, X_test, y_test, None, max_proba_scorer)
    # raise error because scorer expects group

    result1 = _multimetric_score_with_group(estimator, X_test, y_test, group_test, {"mp_score": max_proba_scorer})
    assert isinstance(result1, dict)
    assert set(result1.keys()) == {"mp_score"}
    r = result1["mp_score"]
    assert not pd.isnull(r)
    assert isinstance(r, numbers.Number)
    assert 0 <= r <= 1

    with pytest.raises(TypeError):
        result1 = _multimetric_score_with_group(estimator, X_test, y_test, None, {"mp_score": max_proba_scorer})
        # raise error because scorer expects group

    #######################
    ###  test with both ###
    #######################
    result1 = _multimetric_score_with_group(
        estimator, X_test, y_test, group_test, {"mp_score": max_proba_scorer, "auc": roc_auc_scorer}
    )
    assert isinstance(result1, dict)
    assert set(result1.keys()) == {"auc", "mp_score"}
    r = result1["mp_score"]
    assert not pd.isnull(r)
    assert isinstance(r, numbers.Number)
    assert 0 <= r <= 1


def test_cross_validation_with_max_proba_accuracy():
    np.random.seed(123)
    cv = GroupKFold(n_splits=4)

    max_proba_scorer = _GroupProbaScorer(score_func=max_proba_group_accuracy, sign=1, kwargs={})

    X = np.random.randn(100, 10)
    y = 1 * (np.random.randn(100) > 0)
    groups = np.array([0] * 25 + [1] * 25 + [2] * 25 + [3] * 25)

    estimator = LogisticRegression(solver="lbfgs", random_state=123)

    cv_res = cross_validation(estimator, X, y, groups, scoring=max_proba_scorer, cv=cv)

    assert isinstance(cv_res, pd.DataFrame)
    assert cv_res.shape == (4, 6)

    cv_res = cross_validation(estimator, X, y, groups, scoring={"mp_acc": max_proba_scorer}, cv=cv)

    assert isinstance(cv_res, pd.DataFrame)
    assert cv_res.shape == (4, 6)
    assert "train_mp_acc" in cv_res.columns
    assert "test_mp_acc" in cv_res.columns
    assert cv_res["train_mp_acc"].max() <= 1
    assert cv_res["train_mp_acc"].min() >= 0

    assert cv_res["test_mp_acc"].max() <= 1
    assert cv_res["test_mp_acc"].min() >= 0


@pytest.mark.parametrize(
    "add_third_class, cast_data_frame, cast_string",
    list(itertools.product([True, False], [True, False], [True, False])),
)
def test_cross_validation_classifier_multi_output(add_third_class, cast_data_frame, cast_string):

    estimator = RandomForestClassifier(n_estimators=10, random_state=123)

    X, y = make_classification(n_samples=10)
    yd2 = np.concatenate((y.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)

    if add_third_class:
        yd2[0, 1] = 2

    if cast_string:
        yd2 = yd2.astype("str").astype("object")
        yd2[:, 0] = "cl_a_" + yd2[:, 0]
        yd2[:, 1] = "cl_b_" + yd2[:, 1]

    if cast_data_frame:
        yd2 = pd.DataFrame(yd2)

    cv_res = cross_validation(estimator, X, yd2, cv=3, scoring="log_loss_patched")
    assert cv_res.shape[0] == 3
    assert isinstance(cv_res, pd.DataFrame)
    assert "test_log_loss_patched" in cv_res.columns
    assert "train_log_loss_patched" in cv_res.columns

    cv_res, yhat = cross_validation(
        estimator, X, yd2, cv=3, scoring="log_loss_patched", return_predict=True, method="predict"
    )

    assert cv_res.shape[0] == 3
    assert isinstance(cv_res, pd.DataFrame)
    assert "test_log_loss_patched" in cv_res.columns
    assert "train_log_loss_patched" in cv_res.columns
    assert isinstance(yhat, np.ndarray)
    assert yhat.shape == yd2.shape

    cv_res, yhat_proba = cross_validation(
        estimator, X, yd2, cv=3, scoring="log_loss_patched", return_predict=True, method="predict_proba"
    )

    assert cv_res.shape[0] == 3
    assert isinstance(cv_res, pd.DataFrame)
    assert "test_log_loss_patched" in cv_res.columns
    assert "train_log_loss_patched" in cv_res.columns
    assert isinstance(yhat_proba, list)
    assert len(yhat_proba) == 2
    for j, p in enumerate(yhat_proba):
        assert p.shape == (yd2.shape[0], 2 + 1 * (j == 1) * (add_third_class))
        assert (p.sum(axis=1) - 1).abs().max() <= 10 ** (-10)
        assert isinstance(p, pd.DataFrame)
        assert p.min().min() >= 0
        assert p.max().max() <= 1

        if cast_data_frame:
            assert list(p.columns) == list(np.sort(np.unique(yd2.iloc[:, j])))
        else:
            assert list(p.columns) == list(np.sort(np.unique(yd2[:, j])))


@pytest.mark.parametrize("cast_data_frame", [True, False])
def test_cross_validation_regressor_multi_output(cast_data_frame):

    estimator = RandomForestRegressor(n_estimators=10, random_state=123)

    X, y = make_regression(n_samples=10)
    yd2 = np.concatenate((y.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)

    if cast_data_frame:
        yd2 = pd.DataFrame(yd2)

    cv_res = cross_validation(estimator, X, yd2, cv=2, scoring="r2")
    assert cv_res.shape[0] == 2
    assert isinstance(cv_res, pd.DataFrame)
    assert "test_r2" in cv_res.columns
    assert "train_r2" in cv_res.columns

    cv_res, yhat = cross_validation(estimator, X, yd2, cv=2, scoring="r2", return_predict=True, method="predict")

    assert cv_res.shape[0] == 2
    assert isinstance(cv_res, pd.DataFrame)
    assert "test_r2" in cv_res.columns
    assert "train_r2" in cv_res.columns
    assert isinstance(yhat, np.ndarray)
    assert yhat.shape == yd2.shape
