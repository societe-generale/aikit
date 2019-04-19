# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 13:30:45 2018

@author: Lionel Massoulard
"""


import pandas as pd
import numpy as np

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from tests.helpers.testing_help_models import verif_model

from aikit.models.rotation_forest import (
    GroupPCADecisionTreeClassifier,
    GroupPCADecisionTreeRegressor,
    RandomRotationForestClassifier,
    RandomRotationForestRegressor,
)

from aikit.tools.data_structure_helper import DataTypes

import pytest
import itertools

# In[]

X, y = make_classification(random_state=123)
df1, df2, y1, y2 = train_test_split(X, y)

df1 = pd.DataFrame(df1, columns=["COL_%d" % i for i in range(df1.shape[1])])
df2 = pd.DataFrame(df2, columns=["COL_%d" % i for i in range(df1.shape[1])])

X, y = make_regression(random_state=123)
df1_reg, df2_reg, y1_reg, y2_reg = train_test_split(X, y)

df1_reg = pd.DataFrame(df1_reg, columns=["COL_%d" % i for i in range(df1_reg.shape[1])])
df2_reg = pd.DataFrame(df2_reg, columns=["COL_%d" % i for i in range(df1_reg.shape[1])])


# In[] : GroupPCADecisionTreeClassifier


def test_GroupPCADecisionTreeClassifier():
    klass = GroupPCADecisionTreeClassifier
    model_kwargs = {}

    np.random.seed(1)
    verif_model(
        df1,
        df2,
        y1,
        klass,
        model_kwargs,
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),  # , DataTypes.SparseArray, DataTypes.SparseDataFrame),
        is_classifier=True,
    )


pytest.mark.longtest


@pytest.mark.parametrize(
    "random_state, max_depth, criterion, pca_bootstrap",
    list(itertools.product(range(100), (None, 2, 5), ("gini", "entropy"), (True, False))),
)
def test_GroupPCADecisionTreeClassifier_with_params(random_state, max_depth, criterion, pca_bootstrap):
    klass = GroupPCADecisionTreeClassifier

    model_kwargs = {
        "max_depth": max_depth,
        "criterion": criterion,
        "pca_bootstrap": pca_bootstrap,
        "random_state": random_state,
    }

    verif_model(
        df1,
        df2,
        y1,
        klass,
        model_kwargs,
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),  # , DataTypes.SparseArray, DataTypes.SparseDataFrame),
        is_classifier=True,
    )


def verif_all_GroupPCADecisionTreeClassifier():
    for random_state, max_depth, criterion, pca_bootstrap in itertools.product(
        range(100), (None, 2, 5), ("gini", "entropy"), (True, False)
    ):
        test_GroupPCADecisionTreeClassifier_with_params(
            random_state=random_state, max_depth=max_depth, criterion=criterion, pca_bootstrap=pca_bootstrap
        )


# In[]
def test_RandomRotationForestClassifier():
    klass = RandomRotationForestClassifier
    model_kwargs = {}

    np.random.seed(2)

    verif_model(
        df1,
        df2,
        y1,
        klass,
        model_kwargs,
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),  # , DataTypes.SparseArray, DataTypes.SparseDataFrame),
        is_classifier=True,
    )


@pytest.mark.longtest
@pytest.mark.parametrize(
    "random_state, max_depth, criterion, pca_bootstrap",
    list(itertools.product(range(10), (None, 2, 5), ("gini", "entropy"), (True, False))),
)
def test_RandomRotationForestClassifier_with_params(random_state, max_depth, criterion, pca_bootstrap):

    klass = RandomRotationForestClassifier

    model_kwargs = {
        "max_depth": max_depth,
        "criterion": criterion,
        "pca_bootstrap": pca_bootstrap,
        "random_state": random_state,
    }

    verif_model(
        df1,
        df2,
        y1,
        klass,
        model_kwargs,
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),  # , DataTypes.SparseArray, DataTypes.SparseDataFrame),
        is_classifier=True,
    )


def verif_all_RandomRotationForestClassifier():
    for random_state, max_depth, criterion, pca_bootstrap in itertools.product(
        range(10), (None, 2, 5), ("gini", "entropy"), (True, False)
    ):
        test_RandomRotationForestClassifier_with_params(
            random_state=random_state, max_depth=max_depth, criterion=criterion, pca_bootstrap=pca_bootstrap
        )


# In[]


def test_GroupPCADecisionTreeRegressor():
    # klass = DecisionTreeClassifier
    klass = GroupPCADecisionTreeRegressor
    model_kwargs = {}

    np.random.seed(3)

    verif_model(
        df1_reg,
        df2_reg,
        y1_reg,
        klass,
        model_kwargs,
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),  # , DataTypes.SparseArray, DataTypes.SparseDataFrame),
        is_classifier=False,
    )


@pytest.mark.parametrize(
    "random_state, max_depth, criterion, pca_bootstrap",
    list(itertools.product(range(10), (None, 2, 5), ("mse", "mae"), (True, False))),
)
def test_GroupPCADecisionTreeRegressor_with_params(random_state, max_depth, criterion, pca_bootstrap):
    klass = GroupPCADecisionTreeRegressor

    model_kwargs = {
        "max_depth": max_depth,
        "criterion": criterion,
        "pca_bootstrap": pca_bootstrap,
        "random_state": random_state,
    }

    verif_model(
        df1,
        df2,
        y1,
        klass,
        model_kwargs,
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),  # , DataTypes.SparseArray, DataTypes.SparseDataFrame),
        is_classifier=False,
    )


def verif_all_GroupPCADecisionTreeRegressor():
    for random_state, max_depth, criterion, pca_bootstrap in itertools.product(
        range(100), (None, 2, 5), ("mse", "mae"), (True, False)
    ):
        test_GroupPCADecisionTreeRegressor_with_params(
            random_state=random_state, max_depth=max_depth, criterion=criterion, pca_bootstrap=pca_bootstrap
        )


# In[]
def test_RandomRotationForestRegressor():

    klass = RandomRotationForestRegressor

    model_kwargs = {}

    np.random.seed(4)

    verif_model(
        df1,
        df2,
        y1,
        klass,
        model_kwargs,
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),  # , DataTypes.SparseArray, DataTypes.SparseDataFrame),
        is_classifier=False,
    )


@pytest.mark.longtest
@pytest.mark.parametrize(
    "random_state, max_depth, criterion, pca_bootstrap",
    list(itertools.product(range(10), (None, 2, 5), ("mse", "mae"), (True, False))),
)
def test_RandomRotationForestRegressor_with_params(random_state, max_depth, criterion, pca_bootstrap):

    klass = RandomRotationForestRegressor

    model_kwargs = {
        "max_depth": max_depth,
        "criterion": criterion,
        "pca_bootstrap": pca_bootstrap,
        "random_state": random_state,
    }

    verif_model(
        df1,
        df2,
        y1,
        klass,
        model_kwargs,
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),  # , DataTypes.SparseArray, DataTypes.SparseDataFrame),
        is_classifier=False,
    )


def verif_all_test_RandomRotationForestRegressor():
    for random_state, max_depth, criterion, pca_bootstrap in itertools.product(
        range(100), (None, 2, 5), ("mse", "mae"), (True, False)
    ):
        test_RandomRotationForestRegressor_with_params(
            random_state=random_state, max_depth=max_depth, criterion=criterion, pca_bootstrap=pca_bootstrap
        )


# In[]
