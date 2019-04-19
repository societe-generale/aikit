# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 14:37:49 2018

@author: Lionel Massoulard
"""
import pytest
import itertools

import pandas as pd
import numpy as np

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


from aikit.models.random_forest_addins import WaveRandomForestClassifier, WaveRandomForestRegressor
from aikit.tools.data_structure_helper import DataTypes

from tests.helpers.testing_help_models import verif_model


X, y = make_classification(random_state=123, n_samples=200)
df1, df2, y1, y2 = train_test_split(X, y)

df1 = pd.DataFrame(df1, columns=["COL_%d" % i for i in range(df1.shape[1])])
df2 = pd.DataFrame(df2, columns=["COL_%d" % i for i in range(df1.shape[1])])

X, y = make_regression(random_state=123, n_samples=200)
df1_reg, df2_reg, y1_reg, y2_reg = train_test_split(X, y)

df1_reg = pd.DataFrame(df1_reg, columns=["COL_%d" % i for i in range(df1_reg.shape[1])])
df2_reg = pd.DataFrame(df2_reg, columns=["COL_%d" % i for i in range(df1_reg.shape[1])])


def test_WaveRandomForestClassifier():
    klass = WaveRandomForestClassifier

    model_kwargs = {}

    np.random.seed(4)

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
    "random_state, max_depth, criterion, nodes_to_keep",
    list(itertools.product(range(10), (None, 2, 5), ("gini", "entropy"), (None, 0.9))),
)
def test_WaveRandomForestClassifier_with_params(random_state, max_depth, criterion, nodes_to_keep):

    klass = WaveRandomForestClassifier

    model_kwargs = {
        "max_depth": max_depth,
        "criterion": criterion,
        "nodes_to_keep": nodes_to_keep,
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


def verif_all_WaveRandomForestClassifier():
    for random_state, max_depth, criterion, nodes_to_keep in itertools.product(
        range(10), (None, 2, 5), ("gini", "entropy"), (None, 0.9)
    ):
        test_WaveRandomForestClassifier_with_params(
            random_state=random_state, max_depth=max_depth, criterion=criterion, nodes_to_keep=nodes_to_keep
        )


def test_WaveRandomForestRegressor():
    klass = WaveRandomForestRegressor

    model_kwargs = {}

    np.random.seed(4)

    verif_model(
        df1_reg,
        df2_reg,
        y1_reg,
        klass,
        model_kwargs,
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),  # , DataTypes.SparseArray, DataTypes.SparseDataFrame),
        is_classifier=False,
    )


@pytest.mark.longtest
@pytest.mark.parametrize(
    "random_state, max_depth, criterion, nodes_to_keep",
    list(itertools.product(range(10), (None, 2, 5), ("mse", "mae"), (None, 0.9))),
)
def test_WaveRandomForestRegressor_with_params(random_state, max_depth, criterion, nodes_to_keep):

    klass = WaveRandomForestRegressor

    model_kwargs = {
        "max_depth": max_depth,
        "criterion": criterion,
        "nodes_to_keep": nodes_to_keep,
        "random_state": random_state,
    }

    verif_model(
        df1_reg,
        df2_reg,
        y1_reg,
        klass,
        model_kwargs,
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),  # , DataTypes.SparseArray, DataTypes.SparseDataFrame),
        is_classifier=False,
    )


def verif_all_WaveRandomForestRegressor():
    for random_state, max_depth, criterion, nodes_to_keep in itertools.product(
        range(10), (None, 2, 5), ("mse", "mae"), (None, 0.9)
    ):
        test_WaveRandomForestRegressor_with_params(
            random_state=random_state, max_depth=max_depth, criterion=criterion, nodes_to_keep=nodes_to_keep
        )
