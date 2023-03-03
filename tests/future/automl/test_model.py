# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:46:47 2020

@author: Lionel Massoulard
"""
import numpy as np

from aikit.ml_machine.default import get_default_pipeline

from sklearn.tree import DecisionTreeClassifier


def test_get_default_pipeline_titanic(titanic_dataset):
    df, y, _ = titanic_dataset

    model = get_default_pipeline(df, y)
    assert hasattr(model, "fit")

    model = get_default_pipeline(df.loc[:, ["sex", "age", "sibsp", "parch"]], y)
    assert hasattr(model, "fit")

    model = get_default_pipeline(df, y, final_model=DecisionTreeClassifier())
    assert isinstance(model.models["DecisionTreeClassifier"], DecisionTreeClassifier)


def test_default_pipeline_random_numeric_numpy():
    X = np.random.randn(100, 10)  # noqa
    y = np.random.randn(100)

    model = get_default_pipeline(X, 1 * (y > 0))

    assert hasattr(model, "fit")

    X[0, 0] = np.nan
    model = get_default_pipeline(X, 1 * (y > 0))
    assert hasattr(model, "fit")
