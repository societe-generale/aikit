# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 10:59:59 2018

@author: Lionel Massoulard
"""

import pytest

import numpy as np
import pandas as pd
import scipy.sparse as sps

from sklearn.exceptions import NotFittedError
from sklearn.model_selection._validation import safe_indexing

from aikit.transformers.block_selector import BlockManager, BlockSelector


def test_BlockSelector():

    np.random.seed(123)

    df = pd.DataFrame({"a": np.arange(10), "b": ["aaa", "bbb", "ccc"] * 3 + ["ddd"]})
    arr = np.random.randn(df.shape[0], 5)

    input_features = {"df": df.columns, "arr": ["COL_%d" % i for i in range(arr.shape[1])]}

    # Dictionnary

    block_selector1 = BlockSelector("df")
    block_selector2 = BlockSelector("arr")

    X = {"df": df, "arr": arr}
    Xres1 = block_selector1.fit_transform(X)
    Xres2 = block_selector2.fit_transform(X)

    assert block_selector1.get_feature_names() == ["a", "b"]
    assert block_selector1.get_feature_names(input_features=input_features) == ["a", "b"]

    assert block_selector2.get_feature_names() == [0, 1, 2, 3, 4]
    assert block_selector2.get_feature_names(input_features=input_features) == [
        "COL_0",
        "COL_1",
        "COL_2",
        "COL_3",
        "COL_4",
    ]

    assert id(Xres1) == id(df)
    assert id(Xres2) == id(arr)  # no copy

    # List
    X = [df, arr]
    input_features = [df.columns, ["COL_%d" % i for i in range(arr.shape[1])]]

    block_selector1 = BlockSelector(0)
    block_selector2 = BlockSelector(1)

    Xres1 = block_selector1.fit_transform(X)
    Xres2 = block_selector2.fit_transform(X)

    assert block_selector1.get_feature_names() == ["a", "b"]
    assert block_selector1.get_feature_names(input_features=input_features) == ["a", "b"]

    assert block_selector2.get_feature_names() == [0, 1, 2, 3, 4]
    assert block_selector2.get_feature_names(input_features=input_features) == [
        "COL_0",
        "COL_1",
        "COL_2",
        "COL_3",
        "COL_4",
    ]

    assert id(Xres1) == id(df)
    assert id(Xres2) == id(arr)

    # BlockManager
    X = BlockManager({"df": df, "arr": arr})
    input_features = {"df": df.columns, "arr": ["COL_%d" % i for i in range(arr.shape[1])]}

    block_selector1 = BlockSelector("df")
    block_selector2 = BlockSelector("arr")

    X = {"df": df, "arr": arr}
    Xres1 = block_selector1.fit_transform(X)
    Xres2 = block_selector2.fit_transform(X)

    assert block_selector1.get_feature_names() == ["a", "b"]
    assert block_selector1.get_feature_names(input_features=input_features) == ["a", "b"]

    assert block_selector2.get_feature_names() == [0, 1, 2, 3, 4]
    assert block_selector2.get_feature_names(input_features=input_features) == [
        "COL_0",
        "COL_1",
        "COL_2",
        "COL_3",
        "COL_4",
    ]

    assert id(Xres1) == id(df)
    assert id(Xres2) == id(arr)  # no copy

    # Check not fitted

    block_selector1 = BlockSelector(0)
    with pytest.raises(NotFittedError):
        block_selector1.transform(X)


def test_BlockManager_retrieve():

    np.random.seed(123)

    df = pd.DataFrame({"a": np.arange(10), "b": ["aaa", "bbb", "ccc"] * 3 + ["ddd"]})
    arr = np.random.randn(df.shape[0], 5)

    X = BlockManager({"df": df, "arr": arr})

    assert X.shape == (10, 7)

    df1 = X["df"]
    arr1 = X["arr"]

    assert id(df1) == id(df)
    assert id(arr1) == id(arr)

    with pytest.raises(KeyError):
        X["toto"]

    with pytest.raises(KeyError):
        X[0]

    X = BlockManager([df, arr])
    df1 = X[0]
    arr1 = X[1]

    assert X.shape == (10, 7)

    assert id(df1) == id(df)
    assert id(arr1) == id(arr)

    with pytest.raises(KeyError):
        X["toto"]

    with pytest.raises(KeyError):
        X[3]


def test_BlockManager_subset():
    np.random.seed(123)

    df = pd.DataFrame({"a": np.arange(10), "b": ["aaa", "bbb", "ccc"] * 3 + ["ddd"]})
    arr = np.random.randn(df.shape[0], 5)

    X = BlockManager({"df": df, "arr": arr})

    Xsubset = X.iloc[0:3, :]

    assert isinstance(Xsubset, BlockManager)
    assert (Xsubset["df"] == df.iloc[0:3, :]).all().all()
    assert (Xsubset["arr"] == arr[0:3, :]).all()
    assert np.may_share_memory(Xsubset["arr"], arr)

    Xsubset = X.iloc[[0, 1, 2]]
    assert isinstance(Xsubset, BlockManager)
    assert (Xsubset["df"] == df.iloc[0:3, :]).all().all()
    assert (Xsubset["arr"] == arr[0:3, :]).all()

    Xsubset = X.iloc[np.array([0, 1, 2])]
    assert isinstance(Xsubset, BlockManager)
    assert (Xsubset["df"] == df.iloc[0:3, :]).all().all()
    assert (Xsubset["arr"] == arr[0:3, :]).all()

    #    Xsubset = safe_indexing(X, [0,1,2])
    #    assert isinstance(Xsubset, BlockManager)
    #    assert (Xsubset["df"]  == df.iloc[0:3,:]).all().all()
    #    assert (Xsubset["arr"] == arr[0:3,:]).all()

    Xsubset = safe_indexing(X, np.array([0, 1, 2]))
    assert isinstance(Xsubset, BlockManager)
    assert (Xsubset["df"] == df.iloc[0:3, :]).all().all()
    assert (Xsubset["arr"] == arr[0:3, :]).all()


def test_BlockManager_subset_with_sparse():
    np.random.seed(123)
    X = np.random.randn(100, 10)
    X[X < 0] = 0

    Xs = sps.coo_matrix(X)

    bm = BlockManager({"X": Xs})

    bm2 = bm.iloc[0:10, :]

    assert (bm2["X"].todense() == X[0:10, :]).all()
