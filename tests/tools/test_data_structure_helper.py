# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:12:07 2018

@author: Lionel Massoulard
"""

import pytest

import pandas as pd
import numpy as np
from scipy import sparse

from aikit.enums import DataTypes

from aikit.tools.data_structure_helper import (
    get_type,
    _nbcols,
    _nbrows,
    convert_to_array,
    convert_to_dataframe,
    convert_to_sparsearray,
)
from aikit.tools.data_structure_helper import make2dimensions, make1dimension
from aikit.tools.data_structure_helper import generic_hstack


def test_get_type():
    df = pd.DataFrame({"a": np.arange(10)})
    dfs = pd.SparseDataFrame({"a": [0, 0, 0, 1, 1]})

    assert get_type(df) == DataTypes.DataFrame
    assert get_type(df["a"]) == DataTypes.Serie
    assert get_type(df.values) == DataTypes.NumpyArray
    assert get_type(sparse.coo_matrix(df.values)) == DataTypes.SparseArray
    assert get_type(dfs) == DataTypes.SparseDataFrame


def test__nbcols():
    df = pd.DataFrame({"a": np.arange(10), "b": ["aa", "bb", "cc"] * 3 + ["dd"]})
    assert _nbcols(df) == 2
    assert _nbcols(df.values) == 2
    assert _nbcols(df["a"]) == 1
    assert _nbcols(df["a"].values) == 1


def test__nbrows():
    df = pd.DataFrame({"a": np.arange(10), "b": ["aa", "bb", "cc"] * 3 + ["dd"]})
    assert _nbrows(df) == 10
    assert _nbrows(df.values) == 10
    assert _nbrows(df["a"]) == 10
    assert _nbrows(df["a"].values) == 10


def test_make1dimension():
    df = pd.DataFrame({"a": np.arange(10)})
    assert make1dimension(df).shape == (10,)
    assert make1dimension(df["a"]).shape == (10,)
    assert make1dimension(df.values).shape == (10,)
    assert make1dimension(df["a"].values).shape == (10,)

    df = pd.DataFrame({"a": np.arange(10), "b": ["aa", "bb", "cc"] * 3 + ["dd"]})

    with pytest.raises(ValueError):
        make1dimension(df)  # Can't convert to one dimension if 2 columnx

    with pytest.raises(ValueError):
        make1dimension(df.values)  # Can't convert to one dimension if 2 columnx


def test_make2dimensions():
    df = pd.DataFrame({"a": np.arange(10), "b": ["aa", "bb", "cc"] * 3 + ["dd"]})
    df2 = make2dimensions(df)
    assert id(df2) == id(df)
    assert df2.shape == (10, 2)
    assert make2dimensions(df["a"]).shape == (10, 1)
    assert make2dimensions(df.values).shape == (10, 2)
    assert make2dimensions(df["a"].values).shape == (10, 1)

    xx = np.zeros((10, 2, 2))
    with pytest.raises(ValueError):
        make2dimensions(xx)


def test_conversion():

    np.random.seed(123)

    array1 = np.random.randn(10, 3)

    all_objects = {
        "a1": (array1, DataTypes.NumpyArray),
        "a2": (1 * (array1 > 0), DataTypes.NumpyArray),
        "a3": (array1[:, 1], DataTypes.NumpyArray),
        "df1": (pd.DataFrame(array1, columns=["A", "B", "C"]), DataTypes.DataFrame),
        "df2": (pd.DataFrame(1 * (array1 > 0), columns=["a", "b", "c"]), DataTypes.DataFrame),
        "s1": (sparse.csr_matrix(array1), DataTypes.SparseArray),
        "s2": (sparse.csr_matrix(1 * (array1 > 0)), DataTypes.SparseArray),
        # "dfs1":(pd.SparseDataFrame(sparse.csr_matrix(array1),columns=["A","B","C"]) , data_type.SparseDataFrame)
        # "dfs2":(pd.SparseDataFrame(sparse.csr_matrix(1*(array1 > 0)),columns=["a","b","c"]), data_type.SparseDataFrame)
    }

    for name, (obj, expected_type) in all_objects.items():
        assert get_type(obj) == expected_type

        converted = convert_to_dataframe(obj)
        assert get_type(converted) == DataTypes.DataFrame

        converted = convert_to_array(obj)
        assert get_type(converted) == DataTypes.NumpyArray

        converted = convert_to_sparsearray(obj)
        assert get_type(converted) == DataTypes.SparseArray

        # converted = convert_to_sparsedataframe(obj)
        # assert get_type(converted) == DataTypes.SparseDataFrame

    assert np.array_equal(convert_to_array(all_objects["df1"][0]), all_objects["a1"][0])
    assert np.array_equal(convert_to_array(all_objects["s1"][0]), all_objects["a1"][0])


def test_generic_hstack():
    df1 = pd.DataFrame({"a": list(range(10)), "b": ["aaaa", "bbbbb", "cccc"] * 3 + ["ezzzz"]})
    df2 = pd.DataFrame({"c": list(range(10)), "d": ["aaaa", "bbbbb", "cccc"] * 3 + ["ezzzz"]})

    df12 = generic_hstack((df1, df2))
    assert get_type(df12) == DataTypes.DataFrame
    assert df12.shape == (10, 4)
    assert list(df12.columns) == ["a", "b", "c", "d"]

    df1 = pd.DataFrame({"a": list(range(10)), "b": ["aaaa", "bbbbb", "cccc"] * 3 + ["ezzzz"]})
    df2 = pd.DataFrame(
        {"c": list(range(10)), "d": ["aaaa", "bbbbb", "cccc"] * 3 + ["ezzzz"]},
        index=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    )

    df12 = generic_hstack((df1, df2))
    assert np.array_equal(df12.index.values, np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19]))
    assert get_type(df12) == DataTypes.DataFrame
    assert df12.shape == (10, 4)
    assert list(df12.columns) == ["a", "b", "c", "d"]

    df12 = generic_hstack((df1, df2), output_type=DataTypes.NumpyArray)
    assert get_type(df12) == DataTypes.NumpyArray
    assert df12.shape == (10, 4)

    with pytest.raises(ValueError):
        generic_hstack((df1.head(3), df2.head(4)))

    with pytest.raises(ValueError):
        generic_hstack((df1.head(3).values, df2.head(4)))

    with pytest.raises(ValueError):
        generic_hstack((df1.head(3).values, df2.head(4).values))


def verif_all():
    test_make1dimension()
    test_make2dimensions()
    test_conversion()
    test_generic_hstack()
    test_get_type()
    test__nbcols()
    test__nbrows()
