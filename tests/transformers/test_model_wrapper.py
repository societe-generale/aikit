# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:46:59 2018

@author: Lionel Massoulard
"""
import pytest


import pandas as pd
import numpy as np

import scipy.sparse as sps

import re
import itertools


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import TruncatedSVD

from aikit.transformers.model_wrapper import (
    ModelWrapper,
    ColumnsSelector,
    _concat,
    DebugPassThrough,
    try_to_find_features_names,
    AutoWrapper
)

from aikit.tools.db_informations import guess_type_of_variable, TypeOfVariables
from aikit.enums import DataTypes


def test_ColumnsSelector__get_list_of_columns():
    X = pd.DataFrame({"a": [0, 1, 2], "b": ["AAA", "BBB", "CCC"], "c": ["xx", "yy", "zz"], "d": [0.1, 0.2, 0.3]})

    assert ColumnsSelector._get_list_of_columns("all", X, regex_match=False) is None

    assert ColumnsSelector._get_list_of_columns(["a"], X, regex_match=False) == ["a"]
    assert ColumnsSelector._get_list_of_columns(["a", "b"], X, regex_match=False) == ["a", "b"]
    assert ColumnsSelector._get_list_of_columns([0, 1, 2], X, regex_match=False) == [0, 1, 2]

    assert ColumnsSelector._get_list_of_columns(None, X, regex_match=False) == []

    assert ColumnsSelector._get_list_of_columns("object", X, regex_match=False) == ["b", "c"]

    with pytest.raises(TypeError):
        ColumnsSelector._get_list_of_columns(X.values, "object", regex_match=False)  # error : because no DataFrame

    with pytest.raises(TypeError):
        ColumnsSelector._get_list_of_columns({"type": "not recognized"}, X, regex_match=False)

    with pytest.raises(ValueError):
        ColumnsSelector._get_list_of_columns("object", X, regex_match=True)  # error : because regex_match

    for columns in TypeOfVariables.alls:
        assert ColumnsSelector._get_list_of_columns(columns, X) == [
            c for c in X.columns if guess_type_of_variable(X[c]) == columns
        ]


def test_ColumnsSelector():

    dfX = pd.DataFrame(
        {
            "cat1": ["A", "B", "A", "D"],
            "cat2": ["toto", "tata", "truc", "toto"],
            "num1": [0, 1, 2, 3],
            "num2": [1.1, 1.5, -2, -3.5],
            "num3": [-1, 1, 25, 4],
            "text1": ["aa bb", "bb bb cc", "dd aa cc", "ee"],
            "text2": ["a z", "b e", "d t", "a b c"],
        }
    )

    dfX2 = pd.DataFrame(
        {
            "cat1": ["D", "B"],
            "cat2": ["toto", "newcat"],
            "num1": [5, 6],
            "num2": [0.1, -5.2],
            "num3": [2, -1],
            "text1": ["dd ee", "aa"],
            "text2": ["t a c", "z b"],
        }
    )

    selector = ColumnsSelector(columns_to_use=["text1", "text2"])
    r1 = dfX.loc[:, ["text1", "text2"]]
    r2 = dfX2.loc[:, ["text1", "text2"]]

    assert (selector.fit_transform(dfX) == r1).all().all()
    assert (selector.transform(dfX2) == r2).all().all()
    assert selector.get_feature_names() == ["text1", "text2"]

    selector = ColumnsSelector(columns_to_use=np.array(["text1", "text2"]))
    r1 = dfX.loc[:, ["text1", "text2"]]
    r2 = dfX2.loc[:, ["text1", "text2"]]

    assert (selector.fit_transform(dfX) == r1).all().all()
    assert (selector.transform(dfX2) == r2).all().all()
    assert selector.get_feature_names() == ["text1", "text2"]

    with pytest.raises(ValueError):
        selector.transform(dfX2.loc[:, ["text2", "text1"]])  # Error because not correct number of columns

    dfX2_cp = dfX2.copy()
    dfX2_cp["text3"] = "new_text_column"
    with pytest.raises(ValueError):
        selector.transform(dfX2_cp.loc[:, ["text3", "text1"]])  # Error because text2 not in df

    with pytest.raises(ValueError):
        selector.transform(dfX2.values)  # Error because type changes

    # This error might be ignored later

    ###  Same thing but with 'raise_if_shape_differs=False'
    selector = ColumnsSelector(columns_to_use=np.array(["text1", "text2"]), raise_if_shape_differs=False)
    r1 = dfX.loc[:, ["text1", "text2"]]
    r2 = dfX2.loc[:, ["text1", "text2"]]

    assert (selector.fit_transform(dfX) == r1).all().all()
    assert (selector.transform(dfX2) == r2).all().all()
    assert selector.get_feature_names() == ["text1", "text2"]

    r3 = selector.transform(dfX2.loc[:, ["text2", "text1"]])  # Don't raise error anymore
    assert r3.shape == r2.shape
    assert (r3 == r2).all(axis=None)

    with pytest.raises(ValueError):
        r3 = selector.transform(dfX2_cp.loc[:, ["text3", "text1"]])  # Still raise an error : because text2 isn't present

    with pytest.raises(ValueError):
        selector.transform(dfX2.values)  # Error because type changes

    selector = ColumnsSelector(columns_to_use=["text1", "text2", "text3"])
    with pytest.raises(ValueError):  # Error because 'text3' isn't present
        selector.fit(dfX)

    selector = ColumnsSelector(columns_to_use=["text1", "text2"])
    selector.fit(dfX)

    dfX3 = dfX2.copy()
    del dfX3["text1"]
    with pytest.raises(ValueError):  # Error because 'text1' is no longer present
        selector.transform(dfX3)

    dfX3 = dfX2.copy()
    dfX3.columns = ["cat1", "cat2", "num1", "num2", "num3", "textAA", "text2"]
    with pytest.raises(ValueError):
        selector.transform(dfX3)

    selector = ColumnsSelector(columns_to_use=["^text"], regex_match=True)
    r1 = dfX.loc[:, ["text1", "text2"]]
    r2 = dfX2.loc[:, ["text1", "text2"]]

    dfX3 = dfX.loc[:, ["text2", "cat1", "cat2", "num1", "num2", "num3", "text1"]].copy()

    assert (selector.fit_transform(dfX) == r1).all().all()
    assert (selector.transform(dfX2) == r2).all().all()
    assert (selector.transform(dfX3) == r1).all().all()
    assert selector.get_feature_names() == ["text1", "text2"]

    selector = ColumnsSelector(columns_to_use=[re.compile("^text")], regex_match=True)
    r1 = dfX.loc[:, ["text1", "text2"]]
    r2 = dfX2.loc[:, ["text1", "text2"]]

    dfX3 = dfX.loc[:, ["text2", "cat1", "cat2", "num1", "num2", "num3", "text1"]].copy()

    assert (selector.fit_transform(dfX) == r1).all().all()
    assert (selector.transform(dfX2) == r2).all().all()
    assert (selector.transform(dfX3) == r1).all().all()
    assert selector.get_feature_names() == ["text1", "text2"]

    selector = ColumnsSelector(columns_to_use=["^text"], regex_match=False)
    r1 = dfX.loc[:, ["text1", "text2"]]
    r2 = dfX2.loc[:, ["text1", "text2"]]
    with pytest.raises(ValueError):
        selector.fit_transform(dfX)

    selector2 = ColumnsSelector(columns_to_use=[5, 6])
    assert (selector2.fit_transform(dfX) == r1).all().all()
    assert (selector2.transform(dfX2) == r2).all().all()

    selector2b = ColumnsSelector(columns_to_use=np.array([5, 6]))
    assert (selector2b.fit_transform(dfX) == r1).all().all()
    assert (selector2b.transform(dfX2) == r2).all().all()

    with pytest.raises(ValueError):
        selector2b.transform(dfX.iloc[:, 0:-1])  # missing one column

    selector3 = ColumnsSelector(columns_to_use=[10, 5])
    with pytest.raises(ValueError):
        selector3.fit(dfX)  # Error because column 10 is not here

    selector3 = ColumnsSelector(columns_to_use=[5, 6])
    selector3.fit(dfX)
    dfX_oneless_columns = dfX.copy()
    del dfX_oneless_columns["text1"]
    with pytest.raises(ValueError):
        selector3.transform(dfX_oneless_columns)

    selector_none = ColumnsSelector(columns_to_use="all")
    assert (selector_none.fit_transform(dfX) == dfX).all().all()

    antiselector = ColumnsSelector(columns_to_drop=["cat1", "cat2"])
    assert (antiselector.fit_transform(dfX) == dfX.loc[:, ["num1", "num2", "num3", "text1", "text2"]]).all().all()
    assert antiselector.get_feature_names() == ["num1", "num2", "num3", "text1", "text2"]

    antiselector = ColumnsSelector(columns_to_drop=np.array(["cat1", "cat2"]))
    assert (antiselector.fit_transform(dfX) == dfX.loc[:, ["num1", "num2", "num3", "text1", "text2"]]).all().all()
    assert antiselector.get_feature_names() == ["num1", "num2", "num3", "text1", "text2"]

    antiselector = ColumnsSelector(columns_to_drop=["^cat"], regex_match=True)
    assert (antiselector.fit_transform(dfX) == dfX.loc[:, ["num1", "num2", "num3", "text1", "text2"]]).all().all()
    assert antiselector.get_feature_names() == ["num1", "num2", "num3", "text1", "text2"]

    cols = ["cat1", "cat2", "num1", "num2", "num3", "text1", "text2"]
    antiselector2 = ColumnsSelector(columns_to_drop=cols)
    assert antiselector2.fit_transform(dfX).shape == (4, 0)  # No column
    assert antiselector2.transform(dfX2).shape == (2, 0)
    assert antiselector2.get_feature_names() == []

    cols = [0, 1, 2, 3, 4, 5, 6]
    antiselector3 = ColumnsSelector(columns_to_drop=cols)
    assert antiselector3.fit_transform(dfX.values).shape == (4, 0)  # No column
    assert antiselector3.transform(dfX2.values).shape == (2, 0)  # No column
    assert antiselector3.get_feature_names() == []

    cols = [0, 1, 2, 3, 4, 5, 6]
    antiselector3 = ColumnsSelector(columns_to_drop=np.array(cols))
    assert antiselector3.fit_transform(dfX.values).shape == (4, 0)  # No column
    assert antiselector3.transform(dfX2.values).shape == (2, 0)  # No column
    assert antiselector3.get_feature_names() == []

    antiselector4 = ColumnsSelector(columns_to_drop="all")
    assert antiselector4.fit_transform(dfX.values).shape == (4, 0)  # No column
    assert antiselector4.transform(dfX2.values).shape == (2, 0)
    assert antiselector4.get_feature_names() == []

    antiselector5 = ColumnsSelector(columns_to_drop="all")
    assert antiselector5.fit_transform(dfX).shape == (4, 0)  # No column
    assert antiselector5.transform(dfX2).shape == (2, 0)
    assert antiselector5.get_feature_names() == []

    selector3 = ColumnsSelector(columns_to_use=["num1"])
    n1 = dfX.loc[:, ["num1"]]
    n2 = dfX2.loc[:, ["num1"]]

    #    dfX_copy = dfX.copy()
    r1 = selector3.fit_transform(dfX)
    r2 = selector3.transform(dfX2)

    assert isinstance(r1, pd.DataFrame)
    assert isinstance(r2, pd.DataFrame)

    assert (r1 == n1).all().all()
    assert (r2 == n2).all().all()

    dfrest = dfX.loc[:, ["num1", "num2", "num3", "text1", "text2"]]
    dfrest2 = dfX2.loc[:, ["num1", "num2", "num3", "text1", "text2"]]
    selector4 = ColumnsSelector(columns_to_drop=["cat1", "cat2"])

    assert (selector4.fit_transform(dfX) == dfrest).all().all()
    assert (selector4.fit_transform(dfX2) == dfrest2).all().all()

    selector5 = ColumnsSelector(columns_to_drop=[0, 1])
    assert (selector5.fit_transform(dfX) == dfrest).all().all()
    assert (selector5.fit_transform(dfX2) == dfrest2).all().all()

    selector6 = ColumnsSelector(columns_to_use=[0, 1])
    xx = np.random.randn(10, 5)
    xx2 = np.random.randn(3, 5)
    assert np.array_equal(selector6.fit_transform(xx), xx[:, 0:2])
    assert np.array_equal(selector6.fit_transform(xx2), xx2[:, 0:2])

    selector7 = ColumnsSelector(columns_to_use=["num1", "num2"])

    with pytest.raises(ValueError):
        selector7.fit(xx)

    selector_and_antiselector = ColumnsSelector(columns_to_use=["num1", "num2", "num3"], columns_to_drop=["num3"])
    assert (selector_and_antiselector.fit_transform(dfX) == dfX.loc[:, ["num1", "num2"]]).all().all()
    assert selector_and_antiselector.get_feature_names() == ["num1", "num2"]

    selector_and_antiselector2 = ColumnsSelector(columns_to_use=["num"], columns_to_drop=["3"], regex_match=True)
    assert (selector_and_antiselector2.fit_transform(dfX) == dfX.loc[:, ["num1", "num2"]]).all().all()
    assert selector_and_antiselector2.get_feature_names() == ["num1", "num2"]

    X = np.random.randn(20, 10)
    input_features = [("COL_%d" % i) for i in range(10)]
    selector = ColumnsSelector(columns_to_use=[0, 1, 5, 9])
    Xsubset = selector.fit_transform(X)

    assert (Xsubset == X[:, [0, 1, 5, 9]]).all()
    assert selector.get_feature_names() == [0, 1, 5, 9]
    assert selector.get_feature_names(input_features=input_features) == ["COL_0", "COL_1", "COL_5", "COL_9"]

    selector_with_type = ColumnsSelector(columns_to_use="object")

    r1 = dfX.loc[:, ["cat1", "cat2", "text1", "text2"]]
    r2 = dfX2.loc[:, ["cat1", "cat2", "text1", "text2"]]

    assert (selector_with_type.fit_transform(dfX) == r1).all().all()
    assert (selector_with_type.transform(dfX2) == r2).all().all()
    assert selector_with_type.get_feature_names() == ["cat1", "cat2", "text1", "text2"]

    selector_with_type = ColumnsSelector(columns_to_drop="object")

    r1 = dfX.loc[:, ["num1", "num2", "num3"]]
    r2 = dfX2.loc[:, ["num1", "num2", "num3"]]

    assert (selector_with_type.fit_transform(dfX) == r1).all().all()
    assert (selector_with_type.transform(dfX2) == r2).all().all()
    assert selector_with_type.get_feature_names() == ["num1", "num2", "num3"]

    selector = ColumnsSelector(columns_to_use="object", columns_to_drop=["text1", "text2"])
    r1 = dfX.loc[:, ["cat1", "cat2"]]
    r2 = dfX2.loc[:, ["cat1", "cat2"]]
    assert (selector.fit_transform(dfX) == r1).all().all()
    assert (selector.transform(dfX2) == r2).all().all()
    assert selector.get_feature_names() == ["cat1", "cat2"]


def test_ColumnsSelector_dataframe():
    df = pd.DataFrame(np.array([[0, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0]]), columns=["a", "b", "c"])

    # no columns
    for col in (None, []):
        selector = ColumnsSelector(columns_to_use=col)
        df1 = selector.fit_transform(df)
        assert df1.shape == (df.shape[0], 0)
        assert type(df1) == type(df)
        df1_bis = selector.transform(df)
        assert type(df1_bis) == type(df)
        assert df1_bis.shape == (df.shape[0], 0)
        assert len(selector.get_feature_names()) == df1.shape[1]

    # all columns
    selector = ColumnsSelector(columns_to_use="all")
    df1 = selector.fit_transform(df)
    assert df1.shape == df.shape
    assert type(df1) == type(df)
    assert (df1 == df).all().all()
    assert df1 is df
    df1_bis = selector.transform(df)
    assert df1_bis is df
    assert len(selector.get_feature_names()) == df1.shape[1]

    # 1 columns, str
    selector = ColumnsSelector(columns_to_use=["a"])
    df2 = selector.fit_transform(df)
    assert df2.shape == (df.shape[0], 1)
    assert type(df2) == type(df)
    assert (df2 == df.loc[:, ["a"]]).all().all()
    assert len(selector.get_feature_names()) == df2.shape[1]

    # 1 columns, int
    selector = ColumnsSelector(columns_to_use=[0])
    df2 = selector.fit_transform(df)
    assert df2.shape == (df.shape[0], 1)
    assert type(df2) == type(df)
    assert (df2 == df.loc[:, ["a"]]).all().all()
    assert len(selector.get_feature_names()) == df2.shape[1]

    # 2 columns, str
    selector = ColumnsSelector(columns_to_use=["a", "c"])
    df3 = selector.fit_transform(df)
    assert df3.shape == (df.shape[0], 2)
    assert type(df3) == type(df)
    assert (df3 == df.loc[:, ["a", "c"]]).all().all()
    assert len(selector.get_feature_names()) == df3.shape[1]

    # 2 columns, int
    selector = ColumnsSelector(columns_to_use=["a", "c"])
    df3 = selector.fit_transform(df)
    assert df3.shape == (df.shape[0], 2)
    assert type(df3) == type(df)
    assert (df3 == df.loc[:, ["a", "c"]]).all().all()
    assert len(selector.get_feature_names()) == df3.shape[1]


def test_ColumnsSelector_array():

    mat = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0]])

    # no column
    for col in (None, []):
        selector = ColumnsSelector(columns_to_use=col)
        mat1 = selector.fit_transform(mat)
        assert mat1.shape == (mat.shape[0], 0)
        assert type(mat1) == type(mat)
        mat1_bis = selector.transform(mat)
        assert type(mat1_bis) == type(mat)
        assert mat1_bis.shape == (mat.shape[0], 0)
        assert len(selector.get_feature_names()) == mat1.shape[1]

    # all columns
    selector = ColumnsSelector(columns_to_use="all")
    mat2 = selector.fit_transform(mat)

    assert mat2.shape == mat.shape
    assert type(mat2) == type(mat)
    assert (mat2 == mat).all()
    assert mat2 is mat
    mat2_bis = selector.transform(mat)
    assert mat2_bis is mat
    assert len(selector.get_feature_names()) == mat2.shape[1]

    # 1 column
    selector = ColumnsSelector(columns_to_use=[1])
    mat2 = selector.fit_transform(mat)

    assert mat2.shape == (mat.shape[0], 1)
    assert type(mat2) == type(mat)
    assert (mat[:, [1]] == mat2).all()
    assert len(selector.get_feature_names()) == mat2.shape[1]

    # 2 column
    selector = ColumnsSelector(columns_to_use=[1, 2])
    mat3 = selector.fit_transform(mat)

    assert mat3.shape == (mat.shape[0], 2)
    assert type(mat3) == type(mat)
    assert (mat[:, [1, 2]] == mat3).all()
    assert len(selector.get_feature_names()) == mat3.shape[1]


@pytest.mark.parametrize("sparse_type", [sps.csc_matrix, sps.csr_matrix, sps.coo_matrix])
def test_ColumnsSelector_sparse_matrix(sparse_type):

    mat = sparse_type([[0, 0, 0], [0, 1, 1], [0, 0, 1], [1, 0, 0]])
    # no columns
    for col in (None, []):
        selector = ColumnsSelector(columns_to_use=col)
        mat1 = selector.fit_transform(mat)
        assert mat1.shape == (mat.shape[0], 0)
        assert type(mat1) == type(mat)
        mat1_bis = selector.transform(mat)
        assert type(mat1_bis) == type(mat)
        assert mat1_bis.shape == (mat.shape[0], 0)
        assert len(selector.get_feature_names()) == mat1.shape[1]

    # all columns
    selector = ColumnsSelector(columns_to_use="all")
    mat2 = selector.fit_transform(mat)

    assert mat2.shape == mat.shape
    assert type(mat2) == type(mat)
    assert (mat.toarray() == mat2.toarray()).all()
    assert mat2 is mat
    mat2_bis = selector.transform(mat)
    assert mat2_bis is mat
    assert len(selector.get_feature_names()) == mat2.shape[1]

    # 1 column
    selector = ColumnsSelector(columns_to_use=[1])
    mat2 = selector.fit_transform(mat)

    assert mat2.shape == (mat.shape[0], 1)
    assert type(mat2) == type(mat)
    assert (mat.toarray()[:, [1]] == mat2.toarray()).all()
    assert len(selector.get_feature_names()) == mat2.shape[1]

    # 2 column
    selector = ColumnsSelector(columns_to_use=[1, 2])
    mat3 = selector.fit_transform(mat)

    assert mat3.shape == (mat.shape[0], 2)
    assert type(mat3) == type(mat)
    assert (mat.toarray()[:, [1, 2]] == mat3.toarray()).all()
    assert len(selector.get_feature_names()) == mat3.shape[1]


def test_ColumnsSelector_empty_column():

    dfX = pd.DataFrame(
        {
            "cat1": ["A", "B", "A", "D"],
            "cat2": ["toto", "tata", "truc", "toto"],
            "num1": [0, 1, 2, 3],
            "num2": [1.1, 1.5, -2, -3.5],
            "num3": [-1, 1, 25, 4],
            "text1": ["aa bb", "bb bb cc", "dd aa cc", "ee"],
            "text2": ["a z", "b e", "d t", "a b c"],
        }
    )

    dfX2 = pd.DataFrame(
        {
            "cat1": ["D", "B"],
            "cat2": ["toto", "newcat"],
            "num1": [5, 6],
            "num2": [0.1, -5.2],
            "num3": [2, -1],
            "text1": ["dd ee", "aa"],
            "text2": ["t a c", "z b"],
        }
    )

    for col in ([], None):
        selector = ColumnsSelector(columns_to_use=col)
        df_res = selector.fit_transform(dfX)

        assert df_res.shape == (dfX.shape[0], 0)
        assert isinstance(df_res, pd.DataFrame)
        assert selector.get_feature_names() == []

        df_res2 = selector.transform(dfX2)
        assert df_res2.shape == (dfX2.shape[0], 0)
        assert isinstance(df_res2, pd.DataFrame)


def test_ColumnsSelector_columns_not_present():
    dfX = pd.DataFrame(
        {
            "cat1": ["A", "B", "A", "D"],
            "cat2": ["toto", "tata", "truc", "toto"],
            "num1": [0, 1, 2, 3],
            "num2": [1.1, 1.5, -2, -3.5],
            "num3": [-1, 1, 25, 4],
            "text1": ["aa bb", "bb bb cc", "dd aa cc", "ee"],
            "text2": ["a z", "b e", "d t", "a b c"],
        }
    )

    selector = ColumnsSelector(columns_to_use=["column_isnot_present"])
    with pytest.raises(ValueError):  # error because columns is not in DataFrame
        selector.fit(dfX)


def test__concat():
    assert _concat("text1", "BAG", "toto", sep="__") == "text1__BAG__toto"
    assert _concat("text", None, "") == "text"
    assert _concat("text", None, "word1") == "text__word1"


def test_try_to_find_features_names():

    list_of_words = ["aa bb", "bb bb cc", "dd aa cc", "ee"]
    vec = CountVectorizer()
    vec.fit_transform(list_of_words)

    assert try_to_find_features_names(vec) == ["aa", "bb", "cc", "dd", "ee"]

    pipe = Pipeline([("nothing", DebugPassThrough()), ("vec", CountVectorizer())])

    pipe.fit_transform(list_of_words)

    assert try_to_find_features_names(pipe) == ["aa", "bb", "cc", "dd", "ee"]

    union = FeatureUnion(
        transformer_list=[("bagword", CountVectorizer()), ("bagchar", CountVectorizer(analyzer="char"))]
    )
    union.fit_transform(list_of_words)

    assert try_to_find_features_names(union) == [
        "bagword__aa",
        "bagword__bb",
        "bagword__cc",
        "bagword__dd",
        "bagword__ee",
        "bagchar__ ",
        "bagchar__a",
        "bagchar__b",
        "bagchar__c",
        "bagchar__d",
        "bagchar__e",
    ]

    pipe1 = Pipeline([("nothing", DebugPassThrough()), ("vec", CountVectorizer())])

    pipe2 = Pipeline([("nothing", DebugPassThrough()), ("vec", CountVectorizer(analyzer="char"))])

    union = FeatureUnion(transformer_list=[("bagword", pipe1), ("bagchar", pipe2)])
    union.fit_transform(list_of_words)

    assert try_to_find_features_names(union) == [
        "bagword__aa",
        "bagword__bb",
        "bagword__cc",
        "bagword__dd",
        "bagword__ee",
        "bagchar__ ",
        "bagchar__a",
        "bagchar__b",
        "bagchar__c",
        "bagchar__d",
        "bagchar__e",
    ]

    class DummyModelAcceptInputFeature(object):
        def get_feature_names(self, input_features=None):
            if input_features is None:
                return [0, 1, 2, 3]
            else:
                return input_features

    class DummyModelDontInputFeature(object):
        def get_feature_names(self):
            return [0, 1, 2, 3]

    class DummyModelDoesntHaveGetFeatures(object):
        pass

    m = DummyModelAcceptInputFeature()
    assert try_to_find_features_names(m) == [0, 1, 2, 3]
    assert try_to_find_features_names(m, input_features=["a", "b", "c", "d"]) == ["a", "b", "c", "d"]

    m = DummyModelDontInputFeature()
    assert try_to_find_features_names(m) == [0, 1, 2, 3]
    assert try_to_find_features_names(m, input_features=["a", "b", "c", "d"]) == [0, 1, 2, 3]

    m = DummyModelDoesntHaveGetFeatures()
    assert try_to_find_features_names(m) is None
    assert try_to_find_features_names(m, input_features=["a", "b", "c", "d"]) is None


class _DummyToWrap(BaseEstimator, TransformerMixin):
    def __init__(self, n):
        self.n = n

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.random.randn(X.shape[0], self.n)


class _DummyToWrapWithFeaturesNames(_DummyToWrap):
    def get_feature_names(self):
        return ["r%d" % i for i in range(self.n)]


class _DummyToWrapWithInputFeaturesNames(_DummyToWrap):
    def get_feature_names(self, input_features=None):
        print("input_features")
        print(input_features)
        if input_features is None:
            return ["r%d" % i for i in range(self.n)]
        else:
            return ["c_%s_%d" % (str(input_features[i]), i) for i in range(self.n)]


# def _DummyToWrapWithFeaturesNa


class DummyWrapped(ModelWrapper):
    def __init__(self, n, columns_to_use="all", column_prefix=None, drop_used_columns=True, drop_unused_columns=True):

        self.column_prefix = column_prefix
        self.columns_to_use = columns_to_use
        self.n = n

        super(DummyWrapped, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=False,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=None,
            column_prefix=column_prefix,
            desired_output_type=DataTypes.DataFrame,
            must_transform_to_get_features_name=True,
            dont_change_columns=False,
            drop_used_columns=drop_used_columns,
            drop_unused_columns=drop_unused_columns,
        )

    def _get_model(self, X, y=None):
        return _DummyToWrap(n=self.n)


class DummyWrappedWithFeaturesNames(ModelWrapper):
    def __init__(self, n, columns_to_use="all", column_prefix=None, drop_used_columns=True, drop_unused_columns=True):

        self.columns_to_use = columns_to_use

        self.n = n
        self.column_prefix = column_prefix

        super(DummyWrappedWithFeaturesNames, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=False,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=None,
            column_prefix=column_prefix,
            desired_output_type=DataTypes.DataFrame,
            must_transform_to_get_features_name=True,
            dont_change_columns=False,
            drop_used_columns=drop_used_columns,
            drop_unused_columns=drop_unused_columns,
        )

    def _get_model(self, X, y=None):
        return _DummyToWrapWithFeaturesNames(n=self.n)


class DummyWrappedWithInputFeaturesNames(ModelWrapper):
    def __init__(self, n, columns_to_use="all", column_prefix=None, drop_used_columns=True, drop_unused_columns=True):

        self.columns_to_use = columns_to_use

        self.n = n
        self.column_prefix = column_prefix

        super(DummyWrappedWithInputFeaturesNames, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=False,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=None,
            column_prefix=column_prefix,
            desired_output_type=DataTypes.DataFrame,
            must_transform_to_get_features_name=True,
            dont_change_columns=False,
            drop_used_columns=drop_used_columns,
            drop_unused_columns=drop_unused_columns,
        )

    def _get_model(self, X, y=None):
        return _DummyToWrapWithInputFeaturesNames(n=self.n)


@pytest.mark.parametrize(
    "drop_used, drop_unused, columns_to_use",
    list(itertools.product((True, False), (True, False), ("all", "object", ["num1", "num2", "num3"]))),
)
def test_ModelWrapper_drop_used_unused_columns(drop_used, drop_unused, columns_to_use):
    X = pd.DataFrame(
        {
            "cat1": ["A", "B", "A", "D"],
            "cat2": ["toto", "tata", "truc", "toto"],
            "num1": [0, 1, 2, 3],
            "num2": [1.1, 1.5, -2, -3.5],
            "num3": [-1, 1, 25, 4],
            "text1": ["aa bb", "bb bb cc", "dd aa cc", "ee"],
            "text2": ["a z", "b e", "d t", "a b c"],
        }
    )

    # Compile the output columns to expect...

    if columns_to_use == "all":
        cols_to_use = list(X.columns)
    elif columns_to_use == "object":
        cols_to_use = list(X.select_dtypes(include="object").columns)
    else:
        cols_to_use = columns_to_use

    if drop_used and drop_unused:
        expected_output_columns = []

    elif drop_used and not drop_unused:
        expected_output_columns = [c for c in list(X.columns) if c not in cols_to_use]

    elif not drop_used and drop_unused:
        expected_output_columns = cols_to_use

    else:
        expected_output_columns = list(X.columns)

    model = DummyWrappedWithFeaturesNames(
        n=2, columns_to_use=columns_to_use, drop_used_columns=drop_used, drop_unused_columns=drop_unused
    )
    df_res = model.fit_transform(X)
    assert df_res.shape[0] == X.shape[0]
    assert list(df_res.columns) == expected_output_columns + ["r0", "r1"]
    assert list(df_res.columns) == model.get_feature_names()


def test_dummy_wrapper_features():
    xx = np.random.randn(10, 5)
    input_features = ["COL_%d" % i for i in range(xx.shape[1])]
    df = pd.DataFrame(xx, columns=input_features)

    for column_prefix in (None, "RAND"):
        for i, klass in enumerate((DummyWrapped, DummyWrappedWithFeaturesNames)):

            if i == 0:
                if column_prefix is None:
                    expected = [0, 1]
                else:
                    expected = ["RAND__0", "RAND__1"]
            else:
                if column_prefix is None:
                    expected = ["r0", "r1"]
                else:
                    expected = ["RAND__r0", "RAND__r1"]
            ## On array ##
            dummy = klass(n=2, column_prefix=column_prefix)

            xxres = dummy.fit_transform(xx)
            assert dummy.get_feature_names() == expected
            assert list(xxres.columns) == expected

            dummy = klass(
                n=2,
                columns_to_use=[0, 1],
                drop_used_columns=True,
                drop_unused_columns=False,
                column_prefix=column_prefix,
            )
            xxres = dummy.fit_transform(xx)

            assert dummy.get_feature_names() == [2, 3, 4] + expected
            assert dummy.get_feature_names() == list(xxres.columns)
            assert dummy.get_feature_names(input_features) == ["COL_2", "COL_3", "COL_4"] + expected

            dummy = klass(
                n=2,
                columns_to_use=[0, 1],
                drop_used_columns=False,
                drop_unused_columns=False,
                column_prefix=column_prefix,
            )

            xxres = dummy.fit_transform(xx)

            assert dummy.get_feature_names() == [0, 1, 2, 3, 4] + expected
            assert dummy.get_feature_names() == list(xxres.columns)
            assert dummy.get_feature_names(input_features) == ["COL_0", "COL_1", "COL_2", "COL_3", "COL_4"] + expected

            ## on df ##
            dummy = klass(n=2, column_prefix=column_prefix)

            xxres = dummy.fit_transform(df)
            assert dummy.get_feature_names() == expected
            assert list(xxres.columns) == expected

            for columns_to_use in ([0, 1], ["COL_0", "COL_1"]):
                dummy = klass(
                    n=2,
                    columns_to_use=columns_to_use,
                    drop_used_columns=True,
                    drop_unused_columns=False,
                    column_prefix=column_prefix,
                )
                xxres = dummy.fit_transform(df)

                assert dummy.get_feature_names() == ["COL_2", "COL_3", "COL_4"] + expected
                assert dummy.get_feature_names() == list(xxres.columns)
                assert dummy.get_feature_names(input_features) == ["COL_2", "COL_3", "COL_4"] + expected

                dummy = klass(
                    n=2,
                    columns_to_use=columns_to_use,
                    drop_used_columns=False,
                    drop_unused_columns=False,
                    column_prefix=column_prefix,
                )
                xxres = dummy.fit_transform(df)

                assert dummy.get_feature_names() == ["COL_0", "COL_1", "COL_2", "COL_3", "COL_4"] + expected
                assert dummy.get_feature_names() == list(xxres.columns)
                assert (
                    dummy.get_feature_names(input_features) == ["COL_0", "COL_1", "COL_2", "COL_3", "COL_4"] + expected
                )


def test_dummy_wrapper_features_with_input_features():
    np.random.seed(123)
    xx = np.random.randn(10, 5)
    input_features = ["COL_%d" % i for i in range(xx.shape[1])]
    df = pd.DataFrame(xx, columns=input_features)

    column_prefix = "RAND"
    expected_no_input = ["RAND__c_0_0", "RAND__c_1_1"]
    expected_input = ["RAND__c_COL_0_0", "RAND__c_COL_1_1"]

    ## On array ##
    dummy = DummyWrappedWithInputFeaturesNames(n=2, column_prefix=column_prefix)

    xxres = dummy.fit_transform(xx)
    assert dummy.get_feature_names() == expected_no_input
    assert list(xxres.columns) == expected_no_input
    assert dummy.get_feature_names(input_features) == expected_input

    ## On df ##
    dummy = DummyWrappedWithInputFeaturesNames(n=2, column_prefix=column_prefix)

    xxres = dummy.fit_transform(df)
    assert dummy.get_feature_names() == expected_input
    assert list(xxres.columns) == expected_input
    assert dummy.get_feature_names(input_features) == expected_input


def test_dummy_wrapper_fails():
    np.random.seed(123)
    xx = np.random.randn(10, 5)
    input_features = ["COL_%d" % i for i in range(xx.shape[1])]
    df = pd.DataFrame(xx, columns=input_features)

    dummy = DummyWrapped(n=1)
    dummy.fit(df)

    df_t = dummy.transform(df)
    assert df_t.shape[0] == df.shape[0]

    with pytest.raises(ValueError):
        dummy.transform(df.values)  # fail because wrong type

    with pytest.raises(ValueError):
        dummy.transform(df.iloc[:, 0:3])  # fail because wront number of columns

    df2 = df.copy()
    df2["new_col"] = 10

    with pytest.raises(ValueError):
        dummy.transform(df2)  # fail because wront number of columns

    input_features_wrong_order = input_features[1:] + [input_features[0]]
    with pytest.raises(ValueError):
        dummy.transform(df.loc[:, input_features_wrong_order])  # fail because wront number of columns


class DummyOnlyDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("only works for a DataFrame")
            
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("only works for a DataFrame")
            
        return X
    

class DummyNoDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            raise TypeError("doesn't work on DatraFrame")
            
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            raise TypeError("doesn't work on DatraFrame")
            
        return X


def test_AutoWrapper():
    np.random.seed(123)
    X = np.random.randn(100,10)
    df = pd.DataFrame(X, columns=[f"NUMBER_{j}" for j in range(X.shape[1])])
    df["not_a_number"] = "a"

    model = AutoWrapper(TruncatedSVD(n_components=2, random_state=123))(columns_to_use=["NUMBER_"],
                                                                        regex_match=True,
                                                                        drop_unused_columns=False)
    Xres = model.fit_transform(df)
    
    assert isinstance(Xres, pd.DataFrame)
    assert Xres.shape[0] == df.shape[0]
    
    
    model = AutoWrapper(TruncatedSVD(n_components=2, random_state=123))(columns_to_use=["NUMBER_"],
                                                                        regex_match=True,
                                                                        column_prefix="SVD",
                                                                        drop_unused_columns=False)
    Xres = model.fit_transform(df)
    assert isinstance(Xres, pd.DataFrame)
    assert list(Xres.columns) == ["not_a_number", "SVD__0", "SVD__1"]
    assert Xres.shape[0] == df.shape[0]

    dummy_not_wrapped = DummyOnlyDataFrame()
    with pytest.raises(TypeError):
        dummy_not_wrapped.fit_transform(X)

    dummy_auto_wrapped = AutoWrapper(DummyOnlyDataFrame())()
    Xres = dummy_auto_wrapped.fit_transform(X)
    assert isinstance(Xres, pd.DataFrame)
    assert (Xres.values == X).all()

    dummy_auto_wrapped = AutoWrapper(DummyOnlyDataFrame)()
    Xres = dummy_auto_wrapped.fit_transform(X)
    assert isinstance(Xres, pd.DataFrame)
    assert (Xres.values == X).all()
    
    dummy_not_wrapped = DummyNoDataFrame()
    with pytest.raises(TypeError):
        dummy_not_wrapped.fit_transform(df)
    
    dummy_auto_wrapped = AutoWrapper(DummyNoDataFrame, wrapping_kwargs={"accepted_input_types":(DataTypes.NumpyArray,)})()
    Xres = dummy_auto_wrapped.fit_transform(df)
    assert isinstance(Xres, pd.DataFrame)

        
def test_AutoWrapper_fails_if_not_instance():
    model = 10
    with pytest.raises(TypeError):
        AutoWrapper(model)

