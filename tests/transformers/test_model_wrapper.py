# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:46:59 2018

@author: Lionel Massoulard
"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator

from aikit.transformers.model_wrapper import ModelWrapper, ColumnsSelector
from aikit.transformers.model_wrapper import (
    _concat,
    DebugPassThrough,
    try_to_find_features_names
)
from aikit.enums import DataTypes

import pytest


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

    with pytest.raises(ValueError):
        selector.transform(dfX2.loc[:, ["text2", "text1"]])  # Error because not correct number of columns

    selector = ColumnsSelector(columns_to_use=["text1", "text2", "text3"])
    with pytest.raises(ValueError):
        selector.fit(dfX)

    selector = ColumnsSelector(columns_to_use=["text1", "text2"])
    selector.fit(dfX)

    dfX3 = dfX2.copy()
    del dfX3["text1"]
    with pytest.raises(ValueError):
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

    selector = ColumnsSelector(columns_to_use=["^text"], regex_match=False)
    r1 = dfX.loc[:, ["text1", "text2"]]
    r2 = dfX2.loc[:, ["text1", "text2"]]
    with pytest.raises(ValueError):
        assert selector.fit_transform(dfX).shape[1] == 0

    selector2 = ColumnsSelector(columns_to_use=[5, 6])
    assert (selector2.fit_transform(dfX) == r1).all().all()
    assert (selector2.transform(dfX2) == r2).all().all()

    selector3 = ColumnsSelector(columns_to_use=[10, 5])
    with pytest.raises(ValueError):
        selector3.fit(dfX)

    selector3 = ColumnsSelector(columns_to_use=[5, 6])
    selector3.fit(dfX)
    dfX2 = dfX.copy()
    del dfX2["text1"]
    with pytest.raises(ValueError):
        selector3.transform(dfX2)

    selector_none = ColumnsSelector(columns_to_use=None)
    assert (selector_none.fit_transform(dfX) == dfX).all().all()

    antiselector = ColumnsSelector(columns_to_drop=["cat1", "cat2"])
    assert (antiselector.fit_transform(dfX) == dfX.loc[:, ["num1", "num2", "num3", "text1", "text2"]]).all().all()
    assert antiselector.get_feature_names() == ["num1", "num2", "num3", "text1", "text2"]

    antiselector = ColumnsSelector(columns_to_drop=["^cat"], regex_match=True)
    assert (antiselector.fit_transform(dfX) == dfX.loc[:, ["num1", "num2", "num3", "text1", "text2"]]).all().all()
    assert antiselector.get_feature_names() == ["num1", "num2", "num3", "text1", "text2"]

    cols = ["cat1", "cat2", "num1", "num2", "num3", "text1", "text2"]
    antiselector2 = ColumnsSelector(columns_to_drop=cols)
    assert antiselector2.fit_transform(dfX).shape == (4, 0)  # No column

    cols = [0, 1, 2, 3, 4, 5, 6]
    antiselector3 = ColumnsSelector(columns_to_drop=cols)
    assert antiselector3.fit_transform(dfX.values).shape == (4, 0)  # No column

    selector3 = ColumnsSelector(columns_to_use="num1")
    n1 = dfX.loc[:, ["num1"]]
    n2 = dfX2.loc[:, ["num1"]]

    dfX2 = dfX.copy()
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


# In[]

# from sklearn.preprocessing import PolynomialFeatures
#
# poly = PolynomialFeatures()
#
# xx = np.random.randn(100,5)
# cols = ["COL_%d" % i for i in range(xx.shape[1])]
# df = pd.DataFrame(xx, columns = cols)
#
# xxres = poly.fit_transform(xx)
#
# poly.get_feature_names()
# poly.get_feature_names(cols)
#
# class WrappedPoly(ModelWrapper):
#
#    def __init__(self, degree = 2, columns_to_use = None):
#        self.degree = degree
#
#        super(WrappedPoly,self).__init__(
#            columns_to_use = columns_to_use,
#            regex_match = False,
#            work_on_one_column_only = False,
#            all_columns_at_once = True,
#            accepted_input_types = None,
#            column_prefix = None,
#            desired_output_type = DataTypes.DataFrame,
#            must_transform_to_get_features_name = True,
#            dont_change_columns = False,
#            keep_other_columns = "drop"
#            )
#
#    def _get_model(self, X , y = None):
#        return PolynomialFeatures(degree = self.degree)
#
# poly = WrappedPoly()
# poly.fit_transform(xx)
# poly.get_feature_names()
# poly.get_feature_names(cols)
#
#
# poly.fit_transform(df)
# poly.get_feature_names()
# poly.get_feature_names(cols)
# cols2 = ["A_%d" % i for i in range(xx.shape[1])]
# poly.get_feature_names(cols2)
#


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
        if input_features is None:
            return ["r%d" % i for i in range(self.n)]
        else:
            return ["c_%s_%d" % (str(input_features[i]), i) for i in range(self.n)]


# def _DummyToWrapWithFeaturesNa


class DummyWrapped(ModelWrapper):
    def __init__(self, n, columns_to_use=None, column_prefix=None, keep_other_columns="drop"):

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
            keep_other_columns=keep_other_columns,
        )

    def _get_model(self, X, y=None):
        return _DummyToWrap(n=self.n)


class DummyWrappedWithFeaturesNames(ModelWrapper):
    def __init__(self, n, columns_to_use=None, column_prefix=None, keep_other_columns="drop"):

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
            keep_other_columns=keep_other_columns,
        )

    def _get_model(self, X, y=None):
        return _DummyToWrapWithFeaturesNames(n=self.n)


class DummyWrappedWithInputFeaturesNames(ModelWrapper):
    def __init__(self, n, columns_to_use=None, column_prefix=None, keep_other_columns="drop"):

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
            keep_other_columns=keep_other_columns,
        )

    def _get_model(self, X, y=None):
        return _DummyToWrapWithInputFeaturesNames(n=self.n)


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

            dummy = klass(n=2, columns_to_use=[0, 1], keep_other_columns="delta", column_prefix=column_prefix)
            xxres = dummy.fit_transform(xx)

            assert dummy.get_feature_names() == [2, 3, 4] + expected
            assert dummy.get_feature_names() == list(xxres.columns)
            assert dummy.get_feature_names(input_features) == ["COL_2", "COL_3", "COL_4"] + expected

            dummy = klass(n=2, columns_to_use=[0, 1], keep_other_columns="keep", column_prefix=column_prefix)

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
                    n=2, columns_to_use=columns_to_use, keep_other_columns="delta", column_prefix=column_prefix
                )
                xxres = dummy.fit_transform(df)

                assert dummy.get_feature_names() == ["COL_2", "COL_3", "COL_4"] + expected
                assert dummy.get_feature_names() == list(xxres.columns)
                assert dummy.get_feature_names(input_features) == ["COL_2", "COL_3", "COL_4"] + expected

                dummy = klass(
                    n=2, columns_to_use=columns_to_use, keep_other_columns="keep", column_prefix=column_prefix
                )
                xxres = dummy.fit_transform(df)

                assert dummy.get_feature_names() == ["COL_0", "COL_1", "COL_2", "COL_3", "COL_4"] + expected
                assert dummy.get_feature_names() == list(xxres.columns)
                assert (
                    dummy.get_feature_names(input_features) == ["COL_0", "COL_1", "COL_2", "COL_3", "COL_4"] + expected
                )


def test_dummy_wrapper_features_with_input_features():
    xx = np.random.randn(10, 5)
    input_features = ["COL_%d" % i for i in range(xx.shape[1])]
    df = pd.DataFrame(xx, columns=input_features)

    column_prefix = "RAND"
    expected_no_input = ["RAND__r0", "RAND__r1"]
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


# In[]


def verif_all():
    test_try_to_find_features_names()
    test_ColumnsSelector()
    test__concat()
