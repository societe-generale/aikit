# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:01:47 2018

@author: Lionel Massoulard
"""

import pytest

import pandas as pd
import numpy as np

import pickle
import itertools

from tests.helpers.testing_help import get_sample_df
from aikit.transformers.target import TargetEncoderClassifier, TargetEncoderEntropyClassifier, TargetEncoderRegressor


def test_loc_align():
    # Test to prevent from a change in pandas behavior
    s1 = pd.Series([10, 11, 12], index=[1, 2, 3])
    s2 = pd.Series(data=0, index=[0, 1, 2])

    inter = np.intersect1d(s2.index, s1.index)

    s2.loc[inter] = s1.loc[inter]

    assert list(s2.values) == [0, 10, 11]


def test_TargetEncoderRegressor_columns_to_use_object():
    np.random.seed(123)
    Xnum = np.random.randn(1000, 10)

    dfX = pd.DataFrame(Xnum, columns=["col_%d" % i for i in range(10)])
    dfX["object_column"] = ["string_%2.4f" % x for x in dfX["col_0"]]

    y = np.random.randn(1000)

    # with --object--
    encoder = TargetEncoderRegressor(columns_to_use="object", drop_unused_columns=False)
    dfX_enc = encoder.fit_transform(dfX, y)

    assert not (dfX_enc.dtypes == "object").any()

    # with default behavior
    encoder = TargetEncoderRegressor()
    dfX_enc = encoder.fit_transform(dfX, y)

    assert "object_column" in dfX_enc
    assert (dfX_enc["object_column"] == dfX["object_column"]).all()


@pytest.mark.parametrize("cv, noise_level", list(itertools.product((None, 10), (None, 0.1))))
def test_TargetEncoderRegressor(cv, noise_level):
    df = get_sample_df(100)
    df["cat_col"] = df["text_col"].apply(lambda s: s[0:3])
    np.random.seed(123)
    y = np.random.randn(100)

    encoder = TargetEncoderRegressor(noise_level=noise_level, cv=cv)
    encoder.fit(df, y)
    res = encoder.transform(df)

    assert encoder.get_feature_names() == ["float_col", "int_col", "text_col", "cat_col__target_mean"]
    assert list(res.columns) == ["float_col", "int_col", "text_col", "cat_col__target_mean"]
    assert res["cat_col__target_mean"].isnull().sum() == 0
    assert (res.index == df.index).all()
    assert encoder._columns_informations["input_columns"] == ["cat_col"]

    temp = pd.DataFrame({"cat_col": df["cat_col"], "cat_col__target_mean": res["cat_col__target_mean"]})
    assert temp.groupby("cat_col")["cat_col__target_mean"].std().max() == 0

    encoder = TargetEncoderRegressor(noise_level=noise_level, cv=cv)
    res = encoder.fit_transform(df, y)

    assert encoder.get_feature_names() == ["float_col", "int_col", "text_col", "cat_col__target_mean"]
    assert list(res.columns) == ["float_col", "int_col", "text_col", "cat_col__target_mean"]
    assert res["cat_col__target_mean"].isnull().sum() == 0
    assert (res.index == df.index).all()
    assert encoder._columns_informations["input_columns"] == ["cat_col"]


def test_TargetEncoderRegressor_is_picklable():
    df = get_sample_df(100)
    df["cat_col"] = df["text_col"].apply(lambda s: s[0:3])
    np.random.seed(123)
    y = np.random.randn(100)

    encoder = TargetEncoderRegressor(cv=2)
    encoder.fit(df, y)

    pickled_encoder = pickle.dumps(encoder)
    unpickled_encoder = pickle.loads(pickled_encoder)
    
    assert type(unpickled_encoder) == type(encoder)
    X1 = encoder.transform(df)
    X2 = unpickled_encoder.transform(df)
    
    assert X1.shape == X2.shape
    assert (X1 == X2).all().all()


@pytest.mark.parametrize("cv, noise_level", list(itertools.product((None, 10), (None, 0.1))))
def test_TargetEncoderClassifier(cv, noise_level):
    df = get_sample_df(100)
    df["cat_col"] = df["text_col"].apply(lambda s: s[0:3])

    np.random.seed(123)
    y = 1 * (np.random.randn(100) > 0)

    encoder = TargetEncoderClassifier(noise_level=noise_level, cv=cv)
    encoder.fit(df, y)
    res = encoder.transform(df)

    assert encoder.get_feature_names() == ["float_col", "int_col", "text_col", "cat_col__target_1"]
    assert list(res.columns) == ["float_col", "int_col", "text_col", "cat_col__target_1"]
    assert res["cat_col__target_1"].isnull().sum() == 0
    assert res["cat_col__target_1"].isnull().max() <= 1
    assert res["cat_col__target_1"].isnull().min() >= 0

    temp = pd.DataFrame({"cat_col": df["cat_col"], "cat_col__target_1": res["cat_col__target_1"]})
    assert temp.groupby("cat_col")["cat_col__target_1"].std().max() == 0

    assert (res.index == df.index).all()
    assert encoder._columns_informations["input_columns"] == ["cat_col"]

    encoder = TargetEncoderClassifier(noise_level=noise_level, cv=cv)
    res = encoder.fit_transform(df, y)

    assert encoder.get_feature_names() == ["float_col", "int_col", "text_col", "cat_col__target_1"]
    assert list(res.columns) == ["float_col", "int_col", "text_col", "cat_col__target_1"]

    assert res["cat_col__target_1"].isnull().sum() == 0
    assert res["cat_col__target_1"].isnull().max() <= 1
    assert res["cat_col__target_1"].isnull().min() >= 0

    assert (res.index == df.index).all()
    assert encoder._columns_informations["input_columns"] == ["cat_col"]

    np.random.seed(123)
    y = np.array(["aa", "bb", "cc"])[np.random.randint(0, 3, size=100)]

    encoder = TargetEncoderClassifier(noise_level=noise_level, cv=cv)
    encoder.fit(df, y)
    res = encoder.transform(df)

    assert encoder.get_feature_names() == [
        "float_col",
        "int_col",
        "text_col",
        "cat_col__target_aa",
        "cat_col__target_bb",
        "cat_col__target_cc",
    ]
    assert list(res.columns) == [
        "float_col",
        "int_col",
        "text_col",
        "cat_col__target_aa",
        "cat_col__target_bb",
        "cat_col__target_cc",
    ]

    for col in ("cat_col__target_aa", "cat_col__target_bb", "cat_col__target_cc"):
        assert res[col].isnull().sum() == 0
        assert res[col].isnull().max() <= 1
        assert res[col].isnull().min() >= 0

        temp = pd.DataFrame({"cat_col": df["cat_col"], col: res[col]})
        assert temp.groupby("cat_col")[col].std().max() == 0

    assert encoder._columns_informations["input_columns"] == ["cat_col"]
    assert (res.index == df.index).all()

    encoder = TargetEncoderClassifier(noise_level=noise_level, cv=cv)
    res = encoder.fit_transform(df, y)

    assert encoder.get_feature_names() == [
        "float_col",
        "int_col",
        "text_col",
        "cat_col__target_aa",
        "cat_col__target_bb",
        "cat_col__target_cc",
    ]
    assert list(res.columns) == [
        "float_col",
        "int_col",
        "text_col",
        "cat_col__target_aa",
        "cat_col__target_bb",
        "cat_col__target_cc",
    ]
    for col in ("cat_col__target_aa", "cat_col__target_bb", "cat_col__target_cc"):
        assert res[col].isnull().sum() == 0
        assert res[col].isnull().max() <= 1
        assert res[col].isnull().min() >= 0
    assert (res.index == df.index).all()
    assert encoder._columns_informations["input_columns"] == ["cat_col"]


def test_TargetEncoderClassifier_is_picklable():
    df = get_sample_df(100)
    df["cat_col"] = df["text_col"].apply(lambda s: s[0:3])
    np.random.seed(123)
    y = 1 * (np.random.randn(100) > 0)

    encoder = TargetEncoderClassifier(cv=2)
    encoder.fit(df, y)

    pickled_encoder = pickle.dumps(encoder)
    unpickled_encoder = pickle.loads(pickled_encoder)
    
    assert type(unpickled_encoder) == type(encoder)
    X1 = encoder.transform(df)
    X2 = unpickled_encoder.transform(df)
    
    assert X1.shape == X2.shape
    assert (X1 == X2).all().all()


def test_target_encoder_with_cat_dtypes():
    np.random.seed(123)
    X = get_sample_df(100)
    X["cat_col_1"] = X["text_col"].apply(lambda s: s[0:3])
    y = 1 * (np.random.randn(100) > 0)

    encoder = TargetEncoderClassifier()
    X_no_cat_dtype_encoded = encoder.fit_transform(X, y)

    X_cat_dtype = X.copy()
    X_cat_dtype["cat_col_1"] = X_cat_dtype["cat_col_1"].astype("category")
    X_with_cat_dtype_encoded = encoder.fit_transform(X_cat_dtype, y)

    assert (X_with_cat_dtype_encoded == X_no_cat_dtype_encoded).all().all()
    assert (X_with_cat_dtype_encoded.dtypes == X_no_cat_dtype_encoded.dtypes).all()


@pytest.mark.parametrize("cv, noise_level", list(itertools.product((None, 10), (None, 0.1))))
def test_TargetEncoderEntropyClassifier(cv, noise_level):
    df = get_sample_df(100)
    df["cat_col"] = df["text_col"].apply(lambda s: s[0:3])

    np.random.seed(123)
    y = 1 * (np.random.randn(100) > 0)

    encoder = TargetEncoderEntropyClassifier(noise_level=noise_level, cv=cv)
    encoder.fit(df, y)
    res = encoder.transform(df)

    assert encoder.get_feature_names() == ["float_col", "int_col", "text_col", "cat_col__target_entropy"]
    assert list(res.columns) == ["float_col", "int_col", "text_col", "cat_col__target_entropy"]
    assert res["cat_col__target_entropy"].isnull().sum() == 0
    assert res["cat_col__target_entropy"].isnull().min() >= 0
    temp = pd.DataFrame({"cat_col": df["cat_col"], "cat_col__target_entropy": res["cat_col__target_entropy"]})
    assert temp.groupby("cat_col")["cat_col__target_entropy"].std().max() == 0

    assert encoder._columns_informations["input_columns"] == ["cat_col"]
    assert (res.index == df.index).all()

    encoder = TargetEncoderEntropyClassifier(noise_level=noise_level, cv=cv)
    res = encoder.fit_transform(df, y)

    assert encoder.get_feature_names() == ["float_col", "int_col", "text_col", "cat_col__target_entropy"]
    assert list(res.columns) == ["float_col", "int_col", "text_col", "cat_col__target_entropy"]
    assert res["cat_col__target_entropy"].isnull().sum() == 0
    assert res["cat_col__target_entropy"].isnull().min() >= 0
    assert (res.index == df.index).all()
    assert encoder._columns_informations["input_columns"] == ["cat_col"]
