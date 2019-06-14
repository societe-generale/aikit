# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:59:39 2018

@author: Lionel Massoulard
"""

import pytest

import numpy as np
import pandas as pd

from aikit.transformers.categories import NumericalEncoder
from tests.helpers.testing_help import get_sample_df

try:
    import category_encoders
except ModuleNotFoundError:
    category_encoders = None


def test_NumericalEncoder_dummy():

    ####################
    ### One Hot Mode ###
    ####################

    np.random.seed(123)
    df = get_sample_df(100, seed=123)
    ind = np.arange(len(df))
    df.index = ind

    df["cat_col_1"] = df["text_col"].apply(lambda s: s[0:3])
    df["cat_col_2"] = df["text_col"].apply(lambda s: s[3:6])

    encoder = NumericalEncoder(encoding_type="dummy")
    encoder.fit(df)
    res = encoder.transform(df)

    assert encoder.model._dummy_size == len(encoder.model._dummy_feature_names)
    assert encoder.model._dummy_size == sum(len(v) for k, v in encoder.model.variable_modality_mapping.items())

    assert res.shape[0] == df.shape[0]
    assert res.shape[1] == len(df["cat_col_1"].value_counts()) + len(df["cat_col_2"].value_counts()) + 3
    assert (res.index == df.index).all()

    col = ["float_col", "int_col", "text_col"]
    col1 = ["cat_col_1__%s" % c for c in list(df["cat_col_1"].value_counts().index)]
    col2 = ["cat_col_2__%s" % c for c in list(df["cat_col_2"].value_counts().index)]

    assert col1 == encoder.columns_mapping["cat_col_1"]
    assert col2 == encoder.columns_mapping["cat_col_2"]

    assert encoder.get_feature_names() == encoder.model._feature_names
    assert encoder.get_feature_names() == col + col1 + col2

    assert (res.loc[:, col1 + col2]).isnull().sum().sum() == 0
    assert (res.loc[:, col1 + col2]).max().max() == 1
    assert (res.loc[:, col1 + col2]).min().min() == 0

    assert ((df["cat_col_1"] == "aaa") == (res["cat_col_1__aaa"] == 1)).all()

    df2 = df.copy()
    df2.loc[0, "cat_col_1"] = "something-new"
    df2.loc[1, "cat_col_2"] = None  # Something None

    res2 = encoder.transform(df2)
    assert res2.loc[0, col1].sum() == 0  # no dummy activated
    assert res2.loc[1, col2].sum() == 0  # no dummy activated

    df_with_none = df.copy()
    df_with_none["cat_col_3"] = df_with_none["cat_col_1"]
    df_with_none.loc[0:25, "cat_col_3"] = None

    encoder2 = NumericalEncoder(encoding_type="dummy")
    res2 = encoder2.fit_transform(df_with_none)

    col3b = [c for c in res2.columns if c.startswith("cat_col_3")]
    assert col3b[0] == "cat_col_3____null__"
    assert list(res2.columns) == col + col1 + col2 + col3b
    assert list(res2.columns) == encoder2.get_feature_names()

    assert (res2.loc[:, col1 + col2 + col3b]).isnull().sum().sum() == 0
    assert (res2.loc[:, col1 + col2 + col3b]).max().max() == 1
    assert (res2.loc[:, col1 + col2 + col3b]).min().min() == 0

    assert (df_with_none["cat_col_3"].isnull() == (res2["cat_col_3____null__"] == 1)).all()

    df3 = df.copy()
    df3["cat_col_many"] = [
        "m_%d" % x for x in np.ceil(np.minimum(np.exp(np.random.rand(100) * 5), 50)).astype(np.int32)
    ]

    encoder3 = NumericalEncoder(encoding_type="dummy")
    res3 = encoder3.fit_transform(df3)

    colm = [c for c in res3.columns if c.startswith("cat_col_many")]
    vc = df3["cat_col_many"].value_counts()
    colmb = ["cat_col_many__" + c for c in list(vc.index[vc >= encoder3.min_nb_observations]) + ["__default__"]]

    assert colm == colmb


def test_NumericalEncoder_num():

    ######################
    ### Numerical Mode ###
    ######################

    np.random.seed(123)
    df = get_sample_df(100, seed=123)
    ind = np.arange(len(df))
    df.index = ind

    np.random.shuffle(ind)
    df["cat_col_1"] = df["text_col"].apply(lambda s: s[0:3])
    df["cat_col_2"] = df["text_col"].apply(lambda s: s[3:6])

    encoder = NumericalEncoder(encoding_type="num")
    encoder.fit(df)
    res = encoder.transform(df)

    assert res.shape == df.shape
    assert (res.index == df.index).all()

    assert encoder.get_feature_names() == encoder.model._feature_names
    assert encoder.get_feature_names() == list(res.columns)

    df2 = df.copy()
    df2.loc[0, "cat_col_1"] = "something-new"
    df2.loc[1, "cat_col_2"] = None  # Something None

    res2 = encoder.transform(df2)
    assert res2.loc[0, "cat_col_1"] == -1
    assert res2.loc[1, "cat_col_2"] == -1

    df_with_none = df.copy()
    df_with_none["cat_col_3"] = df_with_none["cat_col_1"]
    df_with_none.loc[list(range(25)), "cat_col_3"] = None

    encoder2 = NumericalEncoder(encoding_type="num")
    res2 = encoder2.fit_transform(df_with_none)

    assert (df_with_none["cat_col_3"].isnull() == (res2["cat_col_3"] == 0)).all()


@pytest.mark.xfail()
def test_bug_CategoryEncoder():
    """ this is a bug in CategoryEncoder """
    if category_encoders is None:
        return  # This will show as 'expected to fail but pass' if package isn't installed
    # If the bug is corrected I'll know it
    for klass in (
        category_encoders.HelmertEncoder,
        category_encoders.PolynomialEncoder,
        category_encoders.PolynomialEncoder,
        category_encoders.SumEncoder,
    ):

        df = pd.DataFrame({"cat_col": (np.array(["a", "b", "c", "d"]))[np.random.randint(0, 4, 100)]})
        enc = klass()
        df_enc = enc.fit_transform(df)

        df2 = df.head().copy()
        df2.loc[df2["cat_col"] == "d", "cat_col"] = "a"
        df2_enc = enc.transform(df2)

        assert df_enc.shape[1] == df2_enc.shape[1]


def verif_all():
    test_NumericalEncoder_dummy()
    test_NumericalEncoder_num()
