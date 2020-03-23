# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:59:39 2018

@author: Lionel Massoulard
"""

import pytest

import itertools
import numpy as np
import pandas as pd

from aikit.transformers import NumericalEncoder, OrdinalOneHotEncoder

# from aikit.transformers.categories import NumericalEncoder
from tests.helpers.testing_help import get_sample_df

import pickle

try:
    import category_encoders
except ModuleNotFoundError:
    category_encoders = None


def test_NumericalEncoder_dummy_output_dtype():
    np.random.seed(123)
    df = get_sample_df(100, seed=123)
    ind = np.arange(len(df))
    df.index = ind

    df["cat_col_1"] = df["text_col"].apply(lambda s: s[0:3])
    df["cat_col_2"] = df["text_col"].apply(lambda s: s[3:6])

    encoder = NumericalEncoder(encoding_type="dummy")
    encoder.fit(df)
    res = encoder.transform(df)

    assert (res.dtypes[res.columns.str.startswith("cat_col_")] == "int32").all()  # check default encoding type = int32


def test_NumericalEncoder_with_cat_dtypes():
    np.random.seed(123)
    X = get_sample_df(100)
    X["cat_col_1"] = X["text_col"].apply(lambda s: s[0:3])

    encoder = NumericalEncoder(columns_to_use=["cat_col_1"])
    X_no_cat_dtype_encoded = encoder.fit_transform(X)

    X_cat_dtype = X.copy()
    X_cat_dtype["cat_col_1"] = X_cat_dtype["cat_col_1"].astype("category")
    X_with_cat_dtype_encoded = encoder.fit_transform(X_cat_dtype)

    assert X_with_cat_dtype_encoded.shape == X_no_cat_dtype_encoded.shape
    assert (X_with_cat_dtype_encoded == X_no_cat_dtype_encoded).all().all()
    assert (X_with_cat_dtype_encoded.dtypes == X_no_cat_dtype_encoded.dtypes).all()


def test_NumericalEncoder_int_as_cat():
    df = get_sample_df(100)[["float_col", "int_col"]]
    df["int_cat"] = np.random.choice((0, 1, 2), 100)
    df["int_cat"] = df["int_cat"].astype("category")

    encoder = NumericalEncoder()
    df_transformed = encoder.fit_transform(df)

    assert "int_cat" not in df_transformed.columns
    assert df["int_cat"].nunique() + 2 == df_transformed.shape[1]
    assert df.loc[df["int_cat"] == 1, "int_cat"].shape[0] == (df["int_cat"] == 1).sum()


def test_NumericalEncoder_nothing_to_do():
    df = get_sample_df(100)[["float_col", "int_col"]]

    encoder = NumericalEncoder()
    df_transformed = encoder.fit_transform(df)

    assert (df.values == df_transformed.values).all().all()
    assert (df.dtypes == df_transformed.dtypes).all()


def test_NumericalEncoder_encode_int():
    df = get_sample_df(100)[["float_col"]]
    df["int_col"] = np.random.choice((0, 1, 2), 100)

    encoder = NumericalEncoder(columns_to_use=["int_col"])
    df_transformed = encoder.fit_transform(df)

    df_copy = df.copy()
    df_copy["int_col"] = df_copy["int_col"].astype("category")

    encoder_2 = NumericalEncoder()
    df_copy_transformed = encoder_2.fit_transform(df_copy)

    assert (df_transformed.values == df_copy_transformed.values).all().all()
    assert (df_transformed.dtypes == df_copy_transformed.dtypes).all()
    assert df_transformed.shape[1] == 1 + df["int_col"].nunique()


def test_NumericalEncoder_columns_to_encode_object():
    np.random.seed(123)
    Xnum = np.random.randn(1000, 10)

    dfX = pd.DataFrame(Xnum, columns=["col_%d" % i for i in range(10)])
    dfX["object_column"] = ["string_%2.4f" % x for x in dfX["col_0"]]

    # with --object--
    encoder = NumericalEncoder(columns_to_use="object")
    dfX_enc = encoder.fit_transform(dfX)

    assert not (dfX_enc.dtypes == "object").any()

    # with default behavior
    encoder = NumericalEncoder()
    dfX_enc = encoder.fit_transform(dfX)

    assert "object_column" in dfX_enc
    assert (dfX_enc["object_column"] == dfX["object_column"]).all()


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
    assert res2.loc[0, "cat_col_2__" + df2.loc[0, "cat_col_2"]] == 1  # activated in the right position
    assert res2.loc[0, col2].sum() == 1  # only one dummy activate

    assert res2.loc[1, col2].sum() == 0  # no dummy activated
    assert res2.loc[1, "cat_col_1__" + df2.loc[1, "cat_col_1"]] == 1  # activated in the right position
    assert res2.loc[1, col1].sum() == 1

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


def test_NumericalEncoder_num_output_dtype():
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

    assert res.dtypes["cat_col_1"] == "int32"
    assert res.dtypes["cat_col_2"] == "int32"


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


def test_NumericalEncoder_num_fit_parameters():

    np.random.seed(123)
    df = get_sample_df(100, seed=123)
    df.index = np.arange(len(df))

    df["cat_col_1"] = df["text_col"].apply(lambda s: s[0:3])
    df["cat_col_2"] = df["text_col"].apply(lambda s: s[4:7])
    df["cat_col_3"] = df["text_col"].apply(lambda s: s[8:11])
    df.loc[0:10, "cat_col_3"] = None

    # All modalities are kept, __null__ category is created
    encoder = NumericalEncoder(
        encoding_type="num",
        min_modalities_number=10,
        max_modalities_number=100,
        max_na_percentage=0,
        min_nb_observations=1,
        max_cum_proba=1,
    )
    res = encoder.fit_transform(df)
    assert len(encoder.model.variable_modality_mapping["cat_col_1"]) == 7
    assert len(encoder.model.variable_modality_mapping["cat_col_3"]) == 8

    # We filter on max_cum_proba, __null__ category is created
    encoder = NumericalEncoder(
        encoding_type="num",
        min_modalities_number=1,
        max_modalities_number=100,
        max_na_percentage=0,
        min_nb_observations=1,
        max_cum_proba=0.6,
    )
    res = encoder.fit_transform(df)
    map1 = encoder.model.variable_modality_mapping["cat_col_1"]
    assert len(map1) == 5
    assert np.all([v in map1 for v in ["eee", "bbb", "ddd", "jjj", "__default__"]])
    map3 = encoder.model.variable_modality_mapping["cat_col_3"]
    assert len(map3) == 6
    assert np.all([v in map3 for v in ["bbb", "ddd", "ccc", "aaa", "jjj", "__default__"]])

    # No __null__ category
    encoder = NumericalEncoder(
        encoding_type="num",
        min_modalities_number=1,
        max_modalities_number=100,
        max_na_percentage=0.2,
        min_nb_observations=1,
        max_cum_proba=1,
    )
    res = encoder.fit_transform(df)
    assert len(encoder.model.variable_modality_mapping["cat_col_3"]) == 7

    # Max modalities
    encoder = NumericalEncoder(
        encoding_type="num",
        min_modalities_number=1,
        max_modalities_number=3,
        max_na_percentage=0.2,
        min_nb_observations=1,
        max_cum_proba=1,
    )
    res = encoder.fit_transform(df)
    assert len(encoder.model.variable_modality_mapping["cat_col_1"]) == 4
    assert len(encoder.model.variable_modality_mapping["cat_col_2"]) == 4
    assert len(encoder.model.variable_modality_mapping["cat_col_3"]) == 4

    assert res["cat_col_1"].nunique() == 4
    assert res["cat_col_2"].nunique() == 4
    assert res["cat_col_3"].nunique() == 4


def test_NumericalEncoder_default_and_null_values():
    np.random.seed(123)
    df = get_sample_df(100, seed=123)
    df.index = np.arange(len(df))

    df["cat_col_1"] = df["text_col"].apply(lambda s: s[0:3])
    df.loc[0:10, "cat_col_1"] = None

    # All modalities are kept, __null__ category is created
    encoder = NumericalEncoder(encoding_type="num", min_modalities_number=2, max_cum_proba=0.8, max_na_percentage=0)

    res = encoder.fit_transform(df)
    assert "__default__" in encoder.model.variable_modality_mapping["cat_col_1"]
    assert "__null__" in encoder.model.variable_modality_mapping["cat_col_1"]

    df["cat_col_1"] = "zzz"  # Never seen value
    res = encoder.transform(df)
    assert res["cat_col_1"].unique()[0] == encoder.model.variable_modality_mapping["cat_col_1"]["__default__"]

    df["cat_col_1"] = None
    res = encoder.transform(df)
    assert res["cat_col_1"].unique()[0] == encoder.model.variable_modality_mapping["cat_col_1"]["__null__"]


def test_NumericalEncoder_with_boolean():
    dfX = pd.DataFrame({"c": [True, False] * 200})

    enc = NumericalEncoder()

    dfX_encoded = enc.fit_transform(dfX)

    assert "c__True" in dfX_encoded.columns
    assert "c__False" in dfX_encoded.columns
    assert ((dfX_encoded["c__True"] == 1) == (dfX["c"])).all()
    assert ((dfX_encoded["c__False"] == 1) == (~dfX["c"])).all()
    assert dfX_encoded["c__True"].dtype == np.int32
    assert dfX_encoded["c__False"].dtype == np.int32


@pytest.mark.parametrize(
    "drop_used_columns, drop_unused_columns, columns_to_use",
    list(itertools.product((True, False), (True, False), ("all", "object", ["num1", "num2", "num3"]))),
)
def test_NumericalEncoder_drop_used_unused_columns(drop_used_columns, drop_unused_columns, columns_to_use):
    # This test will verify the behavior of the encoder regarding the fact to drop or keep the use/unused columns

    df = pd.DataFrame(
        {
            "obj1": ["a", "b", "c", "d"] * 25,
            "obj2": ["AA", "BB"] * 50,
            "num1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10,
            "num2": [100, 101, 102, 103, 104] * 20,
            "num3": [0.01, 0.02, 0.03, 0.04, 0.05] * 20,
        }
    )

    df1 = df.loc[0:20,]
    df2 = df.loc[20:]

    # for drop_used_columns, drop_unused_columns, columns_to_use in list(itertools.product((True,False),(True,False),("all","object",["num1","num2","num3"]))):

    resulting_columns = {col: ["%s__%s" % (col, str(v)) for v in df[col].value_counts().index] for col in df.columns}

    if columns_to_use == "all":
        cols = list(df.columns)
    elif columns_to_use == "object":
        cols = list(df.columns[df.dtypes == "object"])
    else:
        cols = columns_to_use

    if drop_used_columns:
        columns_A = []
    else:
        columns_A = cols

    columns_B = []
    for c in cols:
        columns_B += resulting_columns[c]

    if drop_unused_columns:
        columns_C = []
    else:
        columns_C = [c for c in df.columns if c not in cols]

    final_columns = columns_A + columns_C + columns_B

    encoder = NumericalEncoder(
        columns_to_use=columns_to_use, drop_used_columns=drop_used_columns, drop_unused_columns=drop_unused_columns
    )

    df1_transformed = encoder.fit_transform(df1)
    df2_transformed = encoder.transform(df2)

    assert df1_transformed.shape[0] == df1.shape[0]
    assert df2_transformed.shape[0] == df2.shape[0]
    assert type(df1_transformed) == type(df1)
    assert type(df2_transformed) == type(df2)
    assert (df1_transformed.index == df1.index).all()
    assert (df2_transformed.index == df2.index).all()

    assert df1_transformed.shape[1] == df2_transformed.shape[1]
    assert list(df1_transformed.columns) == list(df2_transformed.columns)

    assert len(df1_transformed.columns) == len(final_columns)
    assert set(df1_transformed) == set(final_columns)


#    assert list(df1_transformed.columns) == final_columns


    encoder = NumericalEncoder()
    encoder.fit(df)

    pickled_encoder = pickle.dumps(encoder)
    unpickled_encoder = pickle.loads(pickled_encoder)
    
    assert type(unpickled_encoder) == type(encoder)
    X1 = encoder.transform(df)
    X2 = unpickled_encoder.transform(df)
    
    assert X1.shape == X2.shape
    assert (X1 == X2).all().all()

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



# In[]

def test_OrdinalOneHotEncoder():

    np.random.seed(123)
    
    X = pd.DataFrame({"ord1":np.random.randint(0,3, size=100),
                      "ord2":np.array(["A","B","C","D"])[np.random.randint(0,4, size=100)]
                          })
            

    # Test 1 :
    model = OrdinalOneHotEncoder(columns_to_use="all")
    model.fit(X)
    assert model.get_feature_names() == ['ord1_g_0', 'ord1_g_1', 'ord2_g_A', 'ord2_g_B', 'ord2_g_C']
    X_enc = model.transform(X)
    assert isinstance(X_enc, pd.DataFrame)
    assert X_enc.shape[0] == X.shape[0]
    assert list(X_enc.columns) == model.get_feature_names()
    
    assert (X_enc["ord1_g_0"] == (X["ord1"] > 0)).all()
    assert (X_enc["ord1_g_1"] == (X["ord1"] > 1)).all()
    
    assert (X_enc["ord2_g_A"] == (X["ord2"] > "A")).all()
    assert (X_enc["ord2_g_B"] == (X["ord2"] > "B")).all()
    assert (X_enc["ord2_g_C"] == (X["ord2"] > "C")).all()
    
    X_orig = model.inverse_transform(X_enc)
    assert X_orig.shape == X.shape
    assert X_orig.columns.tolist() == X.columns.tolist()
    assert (X_orig == X).all().all()

    
    # Test 2 : change order
    model = OrdinalOneHotEncoder(columns_to_use="all", categories={"ord1":[0,1,2], "ord2":["D","C","B","A"]})
    model.fit(X)
    
    assert model.get_feature_names() == ['ord1_g_0', 'ord1_g_1', 'ord2_g_D', 'ord2_g_C', 'ord2_g_B']
    X_enc = model.transform(X)
    assert (X_enc.dtypes == model.dtype).all()
    
    assert isinstance(X_enc, pd.DataFrame)
    assert X_enc.shape[0] == X.shape[0]
    assert list(X_enc.columns) == model.get_feature_names()
    
    assert (X_enc["ord1_g_0"] == (X["ord1"] > 0)).all()
    assert (X_enc["ord1_g_1"] == (X["ord1"] > 1)).all()
    
    assert (X_enc["ord2_g_D"] == (X["ord2"] < "D")).all() # reverse order this time
    assert (X_enc["ord2_g_C"] == (X["ord2"] < "C")).all()
    assert (X_enc["ord2_g_B"] == (X["ord2"] < "B")).all() 
    
    
    X_orig = model.inverse_transform(X_enc)
    assert X_orig.shape == X.shape
    assert X_orig.columns.tolist() == X.columns.tolist()
    assert (X_orig == X).all().all()

