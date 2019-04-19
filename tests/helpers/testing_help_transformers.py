# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 14:55:21 2018

@author: Lionel Massoulard
"""

import pandas as pd
import scipy.sparse

import pytest

from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from tests.helpers.testing_help import rec_assert_equal
from aikit.tools.data_structure_helper import convert_generic, get_type
from aikit.enums import DataTypes


def assert_raise_not_fitted(encoder, df):
    with pytest.raises(NotFittedError):
        encoder.transform(df)


def assert_raise_value_error(encoder, df):
    with pytest.raises(ValueError):
        encoder.transform(df)


def gen_slice(ob, sl):
    """ generic column slicer """
    t = get_type(ob)
    if t in (DataTypes.DataFrame, DataTypes.SparseDataFrame):
        return ob.iloc[:, sl]
    elif t == DataTypes.SparseArray:
        if isinstance(ob, scipy.sparse.coo_matrix):
            ob = scipy.sparse.csc_matrix(ob.copy())
        return ob[:, sl]
    else:
        return ob[:, sl]


def verif_encoder_static(klass, enc_kwargs):
    """ does a bunch of static test on an encoder, ie: tests without any data, without fitting anything """

    assert hasattr(klass, "fit")
    assert hasattr(klass, "transform")
    assert hasattr(klass, "fit_transform")

    encoder0 = klass(**enc_kwargs)  # Create an object ...
    encoder1 = clone(encoder0)  # then try to clone it

    encoder2 = klass()  # Create an empty object and then set its params
    encoder2.set_params(**enc_kwargs)

    df1 = pd.DataFrame()

    assert_raise_not_fitted(encoder0, df1)  # Verify that each object isn't fitted
    assert_raise_not_fitted(encoder1, df1)
    assert_raise_not_fitted(encoder2, df1)

    # Verify type are iddentical
    assert type(encoder0) == type(encoder1)
    assert type(encoder0) == type(encoder2)

    # Verify get_params are identical
    params_0 = encoder0.get_params()
    params_1 = encoder1.get_params()
    params_2 = encoder2.get_params()

    rec_assert_equal(params_0, params_1)  # same parameters ...
    rec_assert_equal(params_0, params_2)

    rec_assert_equal({k: v for k, v in params_0.items() if k in enc_kwargs}, enc_kwargs)  # and same as enc_kwargs
    rec_assert_equal({k: v for k, v in params_1.items() if k in enc_kwargs}, enc_kwargs)
    rec_assert_equal({k: v for k, v in params_2.items() if k in enc_kwargs}, enc_kwargs)


def verif_encoder_with_data(klass, enc_kwargs, df1, df2, y1, fit_type, additional_conversion_fun, extended_all_types):
    """ verification of the behavior of a transform on data """
    # Conversion of input into a different type
    df1_conv = convert_generic(df1, output_type=fit_type)
    df2_conv = convert_generic(df2, output_type=fit_type)

    if additional_conversion_fun is not None:
        df1_conv = additional_conversion_fun(df1_conv)
        df2_conv = additional_conversion_fun(df2_conv)

    if y1 is None:
        encoder = klass(**enc_kwargs)
        df1_transformed_a = encoder.fit_transform(df1_conv)  # 1st test without explicity an y..
        df2_transformed_a = encoder.transform(df2_conv)

    encoder_a = klass(**enc_kwargs)
    params_0 = encoder_a.get_params()

    df1_transformed_a = encoder_a.fit_transform(df1_conv, y=y1)  # Other test with an y (might be None or not)
    df2_transformed_a = encoder_a.transform(df2_conv)

    params_3 = encoder_a.get_params()
    # Rmk : might no be enforce ON all transformeurs
    rec_assert_equal(params_0, params_3)  # Verif that get_params didn't change after fit

    assert df1_transformed_a is not None  # verify that something was created
    assert df2_transformed_a is not None  # verify that something was created

    encoder_cloned = clone(encoder_a)  # Clone again ...

    assert_raise_not_fitted(
        encoder_cloned, df2_conv
    )  # ... and verify that the clone isn't fitted, even if encoder_a is fitted

    # Same thing but using ... fit and then... transformed
    encoder_b = klass(**enc_kwargs)
    encoder_b.fit(df1_conv, y=y1)
    df1_transformed_b = encoder_b.transform(df1_conv)
    df2_transformed_b = encoder_b.transform(df2_conv)

    assert df1_transformed_b is not None
    assert df2_transformed_b is not None

    # Same thing but using clone
    encoder_c = clone(encoder_a)
    df1_transformed_c = encoder_c.fit_transform(df1_conv, y=y1)
    df2_transformed_c = encoder_c.transform(df2_conv)

    # Samething but using empyt class + set_params
    encoder_d = klass()
    encoder_d.set_params(**enc_kwargs)
    df1_transformed_d = encoder_d.fit_transform(df1_conv, y=y1)
    df2_transformed_d = encoder_d.transform(df2_conv)

    # Verif that when passed with the wrong number of columns
    assert_raise_value_error(encoder_a, gen_slice(df1_conv, slice(1, None)))
    assert_raise_value_error(encoder_b, gen_slice(df1_conv, slice(1, None)))
    assert_raise_value_error(encoder_c, gen_slice(df1_conv, slice(1, None)))
    assert_raise_value_error(encoder_d, gen_slice(df1_conv, slice(1, None)))

    for fit_type2, additional_conversion_fun2 in extended_all_types:

        if fit_type == fit_type2:
            continue

        df1_conv2 = convert_generic(df1_conv, output_type=fit_type2)

        # Verif that is I have a different type that what was present during the fit I'll raise an error

        assert_raise_value_error(encoder_a, df1_conv2)
        assert_raise_value_error(encoder_b, df1_conv2)
        assert_raise_value_error(encoder_c, df1_conv2)
        assert_raise_value_error(encoder_d, df1_conv2)

    # Verif shape
    # Nb of rows ...
    assert df1_transformed_a.shape[0] == df1_conv.shape[0]
    assert df1_transformed_b.shape[0] == df1_conv.shape[0]
    assert df1_transformed_c.shape[0] == df1_conv.shape[0]
    assert df1_transformed_d.shape[0] == df1_conv.shape[0]

    assert df2_transformed_a.shape[0] == df2_conv.shape[0]
    assert df2_transformed_b.shape[0] == df2_conv.shape[0]
    assert df2_transformed_c.shape[0] == df2_conv.shape[0]
    assert df2_transformed_d.shape[0] == df2_conv.shape[0]

    # Nb of columns : all the same
    assert df1_transformed_b.shape[1] == df1_transformed_a.shape[1]
    assert df1_transformed_c.shape[1] == df1_transformed_a.shape[1]
    assert df1_transformed_d.shape[1] == df1_transformed_a.shape[1]

    assert df2_transformed_a.shape[1] == df1_transformed_a.shape[1]
    assert df2_transformed_b.shape[1] == df1_transformed_a.shape[1]
    assert df2_transformed_c.shape[1] == df1_transformed_a.shape[1]
    assert df2_transformed_d.shape[1] == df1_transformed_a.shape[1]

    # Verif type
    assert get_type(df2_transformed_a) == get_type(df1_transformed_a)

    assert get_type(df1_transformed_b) == get_type(df1_transformed_a)
    assert get_type(df2_transformed_b) == get_type(df1_transformed_a)

    assert get_type(df1_transformed_c) == get_type(df1_transformed_a)
    assert get_type(df2_transformed_c) == get_type(df1_transformed_a)

    assert get_type(df1_transformed_d) == get_type(df1_transformed_a)
    assert get_type(df2_transformed_d) == get_type(df1_transformed_a)

    # if 'desired_output_type' present, check output type is what it seems
    if "desired_output_type" in enc_kwargs:
        assert get_type(df1_transformed_a) == enc_kwargs["desired_output_type"]

    if getattr(encoder_a, "desired_output_type", None) is not None:
        assert get_type(df1_transformed_a) == encoder_a.desired_output_type

    # Verif columns
    if get_type(df1_transformed_b) in (DataTypes.DataFrame, DataTypes.SparseDataFrame):
        assert list(df2_transformed_a.columns) == list(df1_transformed_a.columns)

        assert list(df1_transformed_b.columns) == list(df1_transformed_a.columns)
        assert list(df2_transformed_b.columns) == list(df1_transformed_a.columns)

        assert list(df1_transformed_c.columns) == list(df1_transformed_a.columns)
        assert list(df2_transformed_c.columns) == list(df1_transformed_a.columns)

        assert list(df2_transformed_d.columns) == list(df1_transformed_a.columns)
        assert list(df1_transformed_d.columns) == list(df1_transformed_a.columns)

        assert encoder_a.get_feature_names() == list(df1_transformed_a.columns)
        assert encoder_b.get_feature_names() == list(df1_transformed_a.columns)
        assert encoder_c.get_feature_names() == list(df1_transformed_a.columns)
        assert encoder_d.get_feature_names() == list(df1_transformed_a.columns)

    # Verif index
    if get_type(df1_transformed_b) in (DataTypes.DataFrame, DataTypes.SparseDataFrame):
        assert (df1_transformed_b.index == df1_transformed_a.index).all()
        assert (df2_transformed_b.index == df2_transformed_a.index).all()

        assert (df1_transformed_c.index == df1_transformed_a.index).all()
        assert (df2_transformed_c.index == df2_transformed_a.index).all()

        assert (df1_transformed_d.index == df1_transformed_a.index).all()
        assert (df2_transformed_d.index == df2_transformed_a.index).all()

        if fit_type in (DataTypes.DataFrame, DataTypes.SparseDataFrame):
            assert (df1_transformed_a.index == df1_conv.index).all()
            assert (df2_transformed_a.index == df2_conv.index).all()
