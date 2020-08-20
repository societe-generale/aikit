# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:30:39 2018

@author: Lionel Massoulard
"""
import pytest
import itertools

import pandas as pd
import numpy as np
import scipy.sparse


from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.datasets import make_classification

from aikit.tools.data_structure_helper import generic_hstack

from tests.helpers.testing_help import rec_assert_equal

from aikit.tools.db_informations import get_columns_informations
from aikit.enums import TypeOfVariables, DataTypes
from aikit.tools.data_structure_helper import convert_generic, get_type, _IS_PD1

from aikit.transformers.categories import NumericalEncoder, CategoricalEncoder, OrdinalOneHotEncoder, category_encoders
from aikit.transformers.target import TargetEncoderClassifier, TargetEncoderEntropyClassifier, TargetEncoderRegressor
from aikit.transformers.base import (
    TruncatedSVDWrapper,
    NumImputer,
    FeaturesSelectorClassifier,
    KMeansTransformer,
    CdfScaler,
    PCAWrapper,
)
from aikit.transformers.text import CountVectorizerWrapper, Word2VecVectorizer, Char2VecVectorizer, Word2Vec

from aikit.transformers.model_wrapper import ColumnsSelector
from aikit.tools.helper_functions import pd_match

from aikit.datasets.datasets import load_dataset, DatasetEnum


X_train, y_train, X_test, y_test, _ = load_dataset(DatasetEnum.titanic)

if y_test is None:
    y_test = 1 * (np.random.rand(X_test.shape[0]) > 0.5)

np.random.seed(123)
ii = np.arange(len(X_train))
np.random.shuffle(ii)

X_train_shuffled = X_train.iloc[ii, :]
y_train_shuffled = y_train[ii]

df1 = X_train_shuffled.copy()
df2 = X_test.copy()
y1 = y_train_shuffled.copy()

df1_nona = df1.fillna(value=0).copy()
df2_nona = df2.fillna(value=0).copy()

db_infos = get_columns_informations(X_train)

variable_by_type = {
    t: [c for c, v in db_infos.items() if v["TypeOfVariable"] == t]
    for t in (TypeOfVariables.TEXT, TypeOfVariables.CAT, TypeOfVariables.NUM)
}


ADDITIONAL_CONVERSIONS_FUNCTIONS = {
    DataTypes.SparseArray: (scipy.sparse.coo_matrix, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix)
}
# TODO : maybe try additionnal change of type : int, int32, ...


def _array_equal(m1, m2):
    if m1.shape != m2.shape:
        return False

    m1 = m1.flatten()
    m2 = m2.flatten()

    ii1 = pd.isnull(m1)
    ii2 = pd.isnull(m2)
    if not (ii1 == ii2).all():
        return False
    return (m1[~ii1] == m2[~ii2]).all()


def _array_almost_equal(m1, m2, tolerance):

    if m1.shape != m2.shape:
        return False

    m1 = m1.flatten()
    m2 = m2.flatten()

    ii1 = pd.isnull(m1)
    ii2 = pd.isnull(m2)
    if not (ii1 == ii2).all():
        return False

    return np.abs(m1[~ii1] - m2[~ii2]).max() <= tolerance


def extend_all_type(all_types):
    all_types_conv = []
    for t in all_types:
        all_types_conv.append((t, None))

        other_functions = ADDITIONAL_CONVERSIONS_FUNCTIONS.get(t, None)
        if other_functions is not None:
            for f in other_functions:
                all_types_conv.append((t, f))

    return all_types_conv

def pd1_pd0_type_equivalent(f):
    if _IS_PD1 and f in (DataTypes.DataFrame, DataTypes.SparseDataFrame):
        return DataTypes.DataFrame
    else:
        return f


def verif_conversion(df, dont_test_sparse_array=False):

    if dont_test_sparse_array:
        all_types = (DataTypes.DataFrame, DataTypes.SparseDataFrame, DataTypes.NumpyArray)
        # Sparse array don't work with object
    else:
        all_types = (DataTypes.DataFrame, DataTypes.SparseDataFrame, DataTypes.NumpyArray, DataTypes.SparseArray)

    extended_all_type = extend_all_type(all_types)

    for output_type, additional_conversion_fun in extended_all_type:

        temp_conversion = convert_generic(df, output_type=output_type)
        if additional_conversion_fun is not None:
            temp_conversion = additional_conversion_fun(temp_conversion)

        assert pd1_pd0_type_equivalent(get_type(temp_conversion)) == pd1_pd0_type_equivalent(output_type)  ## Expected type
        assert temp_conversion.shape == df.shape  ## Expected shape

        if output_type in (DataTypes.DataFrame, DataTypes.SparseDataFrame):
            assert list(df.columns) == list(temp_conversion.columns)  ## Didn't change the column
            assert (df.index == temp_conversion.index).all()  ## Didn't change the index

        if output_type == DataTypes.DataFrame:
            assert temp_conversion is df  ## Didn't copy if same type

        # if output_type == DataTypes.NumpyArray:
        #    assert temp_conversion is df.values ## Didn't copy in that case either

        # Convert everything to numpy array and check equality
        temp_conversion2 = convert_generic(temp_conversion, output_type=DataTypes.NumpyArray)
        assert _array_equal(temp_conversion2, df.values)

        # Now convert again, in another type
        for output_type2 in all_types:

            temp_conversion3 = convert_generic(temp_conversion, output_type=output_type2)
            assert pd1_pd0_type_equivalent(get_type(temp_conversion3)) == pd1_pd0_type_equivalent(output_type2)  ## Expected type
            assert temp_conversion3.shape == df.shape  ## Expected shape

            if output_type == output_type2 and output_type:
                if (_IS_PD1 and output_type != DataTypes.SparseDataFrame) or (not _IS_PD1):
                    assert temp_conversion3 is temp_conversion
                    

            test_conversion4 = convert_generic(temp_conversion3, output_type=DataTypes.NumpyArray)
            assert get_type(test_conversion4) == DataTypes.NumpyArray
            # if output_type == DataTypes.SparseDataFrame and output_type2 == DataTypes.SparseArray:
            #    assert _array_almost_equal(test_conversion4, df.values, tolerance=10**(-4))
            # else:
            assert _array_equal(test_conversion4, df.values)


def test_conversion_nan_sparse():
    """ specific test to verify nan handling when converting into sparse """
    xx0 = np.array([[0, 1], [1, np.nan], [0, 0]])
    xx1 = convert_generic(xx0, output_type=DataTypes.SparseArray)
    xx2 = scipy.sparse.csc_matrix(xx1)
    xx3 = convert_generic(xx2, output_type=DataTypes.SparseDataFrame)
    xx4 = convert_generic(xx3, output_type=DataTypes.SparseArray)
    xx5 = convert_generic(xx4, output_type=DataTypes.NumpyArray)

    assert pd.isnull(xx0[1, 1])
    assert pd.isnull(xx1.toarray()[1, 1])
    assert pd.isnull(xx2[1, 1])
    assert pd.isnull(xx3.iloc[1, 1])
    assert pd.isnull(xx4.toarray()[1, 1])
    assert pd.isnull(xx5[1, 1])

    xx = np.array([[0, 1], [1, np.nan], [0, 0]])
    df = pd.DataFrame(xx)
    dfs = convert_generic(df, output_type=DataTypes.SparseDataFrame)
    if _IS_PD1:
        assert pd.isnull(dfs.values[1,1])
    else:
        assert pd.isnull(dfs.to_coo().toarray()[1, 1])
        


@pytest.mark.xfail
def test_to_coo_sparse_matrix_bug():
    """ 'bug' within SparseDataFrame where nan are dropped """
    xx = np.array([[0, 1], [1, np.nan], [0, 0]])

    df1 = pd.SparseDataFrame(xx)
    df2 = pd.SparseDataFrame(xx, default_fill_value=0)

    assert pd.isnull(df1.to_coo().toarray()[1, 1])  # marche pas
    assert pd.isnull(df2.to_coo().toarray()[1, 1])  # marche


def test_conversion():
    verif_conversion(X_train.loc[:, variable_by_type["NUM"]], dont_test_sparse_array=True)
    verif_conversion(X_train_shuffled.loc[:, variable_by_type["NUM"]], dont_test_sparse_array=True)
    verif_conversion(X_test.loc[:, variable_by_type["NUM"]], dont_test_sparse_array=True)

    verif_conversion(X_train.loc[:, variable_by_type["CAT"]], dont_test_sparse_array=True)
    verif_conversion(X_train_shuffled.loc[:, variable_by_type["CAT"]], dont_test_sparse_array=True)
    verif_conversion(X_test.loc[:, variable_by_type["CAT"]], dont_test_sparse_array=True)

    verif_conversion(X_train, dont_test_sparse_array=True)
    verif_conversion(X_train_shuffled, dont_test_sparse_array=True)
    verif_conversion(X_test, dont_test_sparse_array=True)


def test_conversion_sparse_array():
    verif_conversion(X_train.loc[:, variable_by_type["NUM"]], dont_test_sparse_array=False)
    verif_conversion(X_train_shuffled.loc[:, variable_by_type["NUM"]], dont_test_sparse_array=False)
    verif_conversion(X_test.loc[:, variable_by_type["NUM"]], dont_test_sparse_array=False)


# In[] : test concatenation


def verif_generic_hstack(df, dont_test_sparse_array=False, split_index=4):
    """ generic verification function of hstack """

    if dont_test_sparse_array:
        all_types = (DataTypes.DataFrame, DataTypes.SparseDataFrame, DataTypes.NumpyArray)
        # Sparse array don't work with object
    else:
        all_types = (DataTypes.DataFrame, DataTypes.SparseDataFrame, DataTypes.NumpyArray, DataTypes.SparseArray)

    if split_index >= df.shape[1]:
        raise ValueError("split_index should be less than nb of columns")

    df_left = df.iloc[:, :split_index]
    df_right = df.iloc[:, split_index:]

    extended_all_type = extend_all_type(all_types)

    for output_type, additional_conversion_fun in extended_all_type:
        x_left = convert_generic(df_left, output_type=output_type)
        x_right = convert_generic(df_right, output_type=output_type)

        if additional_conversion_fun is not None:
            x_left = additional_conversion_fun(x_left)
            x_right = additional_conversion_fun(x_right)

        x_left_bis = generic_hstack([x_left])
        assert x_left_bis is x_left  # verif no copy

        ### Same type concatenation
        x_left_right = generic_hstack([x_left, x_right])

        assert pd1_pd0_type_equivalent(get_type(x_left_right)) == pd1_pd0_type_equivalent(output_type)
        assert x_left_right.shape == (x_left.shape[0], x_left.shape[1] + x_right.shape[1])

        if output_type in (DataTypes.DataFrame, DataTypes.SparseDataFrame):
            assert list(df.columns) == list(x_left_right.columns)  ## Didn't change the column
            assert (df.index == x_left_right.index).all()  ## Didn't change the index

        ### Verif value
        test_conversion = convert_generic(x_left_right, output_type=DataTypes.NumpyArray)
        assert _array_equal(test_conversion, df.values)

        for output_type2, additional_conversion_fun2 in extended_all_type:

            x_left = convert_generic(df_left, output_type=output_type)
            if additional_conversion_fun is not None:
                x_left = additional_conversion_fun(x_left)

            x_right = convert_generic(df_right, output_type=output_type2)
            if additional_conversion_fun2 is not None:
                x_right = additional_conversion_fun2(x_right)

            x_left_right = generic_hstack([x_left, x_right])  # Let code guess

            if output_type == output_type2:
                assert pd1_pd0_type_equivalent(get_type(x_left_right)) == pd1_pd0_type_equivalent(output_type)

            assert x_left_right.shape == (x_left.shape[0], x_left.shape[1] + x_right.shape[1])
            test_conversion = convert_generic(x_left_right, output_type=DataTypes.NumpyArray)
            assert _array_equal(test_conversion, df.values)

            for output_type3, additional_conversion_fun3 in extended_all_type:
                x_left_right = generic_hstack([x_left, x_right], output_type=output_type3)
                if additional_conversion_fun3 is not None:
                    x_left_right = additional_conversion_fun3(x_left_right)

                assert pd1_pd0_type_equivalent(get_type(x_left_right)) == pd1_pd0_type_equivalent(output_type3)

                assert x_left_right.shape == (x_left.shape[0], x_left.shape[1] + x_right.shape[1])
                test_conversion = convert_generic(x_left_right, output_type=DataTypes.NumpyArray)
                assert _array_equal(test_conversion, df.values)

    # test concat index

    df_left = df_left.copy()
    df_right = df_right.copy()

    df_left.index = range(len(df_left))
    index_right = np.arange(len(df_right))
    np.random.seed(123)
    np.random.shuffle(index_right)
    df_right.index = index_right

    df_left_right = generic_hstack([df_left, df_right])
    assert (df_left_right.index == df_right.index).all()  # keep index of Right

    df_right_left = generic_hstack([df_right, df_right])
    assert (df_right_left.index == df_right.index).all()  # keep index of Right


def test_generic_hstack():
    verif_generic_hstack(X_train.loc[:, variable_by_type["NUM"]], split_index=4, dont_test_sparse_array=True)
    verif_generic_hstack(X_train_shuffled.loc[:, variable_by_type["NUM"]], split_index=4, dont_test_sparse_array=True)
    verif_generic_hstack(X_test.loc[:, variable_by_type["NUM"]], split_index=4, dont_test_sparse_array=True)

    verif_generic_hstack(X_train.loc[:, variable_by_type["CAT"]], dont_test_sparse_array=True, split_index=2)
    verif_generic_hstack(X_train_shuffled.loc[:, variable_by_type["CAT"]], dont_test_sparse_array=True, split_index=2)
    verif_generic_hstack(X_test.loc[:, variable_by_type["CAT"]], dont_test_sparse_array=True, split_index=2)

    verif_generic_hstack(X_train, dont_test_sparse_array=True)
    verif_generic_hstack(X_train_shuffled, dont_test_sparse_array=True)
    verif_generic_hstack(X_test, dont_test_sparse_array=True)


def test_generic_hstack_sparse():
    verif_generic_hstack(X_train.loc[:, variable_by_type["NUM"]], split_index=4, dont_test_sparse_array=False)
    verif_generic_hstack(X_train_shuffled.loc[:, variable_by_type["NUM"]], split_index=4, dont_test_sparse_array=False)
    verif_generic_hstack(X_test.loc[:, variable_by_type["NUM"]], split_index=4, dont_test_sparse_array=False)


# In[] : Categorical Encoder


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


def verif_encoder(
    df1,
    df2,
    y1,
    klass,
    enc_kwargs,
    all_types,
    additional_test_functions=None,
    randomized_transformer=False,
    difference_tolerence=10 ** (-6),
    difference_fit_transform=False,
):
    """ function to test differents things on en encoder """

    if not isinstance(all_types, (list, tuple)):
        all_types = (all_types,)

    if additional_test_functions is None:
        additional_test_functions = []

    # all_types = (DataTypes.DataFrame, DataTypes.SparseDataFrame)
    assert hasattr(klass, "fit")
    assert hasattr(klass, "transform")
    assert hasattr(klass, "fit_transform")

    encoder0 = klass(**enc_kwargs)  # Create an object ...
    encoder1 = clone(encoder0)  # then try to clone it

    encoder2 = klass()  # Create an empty object and then set its params
    encoder2.set_params(**enc_kwargs)

    assert_raise_not_fitted(encoder0, df1)
    assert_raise_not_fitted(encoder1, df1)
    assert_raise_not_fitted(encoder2, df1)

    # Verify type are iddentical
    assert type(encoder0) == type(encoder1)
    assert type(encoder0) == type(encoder2)

    # Verify get_params are identical
    params_0 = encoder0.get_params()
    params_1 = encoder1.get_params()
    params_2 = encoder2.get_params()

    rec_assert_equal(params_0, params_1)
    rec_assert_equal(params_0, params_2)

    rec_assert_equal({k: v for k, v in params_0.items() if k in enc_kwargs}, enc_kwargs)
    rec_assert_equal({k: v for k, v in params_1.items() if k in enc_kwargs}, enc_kwargs)
    rec_assert_equal({k: v for k, v in params_2.items() if k in enc_kwargs}, enc_kwargs)

    #    ###
    #    all_types_bis = []
    #    for fit_type in all_types:
    #        if fit_type == DataTypes.SparseArray:
    #            for additional_conversion_type_step in (scipy.sparse.coo_matrix,scipy.sparse.csc_matrix,scipy
    # all_types_bis += [(fit_type,sp.csr_matric, sp.csc_matrix,sp.coo_matrix)

    extended_all_types = extend_all_type(all_types)

    for fit_type, additional_conversion_fun in extended_all_types:
        print(f"...testing for {fit_type} ... ")
        # Convert inputs into several type ..
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
        df1_transformed_a = encoder_a.fit_transform(df1_conv, y=y1)  # Other test with an y (might be None or not)
        df2_transformed_a = encoder_a.transform(df2_conv)

        # Verify that get_params didn't change after fit
        # Rmk : might not be enforced on all transformers
        params_3 = encoder_a.get_params()
        rec_assert_equal(params_0, params_3)

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

        encoder_d = klass()
        encoder_d.set_params(**enc_kwargs)
        df1_transformed_d = encoder_d.fit_transform(df1_conv, y=y1)
        df2_transformed_d = encoder_d.transform(df2_conv)

        assert_raise_value_error(encoder_a, gen_slice(df1_conv, slice(1, None)))
        assert_raise_value_error(encoder_b, gen_slice(df1_conv, slice(1, None)))
        assert_raise_value_error(encoder_c, gen_slice(df1_conv, slice(1, None)))
        assert_raise_value_error(encoder_d, gen_slice(df1_conv, slice(1, None)))

        for fit_type2, additional_conversion_fun2 in extended_all_types:

            if pd1_pd0_type_equivalent(fit_type) == pd1_pd0_type_equivalent(fit_type2):
                continue

            df1_conv2 = convert_generic(df1_conv, output_type=fit_type2)

            # Verif that is I have a different type that what was present during the fit I'll raise an error

            
            # I don't want to do this test in pandas1 IF 'fit_type2' is a SparseDataFrame  because that type doesn't exist anymore
            assert_raise_value_error(encoder_a, df1_conv2)
            assert_raise_value_error(encoder_b, df1_conv2)
            assert_raise_value_error(encoder_c, df1_conv2)
            assert_raise_value_error(encoder_d, df1_conv2)
        # Verif that when passed with the wrong number of columns

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

        # Verif equality of fit_transform and fit , transform

        if not randomized_transformer:
            #######################
            #### Test equality ####
            #######################
            if not difference_fit_transform:
                # This test equality between a fit THEN transform and or 'fit_transform', sometimes it is not the same thing
                assert _array_equal(
                    convert_generic(df1_transformed_a, output_type=DataTypes.NumpyArray),
                    convert_generic(df1_transformed_b, output_type=DataTypes.NumpyArray),
                )

            assert _array_equal(
                convert_generic(df2_transformed_a, output_type=DataTypes.NumpyArray),
                convert_generic(df2_transformed_b, output_type=DataTypes.NumpyArray),
            )

            assert _array_equal(
                convert_generic(df1_transformed_a, output_type=DataTypes.NumpyArray),
                convert_generic(df1_transformed_c, output_type=DataTypes.NumpyArray),
            )

            assert _array_equal(
                convert_generic(df2_transformed_a, output_type=DataTypes.NumpyArray),
                convert_generic(df2_transformed_c, output_type=DataTypes.NumpyArray),
            )

            assert _array_equal(
                convert_generic(df1_transformed_a, output_type=DataTypes.NumpyArray),
                convert_generic(df1_transformed_d, output_type=DataTypes.NumpyArray),
            )

            assert _array_equal(
                convert_generic(df2_transformed_a, output_type=DataTypes.NumpyArray),
                convert_generic(df2_transformed_d, output_type=DataTypes.NumpyArray),
            )

        else:
            ##################################
            ### Test that matrix are close ###
            ##################################

            # Carefull : random seed should be set for some model

            if not difference_fit_transform:
                m1, m2 = (
                    convert_generic(df1_transformed_a, output_type=DataTypes.NumpyArray),
                    convert_generic(df1_transformed_b, output_type=DataTypes.NumpyArray),
                )
                assert _array_almost_equal(m1, m2, difference_tolerence)
                # assert np.abs(m1-m2).max().max() <= difference_tolerence

            m1, m2 = (
                convert_generic(df2_transformed_a, output_type=DataTypes.NumpyArray),
                convert_generic(df2_transformed_b, output_type=DataTypes.NumpyArray),
            )
            assert _array_almost_equal(m1, m2, difference_tolerence)

            m1, m2 = (
                convert_generic(df1_transformed_a, output_type=DataTypes.NumpyArray),
                convert_generic(df1_transformed_c, output_type=DataTypes.NumpyArray),
            )
            assert _array_almost_equal(m1, m2, difference_tolerence)

            m1, m2 = (
                convert_generic(df2_transformed_a, output_type=DataTypes.NumpyArray),
                convert_generic(df2_transformed_c, output_type=DataTypes.NumpyArray),
            )
            assert _array_almost_equal(m1, m2, difference_tolerence)

            m1, m2 = (
                convert_generic(df1_transformed_a, output_type=DataTypes.NumpyArray),
                convert_generic(df1_transformed_d, output_type=DataTypes.NumpyArray),
            )
            assert _array_almost_equal(m1, m2, difference_tolerence)

            m1, m2 = (
                convert_generic(df2_transformed_a, output_type=DataTypes.NumpyArray),
                convert_generic(df2_transformed_d, output_type=DataTypes.NumpyArray),
            )
            assert _array_almost_equal(m1, m2, difference_tolerence)

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

        for f in additional_test_functions:
            f(df1_transformed_a, df1)
            f(df1_transformed_b, df1)
            f(df1_transformed_c, df1)
            f(df1_transformed_d, df1)

            f(df2_transformed_a, df2)
            f(df2_transformed_b, df2)
            f(df2_transformed_c, df2)
            f(df2_transformed_d, df2)


# In[] : ColumnsSelector

#########################
#### ColumnsSelector ####
#########################

# klass = ColumnsSelector
def other_selector_test1(df_transformed, df):
    assert list(df_transformed.columns) == variable_by_type["CAT"] + variable_by_type["NUM"]


def other_selector_test2(df_transformed, df):
    assert list(df_transformed.columns) == variable_by_type["NUM"]


def test_ColumnsSelector():
    verif_encoder(
        df1=df1,
        df2=df2,
        y1=None,
        klass=ColumnsSelector,
        enc_kwargs={"columns_to_use": variable_by_type["CAT"] + variable_by_type["NUM"]},
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),
        additional_test_functions=[other_selector_test1],
    )

    int_columns_to_use = pd_match(variable_by_type["NUM"], list(X_train_shuffled.columns))

    verif_encoder(
        df1=df1,
        df2=df2,
        y1=None,
        klass=ColumnsSelector,
        enc_kwargs={"columns_to_use": int_columns_to_use},
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),
        additional_test_functions=[other_selector_test2],
    )

    verif_encoder(
        df1=df1,
        df2=df2,
        y1=None,
        klass=ColumnsSelector,
        enc_kwargs={"columns_to_use": int_columns_to_use},
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame, DataTypes.NumpyArray),
    )

    verif_encoder(
        df1=df1,
        df2=df2,
        y1=None,
        klass=ColumnsSelector,
        enc_kwargs={"columns_to_use": None},
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame, DataTypes.NumpyArray),
    )


# In[] : Wrapped Encoder

#########################
#### Wrapped Encoder ####
#########################


def check_all_numerical(df_transformed, df=None):
    mat = convert_generic(df_transformed, output_type=DataTypes.NumpyArray)
    has_error = False
    try:
        mat.astype(np.float64)
    except ValueError:
        has_error = True

    assert not has_error  # So that I explicty raise an assertion


def check_no_null(df_transformed, df=None):
    df_transformed2 = convert_generic(df_transformed, output_type=DataTypes.DataFrame)
    assert df_transformed2.isnull().sum().sum() == 0


def check_between_01(df_transformed, df=None):
    xx = convert_generic(df_transformed, output_type=DataTypes.NumpyArray)
    assert xx.min() >= 0
    assert xx.max() <= 1


def check_positive(df_transformed, df=None):
    xx = convert_generic(df_transformed, output_type=DataTypes.NumpyArray)
    assert xx.min() >= 0


def nb_columns_verify(nb):
    def fun_test(df_transformed, df=None):
        assert df_transformed.shape[1] == nb

    return fun_test


def didnot_change_column_nb(df_transformed, df):
    assert df_transformed.shape[1] == df.shape[1]


def didnot_change_column_names(df_tranformed, df):
    c1 = getattr(df_tranformed, "columns", None)
    c2 = getattr(df, "columns", None)
    if c1 is not None and c2 is not None:
        assert list(c1) == list(c2)


# In[] : NumericalEncoder

###########################
###  Numerical Encoder  ###
###########################


def test_NumericalEncoder_onehot1():

    cols = variable_by_type["NUM"] + variable_by_type["CAT"]
    # One Hot mode #
    verif_encoder(
        df1=df1.loc[:, cols],
        df2=df2.loc[:, cols],
        y1=None,
        klass=NumericalEncoder,
        enc_kwargs={"columns_to_use": variable_by_type["CAT"], "drop_unused_columns": False},
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),
        additional_test_functions=[check_all_numerical],
    )


def test_NumericalEncoder_onehot2():
    verif_encoder(
        df1=df1.loc[:, variable_by_type["NUM"] + variable_by_type["CAT"]],
        df2=df2.loc[:, variable_by_type["NUM"] + variable_by_type["CAT"]],
        y1=None,
        klass=NumericalEncoder,
        enc_kwargs={},
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),  # DataTypes.NumpyArray),
        additional_test_functions=[check_all_numerical],
    )


def test_NumericalEncoder_onehot3():
    verif_encoder(
        df1=df1.loc[:, variable_by_type["NUM"] + variable_by_type["CAT"]],
        df2=df2.loc[:, variable_by_type["NUM"] + variable_by_type["CAT"]],
        y1=y1,
        klass=NumericalEncoder,
        enc_kwargs={"drop_unused_columns": True},
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),  # DataTypes.NumpyArray),
        additional_test_functions=[check_all_numerical, check_no_null, check_between_01],
    )


def test_NumericalEncoder_numerical1():
    # Numerical Mode #
    cols = variable_by_type["NUM"] + variable_by_type["CAT"]
    verif_encoder(
        df1=df1.loc[:, cols],
        df2=df2.loc[:, cols],
        y1=None,
        klass=NumericalEncoder,
        enc_kwargs={"columns_to_use": variable_by_type["CAT"], "encoding_type": "num", "drop_unused_columns": False},
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),
        additional_test_functions=[check_all_numerical],
    )


def test_NumericalEncoder_numerical2():
    verif_encoder(
        df1=df1.loc[:, variable_by_type["NUM"] + variable_by_type["CAT"]],
        df2=df2.loc[:, variable_by_type["NUM"] + variable_by_type["CAT"]],
        y1=None,
        klass=NumericalEncoder,
        enc_kwargs={"encoding_type": "num"},
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),  # DataTypes.NumpyArray),
        additional_test_functions=[check_all_numerical],
    )


def test_NumericalEncoder_numerical3():
    verif_encoder(
        df1=df1.loc[:, variable_by_type["CAT"]],
        df2=df2.loc[:, variable_by_type["CAT"]],
        y1=y1,
        klass=NumericalEncoder,
        enc_kwargs={"columns_to_use": "all", "encoding_type": "num"},
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),  # DataTypes.NumpyArray),
        additional_test_functions=[check_all_numerical, check_no_null, nb_columns_verify(5)],
    )


# In[] : OrdinalOneHotEncoder
def test_OrdinalOneHotEncoder():
    df1_no_nan = df1.loc[:, ["sex","embarked"]].copy()
    df1_no_nan.loc[ df1["embarked"].isnull(), "embarked"] = "C"
    
    df2_no_nan = df2.loc[:, ["sex","embarked"]].copy()
    df2_no_nan.loc[ df2["embarked"].isnull(), "embarked"] = "C"
    
    assert df1_no_nan.isnull().sum().sum() == 0
    assert df2_no_nan.isnull().sum().sum() == 0

    verif_encoder(
        df1=df1_no_nan.loc[:, ["sex","embarked"]],
        df2=df2_no_nan.loc[:, ["sex","embarked"]],
        y1=y1,
        klass=OrdinalOneHotEncoder,
        enc_kwargs={"columns_to_use": "all"},
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),  # DataTypes.NumpyArray),
        additional_test_functions=[check_all_numerical, check_no_null]
    )
    
# In[] : CategoricalEncoder

##########################
### CategoricalEncoder ###
##########################


@pytest.mark.skipif(category_encoders is None, reason="category_encoders is not installed")
@pytest.mark.longtest
@pytest.mark.parametrize("encoding_type", ["dummy", "binary", "basen", "hashing"])
def test_CategoricalEncoder(encoding_type):

    # "helmer","polynomial","sum_coding","backward_coding"
    # ==> marche pas bien change la taille de la solution
    # for encoding_type in ("dummy","binary","basen","hashing","helmer","polynomial","sum_coding","backward_coding"):
    #    for encoding_type in ("dummy","binary","basen","hashing"):
    # for encoding_type in ("backward_coding",):
    enc_kwargs = {"columns_to_use": variable_by_type["NUM"] + variable_by_type["CAT"], "encoding_type": encoding_type}

    verif_encoder(
        df1=df1,
        df2=df2,
        y1=None,
        klass=CategoricalEncoder,
        enc_kwargs=enc_kwargs,
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),
        additional_test_functions=[check_all_numerical],
    )

    verif_encoder(
        df1=df1.loc[:, variable_by_type["NUM"] + variable_by_type["CAT"]],
        df2=df2.loc[:, variable_by_type["NUM"] + variable_by_type["CAT"]],
        y1=None,
        klass=CategoricalEncoder,
        enc_kwargs={"encoding_type": encoding_type},
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),  # DataTypes.NumpyArray),
        additional_test_functions=[check_all_numerical],
    )


# In[]:TargetEncoder

#####################
### TargetEncoder ###
#####################
# import aikit.transformers_target
# reload(aikit.transformers_target)


@pytest.mark.parametrize("cv, noise_level, smoothing_value", list(itertools.product((None, 10), (None, 0.1), (0, 1))))
def test_TargetEncoderClassifier_fails_no_y(cv, noise_level, smoothing_value):

    enc_kwargs = {
        "columns_to_use": variable_by_type["TEXT"] + variable_by_type["CAT"],
        "cv": cv,
        "noise_level": noise_level,
        "smoothing_value": smoothing_value,
    }
    model = TargetEncoderClassifier(**enc_kwargs)
    with pytest.raises(ValueError):
        model.fit(df1)  # raise because no target


@pytest.mark.longtest
@pytest.mark.parametrize("cv, noise_level, smoothing_value", list(itertools.product((None, 10), (None, 0.1), (0, 1))))
def test_TargetEncoderClassifier1(cv, noise_level, smoothing_value):

    enc_kwargs1 = {
        "columns_to_use": variable_by_type["TEXT"] + variable_by_type["CAT"],
        "cv": cv,
        "noise_level": noise_level,
        "smoothing_value": smoothing_value,
    }

    verif_encoder(
        df1=df1,
        df2=df2,
        y1=y1,
        klass=TargetEncoderClassifier,
        enc_kwargs=enc_kwargs1,
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),
        additional_test_functions=[check_all_numerical],
        randomized_transformer=noise_level is not None,
        difference_tolerence=1.0,
        difference_fit_transform=(noise_level is not None) or (cv is not None),
    )


@pytest.mark.longtest
@pytest.mark.parametrize("cv, noise_level, smoothing_value", list(itertools.product((None, 10), (None, 0.1), (0, 1))))
def test_TargetEncoderClassifier2(cv, noise_level, smoothing_value):

    enc_kwargs2 = {"cv": cv, "noise_level": noise_level, "smoothing_value": smoothing_value, "columns_to_use":"object"}

    verif_encoder(
        df1=df1.loc[:, variable_by_type["TEXT"] + variable_by_type["CAT"]],
        df2=df2.loc[:, variable_by_type["TEXT"] + variable_by_type["CAT"]],
        y1=y1,
        klass=TargetEncoderClassifier,
        enc_kwargs=enc_kwargs2,
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),  # DataTypes.NumpyArray),
        additional_test_functions=[check_all_numerical],
        randomized_transformer=noise_level is not None,
        difference_tolerence=1.0,
        difference_fit_transform=(noise_level is not None) or (cv is not None)
        # Rmk : if there is noise or cv isn't not there will be difference between fit THEN transform and fit_transform
    )


@pytest.mark.longtest
@pytest.mark.parametrize("cv, noise_level, smoothing_value", list(itertools.product((None, 10), (None, 0.1), (0, 1))))
def test_TargetEncoderClassifier3(cv, noise_level, smoothing_value):

    enc_kwargs3 = {"cv": cv,
                   "noise_level":noise_level,
                   "smoothing_value": smoothing_value,
                   "drop_unused_columns": True,
                   "columns_to_use":"all"
                   }


    nb_cols = len(variable_by_type["TEXT"] + variable_by_type["CAT"])
    verif_encoder(
        df1=df1.loc[:, variable_by_type["TEXT"] + variable_by_type["CAT"]],
        df2=df2.loc[:, variable_by_type["TEXT"] + variable_by_type["CAT"]],
        y1=y1,
        klass=TargetEncoderClassifier,
        enc_kwargs=enc_kwargs3,
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),  # DataTypes.NumpyArray),
        additional_test_functions=[check_all_numerical, check_no_null, check_between_01, nb_columns_verify(nb_cols)],
        randomized_transformer=noise_level is not None,
        difference_tolerence=1.0,
        difference_fit_transform=(noise_level is not None) or (cv is not None)
        # Rmk : if there is noise or cv isn't not there will be difference between fit THEN transform and fit_transform
    )


@pytest.mark.longtest
@pytest.mark.parametrize("cv, noise_level, smoothing_value", list(itertools.product((None, 10), (None, 0.1), (0, 1))))
def test_TargetEncoderClassifierEntropy1(cv, noise_level, smoothing_value):

    enc_kwargs1 = {
        "columns_to_use": variable_by_type["TEXT"] + variable_by_type["CAT"],
        "cv": cv,
        "noise_level": noise_level,
        "smoothing_value": smoothing_value,
    }

    verif_encoder(
        df1=df1,
        df2=df2,
        y1=y1,
        klass=TargetEncoderEntropyClassifier,
        enc_kwargs=enc_kwargs1,
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),
        additional_test_functions=[check_all_numerical],
        randomized_transformer=noise_level is not None,
        difference_tolerence=1.0,
        difference_fit_transform=(noise_level is not None) or (cv is not None),
    )


@pytest.mark.longtest
@pytest.mark.parametrize("cv, noise_level, smoothing_value", list(itertools.product((None, 10), (None, 0.1), (0, 1))))
def test_TargetEncoderClassifierEntropy2(cv, noise_level, smoothing_value):
    enc_kwargs2 = {"cv": cv, "noise_level": noise_level, "smoothing_value": smoothing_value, "columns_to_use":"object"}
    verif_encoder(
        df1=df1.loc[:, variable_by_type["TEXT"] + variable_by_type["CAT"]],
        df2=df2.loc[:, variable_by_type["TEXT"] + variable_by_type["CAT"]],
        y1=y1,
        klass=TargetEncoderEntropyClassifier,
        enc_kwargs=enc_kwargs2,
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),  # DataTypes.NumpyArray),
        additional_test_functions=[check_all_numerical],
        randomized_transformer=noise_level is not None,
        difference_tolerence=1.0,
        difference_fit_transform=(noise_level is not None) or (cv is not None)
        # Rmk : if there is noise or cv isn't not there will be difference between fit THEN transform and fit_transform
    )


@pytest.mark.longtest
@pytest.mark.parametrize("cv, noise_level, smoothing_value", list(itertools.product((None, 10), (None, 0.1), (0, 1))))
def test_TargetEncoderClassifierEntropy3(cv, noise_level, smoothing_value):
    enc_kwargs3 = {"cv": cv, "noise_level": noise_level, "smoothing_value": smoothing_value, "columns_to_use":"all", "drop_unused_columns":True}

    nb_cols = len(variable_by_type["TEXT"] + variable_by_type["CAT"])
    verif_encoder(
        df1=df1.loc[:, variable_by_type["TEXT"] + variable_by_type["CAT"]],
        df2=df2.loc[:, variable_by_type["TEXT"] + variable_by_type["CAT"]],
        y1=y1,
        klass=TargetEncoderEntropyClassifier,
        enc_kwargs=enc_kwargs3,
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),  # DataTypes.NumpyArray),
        additional_test_functions=[check_all_numerical, check_no_null, check_positive, nb_columns_verify(nb_cols)],
        randomized_transformer=noise_level is not None,
        difference_tolerence=1.0,
        difference_fit_transform=(noise_level is not None) or (cv is not None)
        # Rmk : if there is noise or cv isn't not there will be difference between fit THEN transform and fit_transform
    )


def verif_TargetEncoderClassifierEntropy():
    """ to manually run the tests if needed """
    for cv in (None, 10):

        for noise_level in (None, 0.1):

            for smoothing_value in (0, 1):

                test_TargetEncoderClassifierEntropy1(cv=cv, noise_level=noise_level, smoothing_value=smoothing_value)

                test_TargetEncoderClassifierEntropy2(cv=cv, noise_level=noise_level, smoothing_value=smoothing_value)

                test_TargetEncoderClassifierEntropy3(cv=cv, noise_level=noise_level, smoothing_value=smoothing_value)


@pytest.mark.longtest
@pytest.mark.parametrize("cv, noise_level, smoothing_value", list(itertools.product((None, 10), (None, 0.1), (0, 1))))
def test_TargetEncoderRegressor1(cv, noise_level, smoothing_value):

    np.random.seed(123)
    y1_num = y1 + np.random.randn(len(y1))

    enc_kwargs1 = {
        "columns_to_use": variable_by_type["TEXT"] + variable_by_type["CAT"],
        "cv": cv,
        "noise_level": noise_level,
        "smoothing_value": smoothing_value,
    }

    randomized_transformer = True  # small numerical difference ...
    if noise_level is not None:
        difference_tolerence = np.max(
            np.abs(y1_num)
        )  # big difference : it is randomized so we can't expected result to be the same
    else:
        difference_tolerence = 10 ** (-6)

    verif_encoder(
        df1=df1,
        df2=df2,
        y1=y1_num,
        klass=TargetEncoderRegressor,
        enc_kwargs=enc_kwargs1,
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),
        additional_test_functions=[check_all_numerical],
        randomized_transformer=randomized_transformer,  # noise_level is not None ,
        difference_tolerence=difference_tolerence,  # 0.5,
        difference_fit_transform=(noise_level is not None) or (cv is not None),
    )


@pytest.mark.longtest
@pytest.mark.parametrize("cv, noise_level, smoothing_value", list(itertools.product((None, 10), (None, 0.1), (0, 1))))
def test_TargetEncoderRegressor2(cv, noise_level, smoothing_value):

    np.random.seed(123)
    y1_num = y1 + np.random.randn(len(y1))

    enc_kwargs2 = {"cv": cv, "noise_level": noise_level, "smoothing_value": smoothing_value}

    randomized_transformer = True  # small numerical difference ...
    if noise_level is not None:
        difference_tolerence = np.max(np.abs(y1_num))  # big difference
    else:
        difference_tolerence = 10 ** (-6)

    verif_encoder(
        df1=df1.loc[:, variable_by_type["NUM"] + variable_by_type["CAT"]],
        df2=df2.loc[:, variable_by_type["NUM"] + variable_by_type["CAT"]],
        y1=y1_num,
        klass=TargetEncoderRegressor,
        enc_kwargs=enc_kwargs2,
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),  # DataTypes.NumpyArray),
        additional_test_functions=[check_all_numerical],
        randomized_transformer=randomized_transformer,  # noise_level is not None ,
        difference_tolerence=difference_tolerence,  # 0.5,
        difference_fit_transform=(noise_level is not None) or (cv is not None)
        # Rmk : if there is noise or cv isn't not there will be difference between fit THEN transform and fit_transform
    )


@pytest.mark.longtest
@pytest.mark.parametrize("cv, noise_level, smoothing_value", list(itertools.product((None, 10), (None, 0.1), (0, 1))))
def test_TargetEncoderRegressor3(cv, noise_level, smoothing_value):

    np.random.seed(123)
    y1_num = y1 + np.random.randn(len(y1))

    enc_kwargs3 = {"cv": cv, "noise_level": noise_level, "smoothing_value": smoothing_value, "drop_unused_columns": True}

    randomized_transformer = True  # small numerical difference ...
    if noise_level is not None:
        difference_tolerence = np.max(np.abs(y1_num))  # big difference
    else:
        difference_tolerence = 10 ** (-6)

    verif_encoder(
        df1=df1.loc[:, variable_by_type["NUM"] + variable_by_type["CAT"]],
        df2=df2.loc[:, variable_by_type["NUM"] + variable_by_type["CAT"]],
        y1=y1_num,
        klass=TargetEncoderRegressor,
        enc_kwargs=enc_kwargs3,
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),  # DataTypes.NumpyArray),
        additional_test_functions=[check_all_numerical, check_no_null, nb_columns_verify(len(variable_by_type["CAT"]))],
        randomized_transformer=randomized_transformer,  # noise_level is not None ,
        difference_tolerence=difference_tolerence,  # 0.5,
        difference_fit_transform=(noise_level is not None) or (cv is not None)
        # Rmk : if there is noise or cv isn't not there will be difference between fit THEN transform and fit_transform
    )


# In[] CountVetorizer

########################
#### CountVetorizer ####
########################


def test_CountVectorizerWrapper():
    verif_encoder(
        df1=df1.loc[:, variable_by_type["TEXT"]],
        df2=df2.loc[:, variable_by_type["TEXT"]],
        y1=None,
        klass=CountVectorizerWrapper,
        enc_kwargs={},
        all_types=(DataTypes.DataFrame,),
        additional_test_functions=[check_all_numerical, check_positive, check_no_null],
    )


if _IS_PD1:
    OUTPUT_TO_TEST = [None, DataTypes.DataFrame, DataTypes.NumpyArray, DataTypes.SparseArray]
else:
    OUTPUT_TO_TEST = [None, DataTypes.DataFrame, DataTypes.NumpyArray, DataTypes.SparseArray,
                      DataTypes.SparseDataFrame],

@pytest.mark.longtest
@pytest.mark.parametrize(
    "desired_output_type",
    OUTPUT_TO_TEST
)
def test_CountVectorizerWrapper1(desired_output_type):
    enc_kwargs = {"columns_to_use": variable_by_type["TEXT"]}
    if desired_output_type is not None:
        enc_kwargs["desired_output_type"] = desired_output_type

    verif_encoder(
        df1=df1,
        df2=df2,
        y1=None,
        klass=CountVectorizerWrapper,
        enc_kwargs=enc_kwargs,
        all_types=(DataTypes.DataFrame,),
        additional_test_functions=[check_all_numerical, check_positive, check_no_null],
    )


@pytest.mark.longtest
@pytest.mark.parametrize(
    "desired_output_type",
    OUTPUT_TO_TEST
)
def test_CountVectorizerWrapper1_tfidf(desired_output_type):
    enc_kwargs = {"columns_to_use": variable_by_type["TEXT"]}
    if desired_output_type is not None:
        enc_kwargs["desired_output_type"] = desired_output_type
    enc_kwargs["tfidf"] = True

    verif_encoder(
        df1=df1,
        df2=df2,
        y1=None,
        klass=CountVectorizerWrapper,
        enc_kwargs=enc_kwargs,
        all_types=(DataTypes.DataFrame,),
        additional_test_functions=[check_all_numerical, check_positive, check_no_null],
        difference_fit_transform=True,  # should be any difference but I have small numerical errors ?
    )


def verif_CountVectorizerWrapper():
    """ to run the test manually if needed """
    for desired_output_type in (
        None,
        DataTypes.DataFrame,
        DataTypes.NumpyArray,
        DataTypes.SparseArray,
        DataTypes.SparseDataFrame,
    ):
        test_CountVectorizerWrapper1(desired_output_type=desired_output_type)

    for desired_output_type in (
        None,
        DataTypes.DataFrame,
        DataTypes.NumpyArray,
        DataTypes.SparseArray,
        DataTypes.SparseDataFrame,
    ):
        test_CountVectorizerWrapper1_tfidf(desired_output_type=desired_output_type)

    test_CountVectorizerWrapper()


# In[] : Test Word2vec


@pytest.mark.skipif(Word2Vec is None, reason="gensim isn't installed")
@pytest.mark.longtest
@pytest.mark.parametrize(
    "text_preprocess, same_embedding_all_columns, size, window",
    itertools.product((None, "default", "nltk", "digit", "default"), (True, False), (50, 100), (3, 5)),
)
def test_Word2VecVectorizer(text_preprocess, same_embedding_all_columns, size, window):
    Xtrain = load_dataset("titanic")[0]
    df1 = Xtrain.loc[0:600, :]
    df2 = Xtrain.loc[600:, :]

    enc_kwargs = {"columns_to_use": ["name", "ticket"]}
    enc_kwargs["text_preprocess"] = text_preprocess
    enc_kwargs["same_embedding_all_columns"] = same_embedding_all_columns
    enc_kwargs["size"] = size
    enc_kwargs["window"] = window
    enc_kwargs["random_state"] = 123

    verif_encoder(
        df1=df1,
        df2=df2,
        y1=None,
        klass=Word2VecVectorizer,
        enc_kwargs=enc_kwargs,
        all_types=(DataTypes.DataFrame,),
        additional_test_functions=[check_all_numerical, check_no_null, nb_columns_verify(size * 2)],
    )


# In[] Test Char2Vec


@pytest.mark.skipif(Word2Vec is None, reason="gensim isn't installed")
@pytest.mark.longtest
@pytest.mark.parametrize(
    "text_preprocess, same_embedding_all_columns, size, window",
    itertools.product((None, "default", "digit", "default"), (True, False), (50, 100), (3, 5)),
)
def test_Char2VecVectorizer(text_preprocess, same_embedding_all_columns, size, window):
    Xtrain = load_dataset("titanic")[0]
    df1 = Xtrain.loc[0:600, :]
    df2 = Xtrain.loc[600:, :]

    enc_kwargs = {"columns_to_use": ["name", "ticket"]}
    enc_kwargs["text_preprocess"] = text_preprocess
    enc_kwargs["same_embedding_all_columns"] = same_embedding_all_columns
    enc_kwargs["size"] = size
    enc_kwargs["window"] = window
    enc_kwargs["random_state"] = 123

    verif_encoder(
        df1=df1,
        df2=df2,
        y1=None,
        klass=Char2VecVectorizer,
        enc_kwargs=enc_kwargs,
        all_types=(DataTypes.DataFrame,),
        additional_test_functions=[check_all_numerical, check_no_null, nb_columns_verify(size * 2)],
        randomized_transformer=True,
        difference_tolerence=0.1,
    )


# In[] : TruncatedSVD

######################
#### TruncatedSVD ####
######################


def type_verifier(allowed_types):
    if not isinstance(allowed_types, (tuple, list)):
        allowed_types = (allowed_types,)

    def fun_test(df_transformed, df=None):
        return get_type(df_transformed) in allowed_types

    return fun_test


def test_TruncatedSVDWrapper1():

    verif_encoder(
        df1=df1_nona,
        df2=df2_nona,
        y1=None,
        klass=TruncatedSVDWrapper,
        enc_kwargs={"columns_to_use": variable_by_type["NUM"], "n_components": 3},
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame) if _IS_PD1 else (DataTypes.DataFrame, ),
        additional_test_functions=[
            check_all_numerical,
            check_no_null,
            nb_columns_verify(3),
            type_verifier(DataTypes.DataFrame),
        ],
        randomized_transformer=True,
        difference_tolerence=10 ** (-6),
        difference_fit_transform=False,
    )


def test_TruncatedSVDWrapper2():

    verif_encoder(
        df1=df1_nona.loc[:, variable_by_type["NUM"]],
        df2=df2_nona.loc[:, variable_by_type["NUM"]],
        y1=None,
        klass=TruncatedSVDWrapper,
        enc_kwargs={"n_components": 3},
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame, DataTypes.NumpyArray, DataTypes.SparseArray) if _IS_PD1 else (DataTypes.DataFrame, DataTypes.NumpyArray, DataTypes.SparseArray),
        additional_test_functions=[
            check_all_numerical,
            check_no_null,
            nb_columns_verify(3),
            type_verifier(DataTypes.DataFrame),
        ],
        randomized_transformer=True,
        difference_tolerence=10 ** (-6),
        difference_fit_transform=False,
    )


def test_TruncatedSVDWrapper3():

    verif_encoder(
        df1=df1_nona,
        df2=df2_nona,
        y1=None,
        klass=TruncatedSVDWrapper,
        enc_kwargs={"columns_to_use": variable_by_type["NUM"]},
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame) if _IS_PD1 else (DataTypes.DataFrame, ),
        additional_test_functions=[check_all_numerical, check_no_null],
        randomized_transformer=True,
        difference_tolerence=10 ** (-6),
    )


def test_TruncatedSVDWrapper4():

    verif_encoder(
        df1=df1_nona.loc[:, variable_by_type["NUM"]],
        df2=df2_nona.loc[:, variable_by_type["NUM"]],
        y1=None,
        klass=TruncatedSVDWrapper,
        enc_kwargs={},
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame, DataTypes.NumpyArray, DataTypes.SparseArray) if _IS_PD1 else (DataTypes.DataFrame, DataTypes.NumpyArray, DataTypes.SparseArray),
        additional_test_functions=[check_all_numerical, check_no_null],
        randomized_transformer=True,
        difference_tolerence=10 ** (-6),
        difference_fit_transform=False,
    )


def verif_TruncatedSVDWrapper():
    """ to run the tests manually if needed """
    test_TruncatedSVDWrapper1()
    test_TruncatedSVDWrapper2()
    test_TruncatedSVDWrapper3()
    test_TruncatedSVDWrapper4()


# In[] : PCA


def test_PCAWrapper1():

    verif_encoder(
        df1=df1_nona,
        df2=df2_nona,
        y1=None,
        klass=PCAWrapper,
        enc_kwargs={"columns_to_use": variable_by_type["NUM"], "n_components": 3},
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),
        additional_test_functions=[
            check_all_numerical,
            check_no_null,
            nb_columns_verify(3),
            type_verifier(DataTypes.DataFrame),
        ],
        randomized_transformer=True,
        difference_tolerence=10 ** (-6),
        difference_fit_transform=False,
    )


def test_PCAWrapper2():

    verif_encoder(
        df1=df1_nona.loc[:, variable_by_type["NUM"]],
        df2=df2_nona.loc[:, variable_by_type["NUM"]],
        y1=None,
        klass=PCAWrapper,
        enc_kwargs={"n_components": 3},
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame, DataTypes.NumpyArray, DataTypes.SparseArray),
        additional_test_functions=[
            check_all_numerical,
            check_no_null,
            nb_columns_verify(3),
            type_verifier(DataTypes.DataFrame),
        ],
        randomized_transformer=True,
        difference_tolerence=10 ** (-6),
        difference_fit_transform=False,
    )


def test_PCAWrapper3():

    verif_encoder(
        df1=df1_nona,
        df2=df2_nona,
        y1=None,
        klass=PCAWrapper,
        enc_kwargs={"columns_to_use": variable_by_type["NUM"]},
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame),
        additional_test_functions=[check_all_numerical, check_no_null],
        randomized_transformer=True,
        difference_tolerence=10 ** (-6),
    )


def test_PCAWrapper4():

    verif_encoder(
        df1=df1_nona.loc[:, variable_by_type["NUM"]],
        df2=df2_nona.loc[:, variable_by_type["NUM"]],
        y1=None,
        klass=PCAWrapper,
        enc_kwargs={},
        all_types=(DataTypes.DataFrame, DataTypes.SparseDataFrame, DataTypes.NumpyArray, DataTypes.SparseArray),
        additional_test_functions=[check_all_numerical, check_no_null],
        randomized_transformer=True,
        difference_tolerence=10 ** (-6),
        difference_fit_transform=False,
    )


# In[] : Inputer

#################
#### Imputer ####
#################
# import aikit.transformers
# reload(aikit.transformers)


def test_NumImputer1():
    verif_encoder(
        df1=df1,
        df2=df2,
        y1=None,
        klass=NumImputer,
        enc_kwargs={"columns_to_use": variable_by_type["NUM"]},
        all_types=(DataTypes.DataFrame,),
        additional_test_functions=[check_all_numerical, check_no_null],
        randomized_transformer=False,
    )


def test_NumImputer2():
    verif_encoder(
        df1=df1.loc[:, variable_by_type["NUM"]],
        df2=df2.loc[:, variable_by_type["NUM"]],
        y1=None,
        klass=NumImputer,
        enc_kwargs={},
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray, DataTypes.SparseArray),
        additional_test_functions=[check_all_numerical, check_no_null],
        randomized_transformer=False,
    )


@pytest.mark.parametrize("strategy, add_is_null", list(itertools.product(("mean", "median", "fix"), (True, False))))
def test_NumImputer3(strategy, add_is_null):
    enc_kwargs = {"columns_to_use": variable_by_type["NUM"], "strategy": strategy, "add_is_null": add_is_null}

    verif_encoder(
        df1=df1,
        df2=df2,
        y1=None,
        klass=NumImputer,
        enc_kwargs=enc_kwargs,
        all_types=(DataTypes.DataFrame,),
        additional_test_functions=[check_all_numerical, check_no_null],
        randomized_transformer=False,
    )


@pytest.mark.parametrize("strategy, add_is_null", list(itertools.product(("mean", "median", "fix"), (True, False))))
def test_NumImputer4(strategy, add_is_null):
    enc_kwargs2 = {"strategy": strategy, "add_is_null": add_is_null}

    verif_encoder(
        df1=df1.loc[:, variable_by_type["NUM"]],
        df2=df2.loc[:, variable_by_type["NUM"]],
        y1=None,
        klass=NumImputer,
        enc_kwargs=enc_kwargs2,
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray, DataTypes.SparseArray),
        additional_test_functions=[check_all_numerical, check_no_null],
        randomized_transformer=False,
    )


def verif_NumImputer():
    test_NumImputer1()
    test_NumImputer2()

    for strategy in ("mean", "median", "fix"):
        for add_is_null in (True, False):

            test_NumImputer3(strategy=strategy, add_is_null=add_is_null)
            test_NumImputer4(strategy=strategy, add_is_null=add_is_null)


# In[]
## With data with NaN
@pytest.mark.parametrize("strategy, add_is_null", list(itertools.product(("mean", "median", "fix"), (True, False))))
def test_NumImputer_withnan1(strategy, add_is_null):

    np.random.seed(123)
    X1_nonan = np.random.randn(1000, 20)
    X2_nonna = np.random.randn(1000, 20)

    X1_nan = X1_nonan.copy()
    X2_nan = X2_nonna.copy()

    # Nans in columns 0
    X1_nan[0, 0] = np.nan
    X1_nan[10, 0] = np.nan

    X2_nan[1, 0] = np.nan
    X2_nan[5, 0] = np.nan

    X1_nan = pd.DataFrame(X1_nan, columns=["col%d" % d for d in range(X1_nan.shape[1])])
    X2_nan = pd.DataFrame(X2_nan, columns=["col%d" % d for d in range(X2_nan.shape[1])])
    X1_nonan = pd.DataFrame(X1_nonan, columns=["col%d" % d for d in range(X1_nonan.shape[1])])
    X2_nonna = pd.DataFrame(X2_nonna, columns=["col%d" % d for d in range(X2_nonna.shape[1])])

    enc_kwargs = {"strategy": strategy, "add_is_null": add_is_null}

    verif_encoder(
        df1=X1_nonan,
        df2=X2_nonna,
        y1=None,
        klass=NumImputer,
        enc_kwargs=enc_kwargs,
        all_types=(DataTypes.DataFrame, DataTypes.SparseArray, DataTypes.NumpyArray),
        additional_test_functions=[check_all_numerical, check_no_null, nb_columns_verify(X1_nonan.shape[1])],
        randomized_transformer=False,
    )


@pytest.mark.parametrize("strategy, add_is_null", list(itertools.product(("mean", "median", "fix"), (True, False))))
def test_NumImputer_withnan2(strategy, add_is_null):

    np.random.seed(123)
    X1_nonan = np.random.randn(1000, 20)
    X2_nonna = np.random.randn(1000, 20)

    X1_nan = X1_nonan.copy()
    X2_nan = X2_nonna.copy()

    # Nans in columns 0
    X1_nan[0, 0] = np.nan
    X1_nan[10, 0] = np.nan

    X2_nan[1, 0] = np.nan
    X2_nan[5, 0] = np.nan

    X1_nan = pd.DataFrame(X1_nan, columns=["col%d" % d for d in range(X1_nan.shape[1])])
    X2_nan = pd.DataFrame(X2_nan, columns=["col%d" % d for d in range(X2_nan.shape[1])])
    X1_nonan = pd.DataFrame(X1_nonan, columns=["col%d" % d for d in range(X1_nonan.shape[1])])
    X2_nonna = pd.DataFrame(X2_nonna, columns=["col%d" % d for d in range(X2_nonna.shape[1])])

    enc_kwargs = {"strategy": strategy, "add_is_null": add_is_null}

    verif_encoder(
        df1=X1_nan,
        df2=X2_nan,
        y1=None,
        klass=NumImputer,
        enc_kwargs=enc_kwargs,
        all_types=(DataTypes.DataFrame, DataTypes.SparseArray, DataTypes.NumpyArray),
        additional_test_functions=[
            check_all_numerical,
            check_no_null,
            nb_columns_verify(X1_nonan.shape[1] + 1 * add_is_null),
        ],
        randomized_transformer=False,
    )


def verif_NumImputer_withnan():
    """ to manually run the tests if needed """

    for strategy in ("mean", "median", "fix"):
        for add_is_null in (True, False):

            test_NumImputer_withnan1(strategy=strategy, add_is_null=add_is_null)

            test_NumImputer_withnan2(strategy=strategy, add_is_null=add_is_null)


# In[] : Feature Selector
########################
### Feature Selector ###
########################


def test_FeaturesSelectorClassifier1():

    verif_encoder(
        df1=df1_nona.loc[:, variable_by_type["NUM"]],
        df2=df2_nona.loc[:, variable_by_type["NUM"]],
        y1=y_train_shuffled,
        klass=FeaturesSelectorClassifier,
        enc_kwargs={"model_params": {"random_state": 123}},
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),
        additional_test_functions=[check_all_numerical, check_no_null],
        randomized_transformer=False,
    )


def test_FeaturesSelectorClassifier2():
    verif_encoder(
        df1=df1_nona.loc[:, variable_by_type["NUM"]],
        df2=df2_nona.loc[:, variable_by_type["NUM"]],
        y1=y_train_shuffled,
        klass=FeaturesSelectorClassifier,
        enc_kwargs={"n_components": 2, "model_params": {"random_state": 123}},
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),
        additional_test_functions=[check_all_numerical, check_no_null, nb_columns_verify(2)],
        randomized_transformer=False,
    )


def test_FeaturesSelectorClassifier3():
    x_sk, y_sk = make_classification(n_samples=1000, n_features=20, n_informative=5, random_state=123)

    verif_encoder(
        df1=x_sk,
        df2=x_sk,
        y1=y_sk,
        klass=FeaturesSelectorClassifier,
        enc_kwargs={"n_components": 2, "model_params": {"random_state": 123}},
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),
        additional_test_functions=[check_all_numerical, check_no_null, nb_columns_verify(2)],
        randomized_transformer=False,
    )


@pytest.mark.parametrize("selector_type", ["forest", "linear", "default"])
def test_FeaturesSelectorClassifier4(selector_type):

    enc_kwargs = {"n_components": 2, "selector_type": selector_type, "model_params": {"random_state": 123}}

    if selector_type == "forest":
        enc_kwargs["model_params"]["n_estimators"] = 100

    verif_encoder(
        df1=df1_nona.loc[:, variable_by_type["NUM"]],
        df2=df2_nona.loc[:, variable_by_type["NUM"]],
        y1=y_train_shuffled,
        klass=FeaturesSelectorClassifier,
        enc_kwargs=enc_kwargs,
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),
        additional_test_functions=[check_all_numerical, check_no_null, nb_columns_verify(2)],
        randomized_transformer=False,
    )


@pytest.mark.parametrize("selector_type", ["forest", "linear", "default"])
def test_FeaturesSelectorClassifier5(selector_type):

    enc_kwargs = {"n_components": 10, "selector_type": selector_type, "model_params": {"random_state": 123}}
    if selector_type == "forest":
        enc_kwargs["model_params"]["n_estimators"] = 100

    x_sk, y_sk = make_classification(n_samples=1000, n_features=20, n_informative=5, random_state=123)

    verif_encoder(
        df1=x_sk,
        df2=x_sk,
        y1=y_sk,
        klass=FeaturesSelectorClassifier,
        enc_kwargs=enc_kwargs,
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),
        additional_test_functions=[check_all_numerical, check_no_null, nb_columns_verify(10)],
        randomized_transformer=False,
    )


def verify_columns(df_transformed, df=None):
    inf_columns_kept = len([c for c in df_transformed.columns if "inf" in c])
    assert inf_columns_kept >= 10


@pytest.mark.parametrize(
    "selector_type", ["forest"]
)  # avec cette base les autres selector ne selectionne pas bien les bonnes datas
def test_FeaturesSelectorClassifier_verify_columns(selector_type):
    X, y = make_classification(n_samples=10000, n_features=100, n_informative=20, shuffle=False, random_state=123)
    X = pd.DataFrame(X, columns=["inf_%d" % d for d in range(10)] + ["noise_%d" % d for d in range(90)])
    np.random.seed(123)
    jj = np.arange(100)
    np.random.shuffle(jj)
    X = X.iloc[:, jj]

    enc_kwargs = {"n_components": 20, "selector_type": selector_type, "model_params": {"random_state": 123}}
    if selector_type == "forest":
        enc_kwargs["model_params"]["n_estimators"] = 100

    verif_encoder(
        df1=X,
        df2=X,
        y1=y,
        klass=FeaturesSelectorClassifier,
        enc_kwargs={"n_components": 20, "selector_type": selector_type, "model_params": {"random_state": 123}},
        all_types=(DataTypes.DataFrame,),
        additional_test_functions=[check_all_numerical, check_no_null, nb_columns_verify(20), verify_columns],
        randomized_transformer=False,
    )


# In[]:

#############################
###  KMeansTransformer  ###
#############################


def check_row_sum1(df_tranformed, df=None):
    mat = convert_generic(df_tranformed, output_type=DataTypes.NumpyArray)
    assert np.abs(mat.sum(axis=1) - 1).max() <= 10 ** (-6)


def test_KMeansTransformer1():
    verif_encoder(
        df1=df1_nona.loc[:, variable_by_type["NUM"]],
        df2=df2_nona.loc[:, variable_by_type["NUM"]],
        y1=y_train_shuffled,
        klass=KMeansTransformer,
        enc_kwargs={"random_state": 123, "n_clusters": 10, "result_type": "probability", "kmeans_other_params":{"n_init":1}},
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray, DataTypes.SparseArray),
        additional_test_functions=[
            check_all_numerical,
            check_no_null,
            check_between_01,
            nb_columns_verify(10),
            type_verifier(DataTypes.DataFrame),
        ],
        randomized_transformer=True,
    )


def test_KMeansTransformer2():

    x_sk, y_sk = make_classification(n_samples=1000, n_features=20, n_informative=5, random_state=123)

    verif_encoder(
        df1=x_sk,
        df2=x_sk,
        y1=y_sk,
        klass=KMeansTransformer,
        enc_kwargs={"n_clusters": 10, "random_state": 123, "result_type": "probability", "kmeans_other_params":{"n_init":1}},
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),
        additional_test_functions=[
            check_all_numerical,
            check_no_null,
            check_between_01,
            nb_columns_verify(10),
            type_verifier(DataTypes.DataFrame),
        ],
        randomized_transformer=True,
    )


@pytest.mark.longtest
@pytest.mark.parametrize("desired_output_type", [DataTypes.DataFrame, DataTypes.NumpyArray])
def test_KMeansTransformer3(desired_output_type):
    verif_encoder(
        df1=df1_nona.loc[:, variable_by_type["NUM"]],
        df2=df2_nona.loc[:, variable_by_type["NUM"]],
        y1=y_train_shuffled,
        klass=KMeansTransformer,
        enc_kwargs={
            "random_state": 123,
            "n_clusters": 10,
            "result_type": "probability",
            "desired_output_type": desired_output_type,
            "n_init": 1
        },
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray, DataTypes.SparseArray),
        additional_test_functions=[
            check_all_numerical,
            check_no_null,
            check_between_01,
            check_row_sum1,
            nb_columns_verify(10),
            type_verifier(desired_output_type),
        ],
        randomized_transformer=True,
    )


@pytest.mark.parametrize("desired_output_type", [DataTypes.DataFrame, DataTypes.NumpyArray])
def test_KMeansTransformer4(desired_output_type):

    x_sk, y_sk = make_classification(n_samples=1000, n_features=20, n_informative=5, random_state=123)

    verif_encoder(
        df1=x_sk,
        df2=x_sk,
        y1=y_sk,
        klass=KMeansTransformer,
        enc_kwargs={"n_clusters": 10, "random_state": 123, "result_type": "probability", "kmeans_other_params":{"n_init":1}},
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),
        additional_test_functions=[
            check_all_numerical,
            check_no_null,
            check_between_01,
            nb_columns_verify(10),
            type_verifier(desired_output_type),
        ],
        randomized_transformer=True,
    )


@pytest.mark.longtest
@pytest.mark.parametrize(
    "temperature, result_type",
    list(itertools.product((0.01, 0.1, 1), ("distance", "inv_distance", "log_distance", "probability", "cluster"))),
)
def test_KMeansTransformer5(temperature, result_type):

    if result_type == "probability":
        f = [check_between_01, check_row_sum1]
    else:
        f = []

    verif_encoder(
        df1=df1_nona.loc[:, variable_by_type["NUM"]],
        df2=df2_nona.loc[:, variable_by_type["NUM"]],
        y1=y_train_shuffled,
        klass=KMeansTransformer,
        enc_kwargs={"random_state": 123, "n_clusters": 10, "temperature": temperature, "result_type": result_type, "kmeans_other_params":{"n_init":1}},
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray, DataTypes.SparseArray),
        additional_test_functions=[
            check_all_numerical,
            check_no_null,
            check_positive,
            nb_columns_verify(10),
            type_verifier(DataTypes.DataFrame),
        ]
        + f,
        randomized_transformer=True,
    )


@pytest.mark.longtest
@pytest.mark.parametrize(
    "temperature, result_type",
    list(itertools.product((0.01, 0.1, 1), ("distance", "inv_distance", "log_distance", "probability", "cluster"))),
)
def test_KMeansTransformer6(temperature, result_type):

    if result_type == "probability":
        f = [check_between_01, check_row_sum1]
    else:
        f = []

    x_sk, y_sk = make_classification(n_samples=1000, n_features=20, n_informative=5, random_state=123)

    verif_encoder(
        df1=x_sk,
        df2=x_sk,
        y1=y_sk,
        klass=KMeansTransformer,
        enc_kwargs={"random_state": 123, "n_clusters": 10, "temperature": temperature, "result_type": result_type, "kmeans_other_params":{"n_init":1}},
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),
        additional_test_functions=[
            check_all_numerical,
            check_no_null,
            check_positive,
            nb_columns_verify(10),
            type_verifier(DataTypes.DataFrame),
        ]
        + f,
        randomized_transformer=True,
    )


def verif_KMeansTransformer():
    """ to manually run the test if needed """

    test_KMeansTransformer1()
    test_KMeansTransformer2()

    for desired_output_type in (DataTypes.DataFrame, DataTypes.NumpyArray):
        test_KMeansTransformer3(desired_output_type=desired_output_type)
        test_KMeansTransformer4(desired_output_type=desired_output_type)

    for temperature in (0.01, 0.1, 1):
        for result_type in ("distance", "inv_distance", "log_distance", "probability", "cluster"):
            test_KMeansTransformer5(temperature=temperature, result_type=result_type)
            test_KMeansTransformer6(temperature=temperature, result_type=result_type)


# In[] : CdsScaler


def test_CdfScaler1():
    verif_encoder(
        df1=df1_nona.loc[:, variable_by_type["NUM"]],
        df2=df2_nona.loc[:, variable_by_type["NUM"]],
        y1=y_train_shuffled,
        klass=CdfScaler,
        enc_kwargs={"random_state": 123},
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),
        additional_test_functions=[
            check_all_numerical,
            check_no_null,
            didnot_change_column_nb,
            didnot_change_column_names,
        ],
        randomized_transformer=False,
    )


def test_CdfScaler2():
    verif_encoder(
        df1=df1_nona.loc[:, variable_by_type["NUM"]],
        df2=df2_nona.loc[:, variable_by_type["NUM"]],
        y1=y_train_shuffled,
        klass=CdfScaler,
        enc_kwargs={"distribution": "kernel", "random_state": 123},
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),
        additional_test_functions=[
            check_all_numerical,
            check_no_null,
            check_between_01,
            didnot_change_column_nb,
            didnot_change_column_names,
        ],
        randomized_transformer=False,
    )


@pytest.mark.longtest
@pytest.mark.parametrize(
    "distribution, output_distribution",
    list(itertools.product(("none", "kernel", "auto-param", "auto-kernel", "rank"), ("uniform", "normal"))),
)
def test_CdfScaler_with_params(distribution, output_distribution):

    verif_encoder(
        df1=df1_nona.loc[:, variable_by_type["NUM"]],
        df2=df2_nona.loc[:, variable_by_type["NUM"]],
        y1=y_train_shuffled,
        klass=CdfScaler,
        enc_kwargs={"distribution": distribution, "output_distribution": output_distribution, "random_state": 123},
        all_types=(DataTypes.DataFrame, DataTypes.NumpyArray),
        additional_test_functions=[
            check_all_numerical,
            check_no_null,
            didnot_change_column_nb,
            didnot_change_column_names,
        ],
        randomized_transformer=False,
    )

