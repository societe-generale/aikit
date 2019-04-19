# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 13:31:32 2018

@author: Lionel Massoulard
"""


import numpy as np
import scipy.sparse

from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from tests.helpers.testing_help import rec_assert_equal
from aikit.tools.data_structure_helper import convert_generic, DataTypes, get_type

import pytest


ADDITIONAL_CONVERSIONS_FUNCTIONS = {
    DataTypes.SparseArray: (scipy.sparse.coo_matrix, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix)
}
# TODO :
def extend_all_type(all_types):
    all_types_conv = []
    for t in all_types:
        all_types_conv.append((t, None))

        other_functions = ADDITIONAL_CONVERSIONS_FUNCTIONS.get(t, None)
        if other_functions is not None:
            for f in other_functions:
                all_types_conv.append((t, f))

    return all_types_conv


# In[]


def assert_raise_not_fitted(model, df):
    with pytest.raises(NotFittedError):
        model.predict(df)


def assert_raise_value_error(model, df):
    with pytest.raises(ValueError):
        model.predict(df)


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


def verif_model(df1, df2, y1, klass, model_kwargs, all_types, is_classifier):
    """ helper function that check (using asserts) a bunch a thing on a model klass
    
    Parameters
    ----------
    
    df1 : array like
        data on which model will be trained
    
    df2 : array like
        data on which model will be tested
        
    klass : type
        type of the model to test
        
    model_kwargs : dict
        kwargs to be passed to klass to create a model
        
    all_types : list of type
        list of input type to test the models on
        
    is_classifier : boolean
        if True the model is a Classifier otherwise a Regressor
   
    
    """

    if not isinstance(all_types, (list, tuple)):
        all_types = (all_types,)

    model0 = klass(**model_kwargs)  # Create an object ...
    model1 = clone(model0)  # then try to clone it

    model2 = klass()  # Create an empty object and then set its params
    model2.set_params(**model_kwargs)

    # Verify type are iddentical
    assert type(model0) == type(model1)
    assert type(model0) == type(model2)

    assert hasattr(klass, "fit")
    assert hasattr(klass, "predict")
    if is_classifier:
        assert hasattr(klass, "predict_proba")

    # Verify get_params are identical
    params_0 = model0.get_params()
    params_1 = model1.get_params()
    params_2 = model2.get_params()

    rec_assert_equal(params_0, params_1)
    rec_assert_equal(params_0, params_2)

    rec_assert_equal({k: v for k, v in params_0.items() if k in model_kwargs}, model_kwargs)
    rec_assert_equal({k: v for k, v in params_1.items() if k in model_kwargs}, model_kwargs)
    rec_assert_equal({k: v for k, v in params_2.items() if k in model_kwargs}, model_kwargs)

    extended_all_types = extend_all_type(all_types)

    if is_classifier:
        yclasses = list(set(np.unique(y1)))
        nb_classes = len(yclasses)

    for fit_type, additional_conversion_fun in extended_all_types:

        # Convert inputs into several type ..
        df1_conv = convert_generic(df1, output_type=fit_type)
        df2_conv = convert_generic(df2, output_type=fit_type)

        if additional_conversion_fun is not None:
            df1_conv = additional_conversion_fun(df1_conv)
            df2_conv = additional_conversion_fun(df2_conv)

        model_a = klass(**model_kwargs)
        model_a.fit(df1_conv, y=y1)

        y1_hat_a = model_a.predict(df1_conv)  # Other test with an y (might be None or not)
        y2_hat_a = model_a.predict(df2_conv)

        if is_classifier:
            y1_hatproba_a = model_a.predict_proba(df1_conv)
            y2_hatproba_a = model_a.predict_proba(df2_conv)

        params_3 = model_a.get_params()  # Verif that get_params didn't change after fit
        # Rmk : might no be enforce ON all transformeurs

        rec_assert_equal(params_0, params_3)

        assert y1_hat_a is not None  # verify that something was created
        assert y2_hat_a is not None  # verify that something was created

        model_cloned = clone(model_a)  # Clone again ...
        assert_raise_not_fitted(
            model_cloned, df2_conv
        )  # ... and verify that the clone isn't fitted, even if model_a is fitted

        # Same thing but using clone
        model_b = clone(model_a)
        model_b.fit(df1_conv, y=y1)

        y1_hat_b = model_b.predict(df1_conv)
        y2_hat_b = model_b.predict(df2_conv)
        if is_classifier:
            y1_hatproba_b = model_b.predict_proba(df1_conv)
            y2_hatproba_b = model_b.predict_proba(df2_conv)

        # Same thing but with set_params
        model_c = klass()
        model_c.set_params(**model_kwargs)
        model_c.fit(df1_conv, y=y1)

        y1_hat_c = model_c.predict(df1_conv)
        y2_hat_c = model_c.predict(df2_conv)

        if is_classifier:
            y1_hatproba_c = model_c.predict_proba(df1_conv)
            y2_hatproba_c = model_c.predict_proba(df2_conv)

        # check error when call with too few columns
        assert_raise_value_error(model_a, gen_slice(df1_conv, slice(1, None)))
        assert_raise_value_error(model_b, gen_slice(df1_conv, slice(1, None)))
        assert_raise_value_error(model_c, gen_slice(df1_conv, slice(1, None)))

        assert y1_hat_a.shape[0] == df1_conv.shape[0]
        assert y1_hat_b.shape[0] == df1_conv.shape[0]
        assert y1_hat_c.shape[0] == df1_conv.shape[0]

        assert y2_hat_a.shape[0] == df2_conv.shape[0]
        assert y2_hat_b.shape[0] == df2_conv.shape[0]
        assert y2_hat_c.shape[0] == df2_conv.shape[0]

        assert y1_hat_a.ndim == y1.ndim
        assert y1_hat_b.ndim == y1.ndim
        assert y1_hat_c.ndim == y1.ndim

        assert y2_hat_a.ndim == y1.ndim
        assert y2_hat_b.ndim == y1.ndim
        assert y2_hat_c.ndim == y1.ndim

        if is_classifier:
            assert y1_hatproba_a.ndim == 2
            assert y1_hatproba_b.ndim == 2
            assert y1_hatproba_c.ndim == 2
            assert y2_hatproba_a.ndim == 2
            assert y2_hatproba_b.ndim == 2
            assert y2_hatproba_c.ndim == 2

            y1_hatproba_a.shape[1] == nb_classes
            y1_hatproba_b.shape[1] == nb_classes
            y1_hatproba_c.shape[1] == nb_classes

            y2_hatproba_a.shape[1] == nb_classes
            y2_hatproba_b.shape[1] == nb_classes
            y2_hatproba_c.shape[1] == nb_classes

            assert hasattr(model_a, "classes_")
            assert hasattr(model_b, "classes_")
            assert hasattr(model_c, "classes_")

            assert list(set(model_a.classes_)) == list(set(yclasses))
            assert list(set(model_b.classes_)) == list(set(yclasses))
            assert list(set(model_c.classes_)) == list(set(yclasses))

            for f in (check_all_numerical, check_between_01, check_no_null):

                f(y1_hatproba_a)
                f(y1_hatproba_b)
                f(y1_hatproba_c)

                f(y2_hatproba_a)
                f(y2_hatproba_b)
                f(y2_hatproba_c)

        # Verif type
        assert get_type(y1_hat_b) == get_type(y1_hat_a)
        assert get_type(y1_hat_c) == get_type(y1_hat_a)
        assert get_type(y2_hat_a) == get_type(y1_hat_a)
        assert get_type(y2_hat_b) == get_type(y1_hat_a)
        assert get_type(y2_hat_c) == get_type(y1_hat_a)
