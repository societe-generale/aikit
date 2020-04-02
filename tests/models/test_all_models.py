# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:02:17 2020

@author: Lionel Massoulard
"""

import pytest

import itertools

import scipy.sparse
import numpy as np
import pandas as pd

from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from aikit.tools.data_structure_helper import convert_generic, get_type
from aikit.enums import DataTypes
from aikit.models.sklearn_lightgbm_wrapper import LGBMClassifier, LGBMRegressor

from tests.helpers.testing_help import rec_assert_equal

def assert_raise_not_fitted(encoder, df):
    with pytest.raises(NotFittedError):
        encoder.predict(df)

def assert_raise_value_error(encoder, df):
    with pytest.raises(ValueError):
        encoder.predict(df)
    
try:
    import lightgbm
except ImportError:
    lightgbm = None
    
ADDITIONAL_CONVERSIONS_FUNCTIONS = {
    DataTypes.SparseArray: (scipy.sparse.coo_matrix, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix)
}


np.random.seed(123)
dfX = pd.DataFrame(np.random.randn(100, 10), columns=[f"COL_{j}" for j in range(10)])
y_reg   = dfX.iloc[:,0] + np.random.randn(100)
y_cla   = np.random.randint(low=0, high=3, size=100)


df1 = dfX.iloc[0:80,:]
y1_reg  = y_reg[0:80]
y1_cla  = y_cla[0:80]
df2 = dfX.iloc[80:,:]


def extend_all_type(all_types):
    all_types_conv = []
    for t in all_types:
        all_types_conv.append((t, None))

        other_functions = ADDITIONAL_CONVERSIONS_FUNCTIONS.get(t, None)
        if other_functions is not None:
            for f in other_functions:
                all_types_conv.append((t, f))

    return all_types_conv

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

def get_target(classification,
                        multi_output,
                        only_two_classes,
                        classes_as_string,
                        target_is_numpy):
    
    if classification:
        y1 = y1_cla.copy()
        if only_two_classes:
            y1[y1 > 1] = 1
            
        if classes_as_string:
            y1 = np.array([f"CL_{y}" for y in y1])
                        
    else:
        y1 = y1_reg

    if multi_output:
        y1 = np.repeat(y1[:,np.newaxis], 2,axis=1)

    if not target_is_numpy:
        if y1.ndim > 1:
            y1 = pd.DataFrame(y1, columns=[f"TARGET_{j}" for j in range(y1.shape[1])])
        else:
            y1 = pd.Series(y1)
            
    return y1


def verif_model(
    df1,
    df2,
    y1,
    klass,
    enc_kwargs,
    all_types
):
    """ function to test differents things on a model """

    is_multiple_output = y1.ndim > 1 and y1.shape[1] >= 1
    nb_outputs = 1
    if is_multiple_output:
        nb_outputs = y1.shape[1]

    if not isinstance(all_types, (list, tuple)):
        all_types = (all_types,)


    # all_types = (DataTypes.DataFrame, DataTypes.SparseDataFrame)
    assert hasattr(klass, "fit")
    assert hasattr(klass, "predict")

    encoder0 = klass(**enc_kwargs)  # Create an object ...
    encoder1 = clone(encoder0)  # then try to clone it

    encoder2 = klass()  # Create an empty object and then set its params
    encoder2.set_params(**enc_kwargs)
    
    if is_classifier(encoder0):
        model_is_classifier=True
        assert hasattr(klass, "predict_proba")
    else:
        model_is_classifier=False
        assert is_regressor(encoder0)
    

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

    extended_all_types = extend_all_type(all_types)
    
    def get_values(y):
        if hasattr(y, "values"):
            return y.values
        else:
            return y
        
    y1_np = get_values(y1)

    for fit_type, additional_conversion_fun in extended_all_types:

        # Convert inputs into several type ..
        df1_conv = convert_generic(df1, output_type=fit_type)
        df2_conv = convert_generic(df2, output_type=fit_type)

        if additional_conversion_fun is not None:
            df1_conv = additional_conversion_fun(df1_conv)
            df2_conv = additional_conversion_fun(df2_conv)

        encoder_a = klass(**enc_kwargs)
        y1_hat_a = encoder_a.fit(df1_conv, y=y1).predict(df1_conv)  # Other test with an y (might be None or not)
        y2_hat_a = encoder_a.predict(df2_conv)

        if model_is_classifier:
            y1_hat_proba_a = encoder_a.predict_proba(df1_conv)
            y2_hat_proba_a = encoder_a.predict_proba(df2_conv)
            if is_multiple_output:
                assert isinstance(y1_hat_proba_a, list)
                assert isinstance(y2_hat_proba_a, list)
                assert len(y1_hat_proba_a) == nb_outputs
                assert len(y2_hat_proba_a) == nb_outputs
                
                for j in range(nb_outputs):
                    # correct shape
                    assert y1_hat_proba_a[j].shape == (y1.shape[0], len(np.unique(y1_np[:, j])))
                    assert y2_hat_proba_a[j].shape[0] == df2_conv.shape[0]
                    assert y2_hat_proba_a[j].shape[1] == y1_hat_proba_a[j].shape[1]
                    
                    # between 0 and 1
                    assert y1_hat_proba_a[j].min() >= 0
                    assert y1_hat_proba_a[j].max() <= 1
                    
                    assert y2_hat_proba_a[j].min() >= 0
                    assert y2_hat_proba_a[j].max() <= 1
                    
                    # sum = 1
                    assert np.abs(y1_hat_proba_a[j].sum(axis=1) - 1).max() <= 10**(-5)
                    assert np.abs(y2_hat_proba_a[j].sum(axis=1) - 1).max() <= 10**(-5)
                    
            else:
                # correct shape
                assert y1_hat_proba_a.shape == (y1.shape[0], len(np.unique(y1_np)))
                assert y2_hat_proba_a.shape == (df2_conv.shape[0], len(np.unique(y1_np)))

                # between 0 and 1
                assert y1_hat_proba_a.min() >= 0
                assert y1_hat_proba_a.max() <= 1
                    
                assert y2_hat_proba_a.min() >= 0
                assert y2_hat_proba_a.max() <= 1
                    
                # sum = 1
                assert np.abs(y1_hat_proba_a.sum(axis=1) - 1).max() <= 10**(-5)
                assert np.abs(y2_hat_proba_a.sum(axis=1) - 1).max() <= 10**(-5)


        assert y1_hat_a is not None  # verify that something was created
        assert y2_hat_a is not None  # verify that something was created
        
        assert y1_hat_a.shape == y1.shape
        assert y2_hat_a.shape[0] == df2_conv.shape[0]
        assert y2_hat_a.shape[1:] == y1.shape[1:]
        
        if model_is_classifier:
            assert hasattr(encoder_a, "classes_")
            if is_multiple_output:
                assert len(encoder_a.classes_) == nb_outputs
                for j in range(nb_outputs):
                    assert list(encoder_a.classes_[j]) == list(np.unique(y1_np[:, j]))
            else:
                assert list(encoder_a.classes_) == list(np.unique(y1_np))

        # Verify that get_params didn't change after fit
        # Rmk : might not be enforced on all transformers
        params_3 = encoder_a.get_params()
        rec_assert_equal(params_0, params_3)


        encoder_cloned = clone(encoder_a)  # Clone again ...

        assert_raise_not_fitted(
            encoder_cloned, df2_conv
        )  # ... and verify that the clone isn't fitted, even if encoder_a is fitted

        # Same thing but using ... fit and then... transformed
        encoder_b = klass(**enc_kwargs)
        encoder_b.fit(df1_conv, y=y1)
        y1_hat_b = encoder_b.predict(df1_conv)
        y2_hat_b = encoder_b.predict(df2_conv)

        assert y1_hat_a is not None
        assert y2_hat_b is not None
        assert y1_hat_b.shape == y1.shape
        
        assert y2_hat_b.shape[0] == df2_conv.shape[0]
        assert y2_hat_b.shape[1:] == y1.shape[1:]
        
        # Same thing but using clone
        encoder_c = clone(encoder_a)
        y1_hat_c = encoder_c.fit(df1_conv, y=y1).predict(df1_conv)
        y2_hat_c = encoder_c.predict(df2_conv)
        
        assert y1_hat_c.shape == y1.shape
        
        assert y2_hat_c.shape[0] == df2_conv.shape[0]
        assert y2_hat_c.shape[1:] == y1.shape[1:]
        
        encoder_d = klass()
        encoder_d.set_params(**enc_kwargs)
        y1_hat_d = encoder_d.fit(df1_conv, y=y1).predict(df1_conv)
        y2_hat_d = encoder_d.predict(df2_conv)

        assert y1_hat_d.shape == y1.shape
        
        assert y2_hat_d.shape[0] == df2_conv.shape[0]
        assert y2_hat_d.shape[1:] == y1.shape[1:]
        
        assert_raise_value_error(encoder_a, gen_slice(df1_conv, slice(1, None)))
        assert_raise_value_error(encoder_b, gen_slice(df1_conv, slice(1, None)))
        assert_raise_value_error(encoder_c, gen_slice(df1_conv, slice(1, None)))
        assert_raise_value_error(encoder_d, gen_slice(df1_conv, slice(1, None)))

# In[] : Test of DecisionTree : mainly to make sur the testing code work...

classif_klasses = [DecisionTreeClassifier]
regress_klasses = [DecisionTreeRegressor]
all_params_to_test_classification = list(itertools.product(
        classif_klasses, # klass
        [True],          # classification
        [True, False],   # multi_output
        [True, False],   # only_two_classes
        [True, False],   # classes_as_string
        [True, False]    # target_is_numpy
))

all_params_to_test_regression = list(itertools.product(
        regress_klasses, # klass
        [False],         # classification
        [True, False],   # multi_output
        [False],         # only_two_classes
        [False],         # classes_as_string
        [True, False]    # target_is_numpy
))

all_params_to_test = all_params_to_test_classification + all_params_to_test_regression


@pytest.mark.parametrize("klass, classification, multi_output, only_two_classes, classes_as_string, target_is_numpy", all_params_to_test)
def test_DecisionTree_generic(klass,
                          classification,
                          multi_output,
                          only_two_classes,
                          classes_as_string,
                          target_is_numpy):
    
    y1 = get_target(
               classification=classification,
               multi_output=multi_output,
               only_two_classes=only_two_classes,
               classes_as_string=classes_as_string,
               target_is_numpy=target_is_numpy
               )
    
    return verif_model(
            df1=df1,
            df2=df2,
            y1=y1,
            klass=klass,
            enc_kwargs = {},
            all_types=(DataTypes.DataFrame, DataTypes.NumpyArray, DataTypes.SparseDataFrame, DataTypes.SparseArray)
    )


if lightgbm is not None:
    classif_klasses = [LGBMClassifier]
    regress_klasses = [LGBMRegressor]
else:
    classif_klasses = []
    regress_klasses = []
    
all_params_to_test_classification = list(itertools.product(
        [True, False],   # do_crossvalidation
        classif_klasses, # klass
        [True],          # classification
        [False],   # multi_output
        [True, False],   # only_two_classes
        [True, False],   # classes_as_string
        [True, False]    # target_is_numpy
))

all_params_to_test_regression = list(itertools.product(
        [True, False],   # do_crossvalidation
        regress_klasses, # klass
        [False],         # classification
        [False],   # multi_output
        [False],         # only_two_classes
        [False],         # classes_as_string
        [True, False]    # target_is_numpy
))

all_params_to_test_lgbm = all_params_to_test_classification + all_params_to_test_regression

@pytest.mark.skipif(lightgbm is None, reason="lightgbm isn't installed")
@pytest.mark.parametrize("do_crossvalidation, klass, classification, multi_output, only_two_classes, classes_as_string, target_is_numpy", all_params_to_test_lgbm)
def test_LGBM_generic(
      do_crossvalidation,
      klass,
      classification,
      multi_output,
      only_two_classes,
      classes_as_string,
      target_is_numpy):
    
    y1 = get_target(
            classification=classification,
            multi_output=multi_output,
            only_two_classes=only_two_classes,
            classes_as_string=classes_as_string,
            target_is_numpy=target_is_numpy
               )
    
    return verif_model(
            df1=df1,
            df2=df2,
            y1=y1,
            klass=klass,
            enc_kwargs = {"do_crossvalidation": do_crossvalidation},
            all_types=(DataTypes.DataFrame, DataTypes.NumpyArray, DataTypes.SparseDataFrame, DataTypes.SparseArray)
    )
    
