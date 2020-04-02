# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:21:04 2018

@author: Lionel Massoulard
"""

import pytest

import pandas as pd
import numpy as np

from aikit.tools.data_structure_helper import convert_generic, DataTypes, _IS_PD1, convert_to_sparseserie
from aikit.tools.db_informations import has_missing_values, guess_type_of_variable, TypeOfVariables, get_n_outputs, get_columns_informations

from tests.helpers.testing_help import get_sample_df


if _IS_PD1:
    SPARSES = [True, False]
else:
    SPARSES = [False]
    
def _convert_sparse(x, sparse):
    if isinstance(x, pd.Series):
        if _IS_PD1 and sparse:
            return convert_to_sparseserie(x)
        else:
            return x # nothing, I don't want to test sparse 
    elif isinstance(x, pd.DataFrame):
        return convert_generic(x, output_type=DataTypes.SparseDataFrame)
    
    else:
        TypeError("This function is for DataFrame or Serie")
        
    
    
@pytest.mark.parametrize("sparse", SPARSES)
def test_has_missing_values(sparse):
    np.random.seed(123)
    s = pd.Series(np.random.randn(10))
    
    s = _convert_sparse(s, sparse)

    r1 = has_missing_values(s)
    assert not r1
    assert isinstance(r1, bool)

    s = pd.Series(np.random.randn(10))
    s[10] = np.nan
    s = _convert_sparse(s, sparse)
    
    r2 = has_missing_values(s)
    assert r2
    assert isinstance(r2, bool)


def test_get_n_outputs():
    y = np.zeros((10,))
    assert get_n_outputs(y) == 1

    y = np.zeros((10, 1))
    assert get_n_outputs(y) == 1

    y = np.zeros((10, 2))
    assert get_n_outputs(y) == 2

    y = pd.Series(np.zeros((10,)))
    assert get_n_outputs(y) == 1

    y = pd.DataFrame(np.zeros((10, 1)))
    assert get_n_outputs(y) == 1

    y = pd.DataFrame(np.zeros((10, 2)))
    assert get_n_outputs(y) == 2

@pytest.mark.parametrize("sparse", SPARSES)
def test_guess_type_of_variable_boolean(sparse):
    s = pd.Series([True, False, True, None] * 10)
    
    s = _convert_sparse(s, sparse)

    assert guess_type_of_variable(s) == TypeOfVariables.CAT

    s = pd.Series([True, False, True] * 10)
    s = _convert_sparse(s, sparse)

    assert guess_type_of_variable(s) == TypeOfVariables.CAT

@pytest.mark.parametrize("sparse", SPARSES)
def test_guess_type_of_variable(sparse):
    df = get_sample_df(100)
    df["cat_col_1"] = df["text_col"].apply(lambda s: s[0:3])
    
    df = _convert_sparse(df, sparse)

    assert guess_type_of_variable(df["float_col"]) == "NUM"
    assert guess_type_of_variable(df["int_col"]) == "NUM"
    assert guess_type_of_variable(df["text_col"]) == "TEXT"
    assert guess_type_of_variable(df["cat_col_1"]) == "CAT"

    df_with_cat = df.copy()
    df_with_cat["cat_col_1"] = df_with_cat["cat_col_1"].astype("category")
    assert np.all([guess_type_of_variable(df[col]) == guess_type_of_variable(df_with_cat[col]) for col in df.columns])
    assert (df.values == df_with_cat.values).all()
