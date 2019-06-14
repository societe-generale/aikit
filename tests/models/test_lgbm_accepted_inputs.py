# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 16:03:16 2018

@author: Lionel Massoulard
"""
import pytest

import numpy as np

try:
    from lightgbm import LGBMClassifier
except ImportError:
    print("This test won't run, please install lightgbm")
    LGBMClassifier = None

import pytest

from aikit.tools.data_structure_helper import convert_generic, DataTypes

np.random.seed(123)
X = 5 * np.random.randn(100, 10)
y = 1 * (np.random.randn(100) > 0)

@pytest.mark.skipif(LGBMClassifier is None, reason="lightgbm is not installed")
class Test_lightgbm(object):
    def test_float(self):
        lgbm = LGBMClassifier()
        lgbm.fit(X, y)

    def test_int(self):
        Xint = X.astype(np.int32)

        lgbm = LGBMClassifier()
        lgbm.fit(Xint, y)

    def test_sparse(self):
        Xsparse = convert_generic(X, output_type=DataTypes.SparseArray)
        lgbm = LGBMClassifier()
        lgbm.fit(Xsparse, y)

    """
    @pytest.mark.xfail
    def test_sparse_int(self):
        Xsparse_int = convert_generic(X, output_type=DataTypes.SparseArray).astype(np.int32)
        lgbm = LGBMClassifier()
        lgbm.fit(Xsparse_int, y)
    """

    def test_sparse_df(self):
        Xsparse_df = convert_generic(X, output_type=DataTypes.SparseDataFrame)
        lgbm = LGBMClassifier()
        lgbm.fit(Xsparse_df, y)

    def test_sparse_df_int(self):
        Xsparse_int = convert_generic(X, output_type=DataTypes.SparseArray).astype(np.int32)
        Xsparse_df_int = convert_generic(Xsparse_int, output_type=DataTypes.SparseDataFrame)
        lgbm = LGBMClassifier()
        lgbm.fit(Xsparse_df_int, y)
