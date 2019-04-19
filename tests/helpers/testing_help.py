# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 18:01:27 2018

@author: Lionel Massoulard
"""

import pandas as pd
import numpy as np


import scipy.sparse as sps

import string


def get_sample_df(size=10, seed=None):

    if seed is not None:
        np.random.seed(seed)

    words = ["aaa", "bbb", "ccc", "ddd", "eee", "fff", "jjj"]

    df = pd.DataFrame(
        {
            "int_col": range(size),
            "text_col": [" ".join(np.random.choice(words, 3)) for _ in range(size)],
            "float_col": np.random.randn(size),
        }
    ).loc[:, ["float_col", "int_col", "text_col"]]

    return df


def get_sample_data(add_na=True):
    xx = np.array(
        [
            [1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
            [0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    if add_na:
        xx[0, 1] = np.nan
        xx[5, 1] = np.nan

    xxs = sps.csc_matrix(xx).copy()
    xxd = pd.DataFrame(xx, columns=["col%d" % d for d in range(xx.shape[1])]).copy()

    return xx, xxd, xxs


def get_random_strings(n_samples=10, max_size=10):
    list_of_texts_random = list(
        np.apply_along_axis(
            lambda s: "".join(s), axis=1, arr=np.random.choice(list(string.ascii_letters), (n_samples, max_size))
        )
    )

    return list_of_texts_random


def rec_assert_equal(obj1, obj2):

    assert type(obj1) == type(obj2)

    if isinstance(obj1, (list, tuple)):

        for o1, o2 in zip(obj1, obj2):
            rec_assert_equal(o1, o2)

    elif isinstance(obj1, dict):
        assert obj1.keys() == obj2.keys()

        for k in obj1.keys():
            rec_assert_equal(obj1[k], obj2[k])

    elif isinstance(obj1, np.ndarray):
        assert obj1.shape == obj2.shape
        assert (obj1 == obj2).all()

    elif isinstance(obj1, pd.Series):
        assert (obj1 == obj2).all()

    elif isinstance(obj1, pd.DataFrame):
        assert (obj1 == obj2).all().all()

    else:
        assert obj1 == obj2
