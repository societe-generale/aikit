# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 14:46:40 2018

@author: Lionel Massoulard
"""

import pytest
from aikit.datasets.datasets import load_dataset, DatasetEnum


@pytest.mark.parametrize("name", DatasetEnum.alls)
def test_load_dataset(name):

    res = load_dataset(name)

    assert isinstance(res, tuple)
    assert len(res) == 5
    df_train, y_train, df_test, y_test, infos = res

    assert df_train is not None
    if name != "pokemon":
        assert y_train is not None
    assert isinstance(infos, dict)

    if y_train is not None:
        assert df_train.shape[0] == y_train.shape[0]
    if df_test is not None:
        assert df_train.shape[1] == df_test.shape[1]
        if y_test is not None:
            assert y_test.shape[0] == df_test.shape[0]
