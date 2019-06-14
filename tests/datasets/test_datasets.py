# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 14:46:40 2018

@author: Lionel Massoulard
"""
import os
import pytest
from aikit.datasets.datasets import load_dataset, _load_public_path


@pytest.mark.parametrize("name", ["titanic"])
def test_load_dataset(tmpdir, name):
    res = load_dataset(name, cache_dir=tmpdir)

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


def test_load_public_path(tmpdir):
    path = _load_public_path(
        'https://github.com/gfournier/aikit-datasets/releases/download/titanic-1.0.0/titanic.tar.gz',
        cache_dir=tmpdir,
        cache_subdir='datasets')
    assert path == os.path.join(tmpdir, 'datasets', 'titanic.csv')
    assert os.path.exists(path)

