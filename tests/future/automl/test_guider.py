# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 14:00:05 2018

@author: Lionel Massoulard
"""

import numpy as np

from aikit.future.automl.guider import kde_transform_quantile, transform_quantile


def test_kde_transform_quantile():
    xx = np.random.randn(100)
    rr = kde_transform_quantile(xx)
    assert xx.shape == rr.shape
    assert rr.min() > 0
    assert rr.max() < 1


def test_transform_quantile():
    xx = np.random.randn(100)
    rr = transform_quantile(xx)
    assert xx.shape == rr.shape
    assert rr.min() > 0
    assert rr.max() < 1
