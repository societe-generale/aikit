# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 14:00:05 2018

@author: Lionel Massoulard
"""

import numpy as np

from aikit.ml_machine.ml_machine_guider import kde_transfo_quantile, transfo_quantile


def test_kde_transfo_quantile():
    xx = np.random.randn(100)
    rr = kde_transfo_quantile(xx)
    assert xx.shape == rr.shape
    assert rr.min() > 0
    assert rr.max() < 1


def test_transfo_quantile():
    xx = np.random.randn(100)
    rr = transfo_quantile(xx)
    assert xx.shape == rr.shape
    assert rr.min() > 0
    assert rr.max() < 1
