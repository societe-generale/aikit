# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 17:56:37 2019

@author: Lionel Massoulard
"""

from .stacking import StackerClassifier, StackerRegressor, OutSamplerTransformer
from .base import DBSCANWrapper, KMeansWrapper, AgglomerativeClusteringWrapper

try:
    import lightgbm
except ImportError:
    lightgbm = None
    
if lightgbm is not None:
    from .sklearn_lightgbm_wrapper import LGBMClassifier, LGBMRegressor, LGBMRanker

__all__ = [
    "StackerClassifier",
    "StackerRegressor",
    "OutSamplerTransformer",
    "DBSCANWrapper",
    "KMeansWrapper",
    "AgglomerativeClusteringWrapper",
]

if lightgbm is not None:
    __all__ += ["LGBMClassifier", "LGBMRegressor", "LGBMRanker"]
