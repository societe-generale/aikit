# -*- coding: utf-8 -*-
"""
Created on Fri May  4 13:33:07 2018

@author: Lionel Massoulard
"""


from .datasets import (
    DatasetEnum,
    load_dataset,
    load_titanic,
    load_imdb,
    load_electricity,
    load_housing,
    load_quora,
    load_abalone,
    load_pokemon,
    load_wikinews,
)

__all__ = [
    "DatasetEnum",
    "load_dataset",
    "load_titanic",
    "load_imdb",
    "load_electricity",
    "load_housing",
    "load_quora",
    "load_abalone",
    "load_pokemon",
    "load_wikinews",
]
