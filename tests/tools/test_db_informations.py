# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:21:04 2018

@author: Lionel Massoulard
"""

import pandas as pd
import numpy as np

from aikit.tools.db_informations import has_missing_values, guess_type_of_variable, TypeOfVariables


def test_has_missing_values():
    s1 = pd.Series(np.random.randn(10))

    r1 = has_missing_values(s1)
    assert not r1
    assert isinstance(r1, bool)

    s2 = s1.copy()
    s2[10] = np.nan
    r2 = has_missing_values(s2)
    assert r2
    assert isinstance(r2, bool)


def verif_all():
    test_has_missing_values()


def test_guess_type_of_variable_boolean():
    s = pd.Series([True,False,True,None]*10)
    assert guess_type_of_variable(s) == TypeOfVariables.CAT
    
    s = pd.Series([True,False,True]*10)
    assert guess_type_of_variable(s) == TypeOfVariables.CAT
    
