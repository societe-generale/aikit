# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:22:04 2018

@author: Lionel Massoulard
"""

from aikit.enums import TypeOfProblem, TypeOfVariables, StepCategories, SpecialModels, DataTypes


def _verif_alls(klass):
    all_attrs = klass.__dict__
    assert "alls" in all_attrs
    for key, value in all_attrs.items():
        if isinstance(value, str) and key == value:
            if value not in all_attrs["alls"]:
                raise ValueError("'%s' should be in 'alls'" % value)

    return klass


def test_TypeOfProblem():
    _verif_alls(TypeOfProblem)


def test_TypeOfVariables():
    _verif_alls(TypeOfVariables)


def test_StepCategories():
    _verif_alls(StepCategories)


def test_SpecialModelss():
    _verif_alls(SpecialModels)


def test_DataTypes():
    _verif_alls(DataTypes)
