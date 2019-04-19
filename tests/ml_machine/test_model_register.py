# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 14:28:44 2018

@author: Lionel Massoulard
"""

from aikit.ml_machine.model_registrer import singleton


def test_singleton():
    @singleton
    class Foo(object):
        def __init__(self, f=1):
            self.f = f

    f1 = Foo()
    f2 = Foo()

    assert f1 is f2
    f1.f = 10
    assert f2.f == 10
