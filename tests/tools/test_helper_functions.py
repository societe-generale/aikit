# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:33:30 2018

@author: Lionel Massoulard
"""

import pytest

import pandas as pd
import numpy as np

from collections import OrderedDict

from aikit.tools.helper_functions import (
    function_has_named_argument,
    diff,
    intersect,
    deep_flatten,
    tuple_include,
    unlist,
    unnest_tuple,
    pd_match,
    _is_number,
    shuffle_all,
    md5_hash,
    clean_column,
    dico_key_filter,
    dico_value_filter,
    dico_key_map,
    dico_value_map,
    _increase_name,
    _find_name_not_present
)


def test__increase_name():
    assert _increase_name('a') == 'a__1'
    assert _increase_name('a__1') == 'a__2'
    assert _increase_name('a__2') == 'a__3'
    assert _increase_name('a__10') == 'a__11'
    
    assert _increase_name('a__b') == 'a__b__1'
    assert _increase_name('a__b__1') == 'a__b__2'
    assert _increase_name('a__b__2') == 'a__b__3'
    assert _increase_name('a__b__10') == 'a__b__11'


def test__find_name_not_present():
    assert _find_name_not_present('not_present', {}) == 'not_present'
    assert _find_name_not_present('not__present', {}) == 'not__present'
    assert _find_name_not_present('a', {'a','b'}) == 'a__1'
    assert _find_name_not_present('a', {'a','a__1', 'b'}) == 'a__2'
    assert _find_name_not_present('a__1', {'a','a__1', 'b'}) == 'a__2'


def test_function_has_named_argument():
    def f1(a, b):
        pass

    def f2(a, b, **kwargs):
        pass

    def f3(a=None, b=10, *args, **kwargs):
        pass

    class Foo(object):
        def f(self, a, b):
            pass

        @staticmethod
        def f2(a, b):
            pass

    class Functor(object):
        def __call__(self, a, b):
            pass

    for f in (f1, f2, f3, Foo.f, Foo().f, Foo.f2, Foo().f2, Functor()):
        assert function_has_named_argument(f, "a")
        assert function_has_named_argument(f, "b")
        assert not function_has_named_argument(f, "c")


def test_diff():
    list1 = [1, 2, 3]
    list2 = [3, 4, 5]

    assert diff(list1, list2) == [1, 2]
    assert diff(list2, list1) == [4, 5]

    assert diff(list1, []) == list1

    list1 = ["a", "b", "c"]
    list2 = ["d", "c", "e"]

    assert diff(list1, list2) == ["a", "b"]
    assert diff(list2, list1) == ["d", "e"]

    assert diff(list1, []) == list1

    assert isinstance(diff((1, 2, 3), (1, 2)), tuple)


def test_intersect():
    list1 = [1, 2, 3]
    list2 = [3, 4, 5]

    assert intersect(list1, list2) == [3]
    assert intersect(list2, list1) == [3]

    list1 = [1, 2, 3, 4]
    list2 = [4, 3, 5, 6]

    assert intersect(list1, list2) == [3, 4]
    assert intersect(list2, list1) == [4, 3]

    assert intersect(list1, []) == []

    list1 = ["a", "b", "c"]
    list2 = ["d", "c", "e"]

    assert intersect(list1, list2) == ["c"]
    assert intersect(list2, list1) == ["c"]

    assert intersect(list1, []) == []
    assert isinstance(diff((1, 2, 3), (1, 2)), tuple)


def test_unlist():
    assert unlist([[1, 10], [32]]) == [1, 10, 32]
    assert unlist([[10], [11], [], [45]]) == [10, 11, 45]


def test_deep_flatten():
    assert deep_flatten([1, (2, 3, 4, [5, 6, 7], [8, 9])]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert deep_flatten([1, [], [], (2, 3, 4, [5, 6, 7], [8, 9])]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_unnest_tuple():
    examples = [
        ((1, 2, 3), (1, 2, 3)),
        ((1, (2, 3)), (1, 2, 3)),
        ((1, (2, (3))), (1, 2, 3)),
        (((1,), (2,), (3, 4)), (1, 2, 3, 4)),
    ]

    for nested_tuple, unnested_tuple in examples:
        assert unnest_tuple(nested_tuple) == unnested_tuple


def test_tuple_include():
    assert tuple_include((1, 2), (1, 2, 3))
    assert not tuple_include((1, 2, 3), (1, 2))
    assert tuple_include((1, 2, 3), (1, 2, 3))
    assert not tuple_include((1, 2, 4), (1, 2, 3))


def test_pd_match():
    values = pd.Series(["a", "b", "c"])
    to_match = pd.Series(["a", "c", "b", "a"])
    expected_res = np.array([0, 2, 1, 0])

    for t1 in (pd.Series, np.array, list, tuple):
        for t2 in (pd.Series, np.array, list, tuple):
            res = pd_match(t1(to_match), t2(values), na_sentinel=-1)
            assert np.array_equal(expected_res, res)

    values = pd.Series(["a", "b", "c"])
    to_match = pd.Series([None, "a", "c", "b", "a"])
    expected_res = np.array([-1, 0, 2, 1, 0])
    for t1 in (pd.Series, np.array, list, tuple):
        for t2 in (pd.Series, np.array, list, tuple):
            res = pd_match(t1(to_match), t2(values), na_sentinel=-1)
            assert np.array_equal(expected_res, res)

    values = pd.Series([0, 1, 2])
    to_match = pd.Series([0, 2, 1, 0])
    expected_res = np.array([0, 2, 1, 0])
    for t1 in (pd.Series, np.array, list, tuple):
        for t2 in (pd.Series, np.array, list, tuple):
            res = pd_match(t1(to_match), t2(values), na_sentinel=-1)
            assert np.array_equal(expected_res, res)

    values = pd.Series([0, 1, 2])
    to_match = pd.Series([np.nan, 0, 2, 1, 0])
    expected_res = np.array([-1, 0, 2, 1, 0])

    for t1 in (pd.Series, np.array, list, tuple):
        for t2 in (pd.Series, np.array, list, tuple):
            res = pd_match(t1(to_match), t2(values), na_sentinel=-1)
            assert np.array_equal(expected_res, res)

    values = pd.Series(["a", "b", "b", "c"])
    to_match = pd.Series(["a", "c", "b", "a"])
    expected_res = np.array([0, 3, 1, 0])

    for t1 in (pd.Series, np.array, list, tuple):
        for t2 in (pd.Series, np.array, list, tuple):
            res = pd_match(t1(to_match), t2(values), na_sentinel=-1)
            assert np.array_equal(expected_res, res)

    values = pd.Series(["a", "b", None, "c"])
    to_match = pd.Series(["a", "c", "b", "a"])
    expected_res = np.array([0, 3, 1, 0])

    for t1 in (pd.Series, np.array, list, tuple):
        for t2 in (pd.Series, np.array, list, tuple):
            res = pd_match(t1(to_match), t2(values), na_sentinel=-1)
            assert np.array_equal(expected_res, res)

    values = pd.Series(["a", "b", None, "c"])
    to_match = pd.Series(["a", "c", "b", "a", None])
    expected_res = np.array([0, 3, 1, 0, 2])

    for t1 in (pd.Series, np.array, list, tuple):
        for t2 in (pd.Series, np.array, list, tuple):
            res = pd_match(t1(to_match), t2(values), na_sentinel=-1)
            assert np.array_equal(expected_res, res)

    values = pd.Series(["a", "b", None, "c"])
    to_match = pd.Series(["a", "c", "b", "a", np.nan])
    expected_res = np.array([0, 3, 1, 0, -1])

    for t1 in (pd.Series, np.array, list, tuple):
        for t2 in (pd.Series, np.array, list, tuple):
            res = pd_match(t1(to_match), t2(values), na_sentinel=-1)
            assert np.array_equal(expected_res, res)


def test__is_number():
    examples_numbers = [10, 10.1, np.float16(19.0), np.int32(12), np.nan]

    for x in examples_numbers:
        assert _is_number(x)

    examples_not_numbers = ["toto", None, "a", "10.0", "0"]
    for x in examples_not_numbers:
        assert not _is_number(x)


def test_shuffle_all():

    x = list(range(10))
    r = [4, 0, 7, 5, 8, 3, 1, 6, 9, 2]
    assert r == shuffle_all(x, seed=123)
    assert (np.array(r) == shuffle_all(np.array(x), seed=123)).all()
    assert (shuffle_all(pd.Series(x), seed=123) == pd.Series(r, index=r)).all()

    xx = np.random.randn(10, 3)
    assert np.array_equal(shuffle_all(xx, seed=123), xx[np.array(r), :])

    a, b = shuffle_all(xx, x, seed=123)
    assert np.array_equal(a, xx[np.array(r), :])
    assert b == r

    xx = np.random.randn(10, 2, 3)
    assert np.array_equal(shuffle_all(xx, seed=123), xx[np.array(r), :, :])


def test_md5_hash():
    ob1 = ((1, 2), {"1": "2"})
    ob2 = ((1, 2), {1: "2"})
    ob3 = ([1, 2], {1: 2})

    h1 = md5_hash(ob1)
    h2 = md5_hash(ob2)
    h3 = md5_hash(ob3)
    assert len(set((h1, h2, h3))) == 3  # Test 3 different hash

    assert h1 == "d5f3de055dd4049def2766a9a6a3e914"
    assert h2 == "e8c67e026e91872ef85bc56cf67ab97a"
    assert h3 == "f409ec84efccad047568fa1ca5d0f990"


def test_clean_column():
    examples = [
        ("UPPER_CASED", "upper_cased"),
        ("already_clean", "already_clean"),
        ("too_many_dash___", "too_many_dash_"),
        ("$ and €", "usd_and_eur"),
        ("£ and ¥", "gbp_and_jpy"),
        ("(something)", "something"),
        ("[something]", "something"),
        ("[?/something]", "something"),
        ("#_of_thing", "number_of_thing"),
        ("% notional", "pct_notional"),
        ("with.dots", "with_dots"),
        ("with space", "with_space"),
        ("with ? question mark", "with_question_mark"),
        ("slash/", "slash"),
        ("antislash\\", "antislash"),
        ("quote'", "quote_"),
        ("dash-dash", "dash_dash"),
        ("more\nthan\none\nline", "more_than_one_line"),
    ]

    for s, expected_result in examples:
        assert clean_column(s) == expected_result


@pytest.mark.parametrize("dict_type", (dict, OrderedDict))
def test_dico_key_filter(dict_type):
    def f(x):
        return x >= 1

    dico = {0: "a", 1: "b", 2: "c"}

    dico = dict_type(dico)

    fdico = dico_key_filter(dico, f)

    assert type(dico) == type(fdico)
    assert set(fdico.keys()) == set([k for k, v in dico.items() if f(k)])
    for k, v in fdico.items():
        assert dico[k] == v
    assert id(fdico) != id(dico)


@pytest.mark.parametrize("dict_type", (dict, OrderedDict))
def test_dico_value_filter(dict_type):
    def f(x):
        return x >= 1

    dico = {"a": 0, "b": 1, "c": 2}

    dico = dict_type(dico)

    fdico = dico_value_filter(dico, f)

    assert type(dico) == type(fdico)
    assert set(fdico.keys()) == set([k for k, v in dico.items() if f(v)])
    for k, v in fdico.items():
        assert dico[k] == v
    assert id(fdico) != id(dico)


@pytest.mark.parametrize("dict_type", (dict, OrderedDict))
def test_dico_key_map(dict_type):
    def f(x):
        return x + 1

    dico = {0: "a", 1: "b", 2: "c"}
    dico = dict_type(dico)

    mdico = dico_key_map(dico, f)
    assert type(mdico) == type(dico)
    assert set(mdico.keys()) == set([f(k) for k in dico.keys()])
    for k, v in dico.items():
        assert mdico[f(k)] == v

    assert id(dico) != id(mdico)


@pytest.mark.parametrize("dict_type", (dict, OrderedDict))
def test_dico_value_map(dict_type):
    def f(x):
        return x + 1

    dico = {"a": 0, "b": 1, "c": 2}
    dico = dict_type(dico)

    mdico = dico_value_map(dico, f)
    assert type(mdico) == type(dico)
    assert set(mdico.keys()) == set(dico.keys())
    for k, v in dico.items():
        assert mdico[k] == f(v)

    assert id(dico) != id(mdico)
