# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:33:30 2018

@author: Lionel Massoulard
"""

import pandas as pd
import numpy as np

from aikit.tools.helper_functions import (
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
)


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
        ("already_clean", "already_clean"),
        ("too_many_dash___", "too_many_dash_"),
        ("$ and €", "usd_and_eur"),
        ("(something)", "something"),
        ("[something]", "something"),
        ("[?/something]", "something"),
        ("#_of_thing", "number_of_thing"),
        ("% notional", "pct_notional"),
    ]

    for s, expected_result in examples:
        assert clean_column(s) == expected_result
