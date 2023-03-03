# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:36:15 2018

@author: Lionel Massoulard
"""
import pytest

import numpy as np
from scipy.stats import randint

from sklearn.utils.validation import check_random_state


from aikit.future.automl.hyper_parameters import (
    HyperRangeInt,
    HyperChoice,
    HyperMultipleChoice,
    HyperLogRangeFloat,
    HyperRangeBetaFloat,
    HyperRangeBetaInt,
    HyperRangeFloat,
    HyperListOfDifferentSizes,
    HyperComposition,
    HyperCrossProduct,
    _get_rand,
)


ALL_SIMPLE_RANDOM_DISTRIBUTION = [
    (HyperChoice, {"values": ["a", "b", "c", "d"]}),
    (HyperMultipleChoice, {"possible_choices": ("a", "b", "c", "d", "e")}),
    (HyperRangeInt, {"start": 10, "end": 100}),
    (HyperRangeFloat, {"start": 10, "end": 100}),
    (HyperRangeBetaFloat, {"start": 0, "end": 1, "alpha": 3, "beta": 1}),
    (HyperRangeBetaInt, {"start": 0, "end": 10, "alpha": 3, "beta": 1}),
    (HyperLogRangeFloat, {"start": 1, "end": 1000}),
]


def test_get_rand():
    hp = HyperRangeInt(0, 2, random_state=123)

    assert _get_rand(hp) in (0, 1, 2)
    assert _get_rand(["a", "b", "c"]) in ("a", "b", "c")
    assert _get_rand(1) == 1
    assert _get_rand(randint(1, 3)) in (1, 2)


def test_get_rand_with_random_state_with_seed():
    hp = HyperRangeInt(0, 2)

    # Verif returns always  the same in the case where random_state is fixed
    assert len(set([_get_rand(hp, random_state=123) for _ in range(10)])) == 1
    assert len(set([_get_rand(["a", "b", "c"], random_state=123) for _ in range(10)])) == 1
    assert len(set([_get_rand(randint(1, 3), random_state=123) for _ in range(10)])) == 1


def test_get_rand_with_random_state_with_state():
    random_state = check_random_state(123)
    hp = HyperRangeInt(0, 2, random_state=random_state)

    assert len(set([hp.get_rand() for _ in range(10)])) > 1  # different values
    assert len(set([_get_rand(["a", "b", "c"], random_state=random_state) for _ in range(10)])) > 1
    assert len(set([_get_rand(randint(1, 3), random_state=random_state) for _ in range(10)])) > 1


def test_hyper_range_int():
    hp = HyperRangeInt(10, 100, step=10, random_state=123)
    possible_choices = set([10 * (i + 1) for i in range(10)])
    all_x = []
    for _ in range(100):
        x = hp.get_rand()
        assert isinstance(x, int)
        assert x >= 10
        assert x <= 100
        assert x in possible_choices
        all_x.append(x)

    assert len(set(all_x)) > 1  # I don't want always the same value


def test_hyper_range_beta_int():
    hp = HyperRangeBetaInt(2, 5, random_state=123)

    all_x = []
    for _ in range(100):
        x = hp.get_rand()
        assert isinstance(x, int)
        assert x >= 2
        assert x <= 5
        all_x.append(x)

    assert set(all_x) == {2, 3, 4, 5}


def _all_same(all_gen):
    """ helper function to test if things are all the same """
    if len(all_gen) == 1:
        return True
    for gen in all_gen[1:]:
        if gen != all_gen[0]:
            return False
    # I don't want to use 'set' because thing might not be hashable

    return True


def test_all_same():
    assert _all_same([1, 1, 1])
    assert _all_same(["a", "a", "a"])
    assert not _all_same([1, 2, 3])
    assert not _all_same(["a", "b", "c"])

    assert _all_same([{"a": 10}, {"a": 10}, {"a": 10}])
    assert not _all_same([{"a": 10}, {"a": 10}, {"a": 11}])
    assert not _all_same([{"a": 10}, {"a": 10}, {"b": 11}])


@pytest.mark.parametrize("klass, kwargs", ALL_SIMPLE_RANDOM_DISTRIBUTION)
def test_hyper_random_state(klass, kwargs):
    NB_GEN = 50

    # . Test not everything is the same
    hp = klass(random_state=123, **kwargs)
    all_gen = [hp.get_rand() for _ in range(NB_GEN)]
    assert not _all_same(all_gen)

    # . Regenerate things and test exactly same values
    hp = klass(random_state=123, **kwargs)
    all_gen2 = [hp.get_rand() for _ in range(NB_GEN)]

    assert all_gen == all_gen2

    # . Regenerate but pass a random generator
    random_state = check_random_state(123)
    hp = klass(random_state=random_state, **kwargs)
    all_gen3 = [hp.get_rand() for _ in range(NB_GEN)]
    assert all_gen == all_gen3

    # . Regenerate by setting random_state
    hp.random_state = 123
    all_gen4 = [hp.get_rand() for _ in range(NB_GEN)]
    assert all_gen == all_gen4

    return all_gen


def test_hyper_list_of_different_sizes():
    def klass(random_state):
        return HyperListOfDifferentSizes(
            nb_dist=HyperRangeInt(1, 5), value_dist=HyperRangeInt(50, 150), random_state=random_state
        )

    all_gen = test_hyper_random_state(klass, {})

    for res in all_gen:
        assert isinstance(res, list)
        assert len(res) <= 5
        assert len(res) >= 1

        for r in res:
            assert isinstance(r, int)
            assert r >= 50
            assert r <= 150


def test_hyper_composition():
    def klass(random_state):
        return HyperComposition([HyperRangeInt(0, 100), HyperRangeInt(200, 1000)], random_state=random_state)

    all_gen = test_hyper_random_state(klass, {})

    for gen in all_gen:
        assert isinstance(gen, int)
        assert (0 <= gen <= 100) or (200 <= gen <= 1000)

    def klass(random_state):
        return HyperComposition(
            [(0.9, HyperRangeInt(0, 100)), (0.1, HyperRangeInt(200, 1000))], random_state=random_state
        )

    test_hyper_random_state(klass, {})
    for gen in all_gen:
        assert isinstance(gen, int)
        assert (0 <= gen <= 100) or (200 <= gen <= 1000)

    def klass(random_state):
        return HyperComposition([(0.9, "choice_a"), (0.1, "choice_b")], random_state=random_state)

    test_hyper_random_state(klass, {})


def test_hyper_cross_product():
    # 1. list of distribution input
    def klass(random_state):
        return HyperCrossProduct([HyperRangeInt(0, 10), HyperRangeFloat(0, 1)], random_state=random_state)

    all_gen = test_hyper_random_state(klass, {})
    for gen in all_gen:
        assert isinstance(gen, list)
        assert len(gen) == 2
        g1, g2 = gen
        assert isinstance(g1, int)
        assert 0 <= g1 <= 10
        assert isinstance(g2, float)
        assert 0 <= g2 <= 1

    # 2. dico of distribution
    def klass(random_state):
        return HyperCrossProduct(
            {"int_value": HyperRangeInt(0, 10), "float_value": HyperRangeFloat(0, 1)}, random_state=random_state
        )

    all_gen = test_hyper_random_state(klass, {})
    for gen in all_gen:
        assert isinstance(gen, dict)
        assert set(gen.keys()) == {"int_value", "float_value"}
        g1 = gen["int_value"]
        g2 = gen["float_value"]
        assert isinstance(g1, int)
        assert 0 <= g1 <= 10
        assert isinstance(g2, float)
        assert 0 <= g2 <= 1

    # 3. dico of distribution with choice and constant
    def klass(random_state):
        return HyperCrossProduct(
            {
                "int_value": randint(0, 10),
                "float_value": HyperRangeFloat(0, 1),
                "choice": ("a", "b", "c"),
                "constant": 10,
            },
            random_state=random_state,
        )

    all_gen = test_hyper_random_state(klass, {})
    for gen in all_gen:
        assert isinstance(gen, dict)
        assert set(gen.keys()) == {"int_value", "float_value", "choice", "constant"}
        g_int = gen["int_value"]
        g_float = gen["float_value"]
        g_choice = gen["choice"]
        g_constant = gen["constant"]

        assert isinstance(g_int, int)
        assert 0 <= g_int <= 9
        assert isinstance(g_float, float)
        assert 0 <= g_float <= 1
        assert g_choice in ("a", "b", "c")
        assert g_constant == 10


def test_addition_hyper_cross_product():

    # Test 1: HyperCrossProduct + HyperCrossProduct
    hh1 = HyperCrossProduct({"a": [0, 1, 2], "c": HyperLogRangeFloat(0.01, 1)})
    hh2 = HyperCrossProduct({"d": ["aaaa", "bbb", None], "e": HyperRangeInt(0, 100)})
    hh12 = hh1 + hh2

    assert isinstance(hh12, HyperCrossProduct)

    res = [hh12.get_rand() for _ in range(10)]
    for r in res:
        assert isinstance(r, dict)
        assert set(r.keys()) == {"a", "c", "d", "e"}
        assert r["a"] in (0, 1, 2)
        assert r["d"] in ("aaaa", "bbb", None)

    # Test 2: HyperCrossProduct + dict
    hh1 = HyperCrossProduct({"a": [0, 1, 2], "c": HyperLogRangeFloat(0.01, 1)})

    hh2 = {"d": ["aaaa", "bbb", None], "e": HyperRangeInt(0, 100)}
    hh12 = hh1 + hh2
    assert isinstance(hh12, HyperCrossProduct)

    res = [hh12.get_rand() for _ in range(10)]
    for r in res:
        assert isinstance(r, dict)
        assert set(r.keys()) == {"a", "c", "d", "e"}
        assert r["a"] in (0, 1, 2)
        assert r["d"] in ("aaaa", "bbb", None)

    # Test 3: HyperCrossProduct update
    hh1 = HyperCrossProduct({"a": [0, 1, 2], "c": HyperLogRangeFloat(0.01, 1)})

    hh2 = {"a": [10, 11, 12], "e": HyperRangeInt(0, 100)}
    # 'a' will be updated

    hh12 = hh1 + hh2
    assert isinstance(hh12, HyperCrossProduct)

    res = [hh12.get_rand() for _ in range(10)]
    for r in res:
        assert isinstance(r, dict)
        assert set(r.keys()) == {"a", "c", "e"}
        assert r["a"] in (10, 11, 12)
        assert r["a"] not in (0, 1, 2)


def test_addition_hyper_composition():

    # Test 1: regular composition
    hh1 = HyperComposition(
        [
            (0.5, HyperCrossProduct({"a": [0, 1, 2], "c": HyperLogRangeFloat(0.01, 1)})),
            (0.5, HyperCrossProduct({"a": [10, 11, 12], "c": HyperLogRangeFloat(0.01, 1)})),
        ]
    )
    np.random.seed(123)

    res = [hh1.get_rand() for _ in range(50)]
    for r in res:
        assert isinstance(r, dict)
        assert set(r.keys()) == {"a", "c"}
        assert r["a"] in (0, 1, 2, 10, 11, 12)
        assert r["c"] <= 1
        assert r["c"] >= 0.01

    assert set([r["a"] for r in res]) == {0, 1, 2, 10, 11, 12}
    # with this seed + 50 observations I should to have everything
    # ... with any having something missing has really low proba

    # Test 2: HyperComposition + params
    hh2 = HyperCrossProduct({"d": HyperRangeFloat(1.1, 10)})
    hh12 = hh1 + hh2
    assert isinstance(hh12, HyperComposition)

    res = [hh12.get_rand() for _ in range(50)]
    for r in res:
        assert isinstance(r, dict)
        assert set(r.keys()) == {"a", "c", "d"}
        assert r["a"] in (0, 1, 2, 10, 11, 12)
        assert r["c"] <= 1
        assert r["c"] >= 0.01
        assert r["d"] <= 10
        assert r["d"] >= 1.1

    assert set([r["a"] for r in res]) == {0, 1, 2, 10, 11, 12}
    # with this seed + 50 observations I should to have everything

    # Test 3: HyperComposition update
    for is_hyper in (True, False):
        hh2 = {"c": HyperRangeFloat(1.1, 10)}
        if is_hyper:
            hh2 = HyperCrossProduct(hh2)

        hh12 = hh1 + hh2
        assert isinstance(hh12, HyperComposition)
        res = [hh12.get_rand() for _ in range(50)]
        for r in res:
            assert isinstance(r, dict)
            assert set(r.keys()) == {"a", "c"}
            assert r["a"] in (0, 1, 2, 10, 11, 12)
            assert r["c"] <= 10
            assert r["c"] >= 1.1

        assert set([r["a"] for r in res]) == {0, 1, 2, 10, 11, 12}
        # with this seed + 50 observations I should to have everything
