from collections import OrderedDict

import pytest

from aikit.future.util.dict import filter_dict_on_keys, filter_dict_on_values, map_dict_keys, map_dict_values


@pytest.mark.parametrize("dict_type", (dict, OrderedDict))
def test_dico_key_filter(dict_type):
    def f(x):
        return x >= 1

    d = {0: "a", 1: "b", 2: "c"}
    d = dict_type(d)

    filtered_dict = filter_dict_on_keys(d, f)

    assert type(d) == type(filtered_dict)
    assert set(filtered_dict.keys()) == set([k for k, v in d.items() if f(k)])
    for k, v in filtered_dict.items():
        assert d[k] == v
    assert id(filtered_dict) != id(d)


@pytest.mark.parametrize("dict_type", (dict, OrderedDict))
def test_dico_value_filter(dict_type):
    def f(x):
        return x >= 1

    d = {"a": 0, "b": 1, "c": 2}

    d = dict_type(d)

    filtered_dict = filter_dict_on_values(d, f)

    assert type(d) == type(filtered_dict)
    assert set(filtered_dict.keys()) == set([k for k, v in d.items() if f(v)])
    for k, v in filtered_dict.items():
        assert d[k] == v
    assert id(filtered_dict) != id(d)


@pytest.mark.parametrize("dict_type", (dict, OrderedDict))
def test_dico_key_map(dict_type):
    def f(x):
        return x + 1

    d = {0: "a", 1: "b", 2: "c"}
    d = dict_type(d)

    mapped_dict = map_dict_keys(d, f)
    assert type(mapped_dict) == type(d)
    assert set(mapped_dict.keys()) == set([f(k) for k in d.keys()])
    for k, v in d.items():
        assert mapped_dict[f(k)] == v

    assert id(d) != id(mapped_dict)


@pytest.mark.parametrize("dict_type", (dict, OrderedDict))
def test_dico_value_map(dict_type):
    def f(x):
        return x + 1

    d = {"a": 0, "b": 1, "c": 2}
    d = dict_type(d)

    mapped_dict = map_dict_values(d, f)
    assert type(mapped_dict) == type(d)
    assert set(mapped_dict.keys()) == set(d.keys())
    for k, v in d.items():
        assert mapped_dict[k] == f(v)

    assert id(d) != id(mapped_dict)
