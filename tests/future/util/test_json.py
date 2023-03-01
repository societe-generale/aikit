# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:38:58 2018

@author: Lionel Massoulard
"""
from io import StringIO
import json
from collections import OrderedDict

from aikit.future.util.json import json_encoder, json_decoder, SpecialJSONEncoder, SpecialJSONDecoder


def test_json_encoder_decoder():

    def _test_one_object(ob):
        s = StringIO()
        ob2 = json_encoder(ob)
        json.dump(ob2, s, indent=4)
        # print(s.getvalue())

        s2 = StringIO(s.getvalue())

        ob3 = json.load(s2)
        ob4 = json_decoder(ob3)
        assert ob4 == ob

    def _test_one_object2(ob):
        s = StringIO()
        json.dump(ob, s, indent=4, cls=SpecialJSONEncoder)
        s2 = StringIO(s.getvalue())
        ob2 = json.load(s2, cls=SpecialJSONDecoder)
        assert ob == ob2

    list_of_tests = [
        ("a", "b", "c"),
        ["a", "b", "c"],
        {1: 2},
        {"1": "a"},
        (1, 2),
        [1, 2],
        {(1, 2): (3, 4), "a": "A", "c": None, 4: [10, 11]},
        OrderedDict([(2, 1), (3, 4)]),
        {"nested_dict": {"a": [1, 2, 3], "b": (1, 2, 3)}},
    ]

    for obj in list_of_tests:
        _test_one_object(obj)
        _test_one_object2(obj)

    _test_one_object(list_of_tests)
    _test_one_object2(list_of_tests)
