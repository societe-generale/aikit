# -*- coding: utf-8 -*-
"""
Created on Wed May  9 09:11:17 2018

@author: Lionel Massoulard
"""
from collections import OrderedDict
import json


def json_encoder(obj):
    """ method to transform an object into a version of it that won't lose information when saved in json """
    if isinstance(obj, dict):
        return {
            "__items__": [(json_encoder(k), json_encoder(v)) for k, v in obj.items()],
            "__type__": "__dict__"
        }

    elif isinstance(obj, OrderedDict):
        return {
            "__items__": [(json_encoder(k), json_encoder(v)) for k, v in obj.items()],
            "__type__": "__OrderedDict__"
        }

    elif isinstance(obj, set):
        return {
            "__items__": [json_encoder(v) for v in obj],
            "__type__": "__set__"
        }

    elif isinstance(obj, tuple):
        return {
            "__items__": [json_encoder(v) for v in obj],
            "__type__": "__tuple__"
        }

    elif isinstance(obj, list):
        return [json_encoder(v) for v in obj]

    else:
        return obj


def json_decoder(obj):
    """ method to decode an object into it's real python representation """
    if isinstance(obj, dict):
        t = obj.get("__type__", None)
        if t is not None:
            if t == "__tuple__":
                return tuple([json_decoder(v) for v in obj["__items__"]])

            elif t == "__dict__":
                return {json_decoder(k): json_decoder(v) for k, v in obj["__items__"]}

            elif t == "__OrderedDict__":
                return OrderedDict([(json_decoder(k), json_decoder(v)) for k, v in obj["__items__"]])

            elif t == "__set__":
                return set([json_decoder(v) for v in obj["__items__"]])

            else:
                raise TypeError(f"No decoder for '__type__': {t}")

        return obj
    elif isinstance(obj, list):
        return [json_decoder(o) for o in obj]

    else:
        return obj


class SpecialJSONEncoder(json.JSONEncoder):
    """ JSON Encoder class """

    def iterencode(self, o, _one_shot=False):
        return super(SpecialJSONEncoder, self).iterencode(json_encoder(o), _one_shot=_one_shot)

    def encode(self, o):
        return super(SpecialJSONEncoder, self).encode(json_encoder(o))


class SpecialJSONDecoder(json.JSONDecoder):
    """ JSON Decoder class """

    def decode(self, s):  # noqa
        return json_decoder(super(SpecialJSONDecoder, self).decode(s))


def save_json(obj, filename, indent=4, **kwargs):
    """ save a json into a file, using the custom Encoder class """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, cls=SpecialJSONEncoder, **kwargs)


def load_json(filename, **kwargs):
    """ read a json from a file, using the custom Decoder class """
    with open(filename, "r", encoding="utf-8") as f:
        obj = json.load(f, cls=SpecialJSONDecoder)
    return obj
