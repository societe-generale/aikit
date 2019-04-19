# -*- coding: utf-8 -*-
"""
Created on Wed May  9 09:11:17 2018

@author: Lionel Massoulard
"""

# import logging
# logger = logging.getLogger(__name__)

# def log_something_from_json(msg = "something from json"):
#    logger.info(msg)

from collections import OrderedDict
import json


def json_encoder(ob):
    """ method to transform an object into a version of it 
    that won't loose information when saved in json
    """
    if isinstance(ob, dict):

        return {"__items__": [(json_encoder(k), json_encoder(v)) for k, v in ob.items()], "__type__": "__dict__"}

    elif isinstance(ob, OrderedDict):

        return {"__items__": [(json_encoder(k), json_encoder(v)) for k, v in ob.items()], "__type__": "__OrderedDict__"}

    elif isinstance(ob, set):
        return {"__items__": [json_encoder(v) for v in ob], "__type__": "__set__"}

    elif isinstance(ob, tuple):
        return {"__items__": [json_encoder(v) for v in ob], "__type__": "__tuple__"}

    elif isinstance(ob, list):
        return [json_encoder(v) for v in ob]

    else:
        return ob


def json_decoder(ob):
    """ method to decode an object into it's real python representation """
    if isinstance(ob, dict):
        t = ob.get("__type__", None)
        if t is not None:
            if t == "__tuple__":
                return tuple([json_decoder(v) for v in ob["__items__"]])

            elif t == "__dict__":
                return {json_decoder(k): json_decoder(v) for k, v in ob["__items__"]}

            elif t == "__OrderedDict__":
                return OrderedDict([(json_decoder(k), json_decoder(v)) for k, v in ob["__items__"]])

            elif t == "__set__":
                return set([json_decoder(v) for v in ob["__items__"]])

            else:
                raise TypeError("I don't know how to decode this '__type__' : %s" % t)

        return ob
    elif isinstance(ob, list):
        return [json_decoder(o) for o in ob]

    else:
        return ob


class SpecialJSONEncoder(json.JSONEncoder):
    """ JSON Encoder class """

    def iterencode(self, o, _one_shot=False):
        return super(SpecialJSONEncoder, self).iterencode(json_encoder(o), _one_shot=_one_shot)

    def encode(self, o):
        return super(SpecialJSONEncoder, self).encode(json_encoder(o))


class SpecialJSONDecoder(json.JSONDecoder):
    """ JSON Decoder class """

    def decode(self, s):
        return json_decoder(super(SpecialJSONDecoder, self).decode(s))


def save_json(obj, fname, **kwargs):
    """ save a json into a file, using the custom Encoder class """
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, cls=SpecialJSONEncoder, **kwargs)


def load_json(fname, **kwargs):
    """ read a json from a file, using the custom Decoder class """
    with open(fname, "r", encoding="utf-8") as f:
        obj = json.load(f, cls=SpecialJSONDecoder)

    return obj


# In[]
