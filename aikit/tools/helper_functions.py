# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 12:28:59 2018

@author: Lionel Massoulard

This file contains helper function to facilitate and fasten lots of small python operations
"""

import pandas as pd
import numpy as np

from decorator import decorate
import sys
import inspect

from collections import OrderedDict

import json
import pickle

import re
import os
from io import StringIO
import hashlib

from sklearn.utils import check_random_state, safe_indexing

from aikit.tools.json_helper import SpecialJSONEncoder


def function_has_named_argument(f, attr):
    """ return True if a given function accept a given attribute """
    return attr in inspect.getfullargspec(f).args


def diff(list1, list2):
    """ difference list1 minus list2 """
    res = [l for l in list1 if l not in list2]  # copied from useful functions
    if isinstance(list1, tuple):
        res = tuple(res)
    return res


def intersect(list1, list2):
    """ intersection of 2 lists """
    res = [l for l in list1 if l in list2]
    if isinstance(list1, tuple):
        res = tuple(res)
    return res


def not_none(original_list):
    """ filter : not None only """
    return list(filter(lambda x: not pd.isnull(x), original_list))


def lunique(original_list):
    """ list unique """
    result = []
    for o in original_list:
        if o not in result:
            result.append(o)

    return result
    # je fais ca plutot que de list(set(...)) pour garder l'ordre


def unlist(list_of_list):
    """ transform a list of list into one list with all the elements 
    
    Examples
    --------
    >>> unlist([[1,10],[32]])
    [1,10,32]
    >>> unlist([[10],[11],[],[45]])
    [10,11,45]
    """

    res = []
    for l in list_of_list:
        res += l
    return res
    # return reduce(lambda x,y:x+y,list_of_list,[])


def deep_flatten(nested_object):
    """ flatten a nested list of object into a simple list """
    result = []
    for ob in nested_object:
        if isinstance(ob, (list, tuple, set)):
            result += deep_flatten(ob)
        else:
            result.append(ob)

    return result


def unnest_tuple(nested_tuple):
    """ helper function to unnested a nested tuple """

    def _rec(nested_tuple):

        if isinstance(nested_tuple, (tuple, list)):
            res = []
            for o in nested_tuple:
                res += _rec(o)

            return res
        else:
            return [nested_tuple]

    return tuple(_rec(nested_tuple))


def tuple_include(t1, t2):
    return all([t in t2 for t in t1])


def dico_copy(dico):
    """ dictionnary (shallow) copy """
    if isinstance(dico, OrderedDict):
        return OrderedDict([(k, v) for k, v in dico.items()])

    elif isinstance(dico, dict):
        return {k: v for k, v in dico.items()}  # So that I use fast dico completion when possible

    else:
        res = dico.__class__()
        for k, v in dico.items():
            res[k] = v
        return res


def dico_add(*dicos):
    """ dictionnaries addition, with (shallow) copy """
    if len(dicos) == 0:
        return {}

    res = dicos[0].__class__()
    for dico in dicos:
        res.update(dico)

    return res


def dico_keyvalue_filter(dico, f):
    """ filter a dictionnary according to a function on its keys and values """
    if isinstance(dico, dict):
        return {k: v for k, v in dico.items() if f(k, v)}

    elif isinstance(dico, OrderedDict):
        return OrderedDict([(k, v) for k, v in dico.items() if f(k, v)])

    else:
        res = dico.__class__()
        for k, v in res.items():
            if f(k, v):
                res[k] = v
        return res


def dico_key_filter(dico, f):
    """ filter a dictionnary according to a function on its keys 
    keep original type of dict
    """
    if isinstance(dico, OrderedDict):
        return OrderedDict([(k, v) for k, v in dico.items() if f(k)])

    elif isinstance(dico, dict):
        return {k: v for k, v in dico.items() if f(k)}

    else:
        res = dico.__class__()
        for k, v in res.items():
            if f(k):
                res[k] = v
        return res


def dico_value_filter(dico, f):
    """ filter a dictionnary according to a function on its values """
    if isinstance(dico, OrderedDict):
        return OrderedDict([(k, v) for k, v in dico.items() if f(v)])

    elif isinstance(dico, dict):
        return {k: v for k, v in dico.items() if f(v)}

    else:
        res = dico.__class__()
        for k, v in res.items():
            if f(v):
                res[k] = v
        return res


def dico_key_map(dico, f):
    """ apply a function on the key of a dictionnary """
    if isinstance(dico, OrderedDict):
        return OrderedDict([(f(k), v) for k, v in dico.items()])

    elif isinstance(dico, dict):
        return {f(k): v for k, v in dico.items()}

    else:
        res = dico.__class__()
        for k, v in res.items():
            res[f(k)] = v
        return res


def dico_value_map(dico, f):
    """ apply a function on the values of a dictionnary """
    if isinstance(dico, OrderedDict):
        return OrderedDict([(k, f(v)) for k, v in dico.items()])
    elif isinstance(dico, dict):
        return {k: f(v) for k, v in dico.items()}
    else:
        res = dico.__class__()
        for k, v in res.items():
            res[k] = f(v)
        return res


# def deep_apply(obj, fun):
#    """ this function apply a given function to all the elements, recursively going deeper in the object """
#    if isinstance(obj,list):
#        return [deep_apply(o,fun) for o in obj]
#
#    elif isinstance(obj,tuple):
#        return tuple([deep_apply(o,fun) for o in obj])
#
#    elif isinstance(obj,(dict,OrderedDict)):
#
#        res = obj.__class__()
#        for k,v in obj.items():
#            res[ deep_apply(k, fun) ] = deep_apply(v, fun)
#
#        return res
#
#    elif isinstance(obj,set):
#        return set((fun(o) for o in obj))
#
#    else:
#        return fun(obj)
#
# def remove_nan(x):
#    if hasattr(x,"shape"):
#        return x
#    try:
#        isn = pd.isnull(x)
#    except (ValueError,TypeError):
#        isn = False
#
#    if isn:
#        return None
#    else:
#        return x
#
# example = {"proba":np.nan,"result":[1,2,np.nan,None]}
# deep_apply(example,remove_nan)
#
# example = {"proba":np.nan,"result":np.array([1,2,3])}
# deep_apply(example,remove_nan)


def save_json(obj, fname):
    """ saves a json on a file
    
    Parameters
    ----------
    obj : object
        the object that we want to saved, must be json serializable
        
    fname : str
        the path on the drive to save it
        
    """
    with open(fname, "w") as f:
        json.dump(obj, f, indent=4)


def load_json(fname):
    """ loads a json from a file
    
    Parameters
    ----------
    fname : str
        the path of the file to load
        
    Returns
    -------
    the python object corresponding to the json
    
    """
    with open(fname, "r") as f:
        obj = json.load(f)
    return obj


def save_pkl(obj, fname):
    """ saves a pickle object on a file
    
    Parameters
    ----------
    obj : object
        the python object that we want to serialize (must be picklable


    fname : str
        the pat on the drive to save it
        
    
    """
    with open(fname, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(fname):
    """Loads a pickle object from the drive
    
    Parameters
    ----------
    fname : str
        the path of the file to load
        
    Returns
    -------
    the python object un-pickled
    """
    with open(fname, "rb") as f:
        obj = pickle.load(f)
    return obj


def pd_match(to_match, values, na_sentinel=-1):
    """ un-optimized matching function """
    dico = {}
    for i, v in enumerate(values):
        if v not in dico:
            dico[v] = i

    def f(v):
        try:
            res = dico[v]
        except KeyError:
            res = na_sentinel

        return res

    index = np.array([f(v) for v in to_match])
    return index


def clean_column(s):
    """ utils function that clean a string to be a cleaner name of column
    
    Parameter
    ---------
    s : string
        the string to clean
        
    Return
    ------
    cleaned string
    
    """

    if s is None:
        return None

    r = s.strip().lower()
    r = re.sub(r"[?\(\)/\[\]\\]", "", r)
    r = re.sub(r"[:' \-\.\n]", "_", r)
    r = re.sub("_+", "_", r)
    r = r.replace("#", "number")
    r = r.replace("%", "pct")
    r = r.replace("$", "usd")
    r = r.replace("&", "_and_")
    r = r.replace("€", "eur")
    r = r.replace("£", "gbp")
    r = r.replace("¥", "jpy")

    return r


def system_and_caller_information():
    """ retrieve information about the caller
    Useful if a same thing is launch by severals processes 
    
    Returns
    -------
    dictionnary with :
    
    computer : name of the computer
    file     : path of the function
    system   : os that runs the code
    user     : name of the user
    version  : version of python
    
    Examples
    --------
    >>> system_and_caller_information()
    {'computer': 'name_of_computer',
     'file'    : 'helper_functions.py',
     'system'  : 'win32',
     'user'    : 'current_user',
     'version' : '3.6.4'
        
    """
    from platform import node, python_version
    from sys import platform
    from getpass import getuser

    res = {
        "system": platform,
        "computer": node(),
        "user": getuser(),
        "file": __file__,
        "version": python_version(),
        "pid": os.getpid(),
    }

    return res


def exception_improved_logging(f):
    """ decorator to increment the error message with the name of the class in which it appears """
    # useful when using inherited class

    def _exception_handling(f, self, *args, **kwargs):
        try:
            return f(self, *args, **kwargs)
        except Exception as e:
            raise type(e)(str(e) + " (within %s)" % self.__class__).with_traceback(sys.exc_info()[2])

    return decorate(f, _exception_handling)


def _is_number(x):
    """ small function to test if something is a python number """
    return issubclass(type(x), (float, int, np.number))


def shuffle_all(*args, seed=None):
    """ shuffle its arguments (same shuffling for each)
    
    Parameters
    ----------
    
    *args : array, matrix, DataFrame, Serie, ... 
        things to shuffle
        
    seed : optional
        the seed to use
        
    Returns
    -------
    if all the things shuffled
    
    Examples
    --------
    >>> shuffle_all([1,2,3,4,5], seed=123)
    [2, 4, 5, 1, 3]
    >>> shuffle_all(np.array([1,2,3,4,5]),seed=123)
    np.array([2, 4, 5, 1, 3])
    >>> shuffle_all(np.array([1,2,3,4,5]),np.array(["1","2","3","4","5"]),seed=123)
    [array([2, 4, 5, 1, 3]), array(['2', '4', '5', '1', '3']
    >>> shuffle_all(np.array([[1,10],[2,20],[3,30],[4,40],[5,50]]), seed = 123)
    array([[ 2, 20],
           [ 4, 40],
           [ 5, 50],
           [ 1, 10],
           [ 3, 30]])
    """

    rand = check_random_state(seed)

    def nelements(arg):
        if hasattr(arg, "shape"):
            return arg.shape[0]
        else:
            return len(arg)

    larg = [nelements(arg) for arg in args]
    if set(larg) != set([larg[0]]):
        raise ValueError("Different length")

    N = larg[0]
    ii = np.arange(N)
    rand.shuffle(ii)

    shuffled_args = [safe_indexing(arg, ii) for arg in args]

    if len(args) == 1:
        return shuffled_args[0]
    else:
        return shuffled_args


def md5_hash(ob):
    """ hash of an object, constant across computers/process
    works with non hashable python object
    """

    s = StringIO()
    json.dump(ob, s, cls=SpecialJSONEncoder)
    m = hashlib.md5()
    m.update(s.getvalue().encode("utf-8"))
    return m.hexdigest()
