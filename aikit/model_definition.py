# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:30:27 2018

@author: Lionel Massoulard
"""

import copy
import inspect

from sklearn.base import BaseEstimator
import numpy as np

from aikit.model_registration import DICO_NAME_KLASS
from aikit.enums import SpecialModels


def _is_graphpipeline(param):
    """ from a json param, detects if it represents a GraphPipeline model """
    if not isinstance(param, tuple):
        return False

    if len(param) not in (2, 3):
        return False

    if not isinstance(param[0], str):
        return False

    if param[0] != SpecialModels.GraphPipeline:
        return False

    if not isinstance(param[1], (dict, list)):
        return False

    return True


def _is_model(param):
    """ does the param represent a simple model """
    if not isinstance(param, (tuple, list)):
        return False, None

    if len(param) == 0:
        return False, None

    try:
        model_klass = DICO_NAME_KLASS.get(param[0], None)
    except TypeError:
        # Can arise if I try to hash something that isn't hashable
        model_klass = None

    if model_klass is not None:
        return True, model_klass

    return False, None


# def param_from_sklearn_model(model):
#    """ convert a sklearn-like model into a json-like parameters """
# TODO


def sklearn_model_from_param(param, _copy=True):
    """ convert a parameter into a sklearn model 
    
    The syntax is that a model is represented by a 2-uple with its name and its arguments.
    
    Example
    -------
    
    >>> sklearn_model_from_param( ("RandomForestClassifier",{"n_estimators":100}))
    >>> RandomForestClassifier(n_estimators=100)
    
    
    """

    if _copy:
        param = copy.deepcopy(param)  # Internal copy

    model_node, model_klass = _is_model(param)

    if model_node and param[0] == SpecialModels.GraphPipeline:

        ##########################
        ### GraphPipeline node ###
        ##########################

        rest_param = param[1:]
        list_args = []

        for i, arg in enumerate(rest_param[:-1]):
            if i == 0:
                list_args.append(sklearn_model_from_param(arg, _copy=False))
            else:
                # Second argument is edges => I don't want to translate it
                list_args.append(arg)

        # If last attribute is a dict, it is named arguments
        if isinstance(rest_param[-1], dict):
            dict_args = rest_param[-1]
            for k, v in dict_args.items():
                if k != "edges":
                    dict_args[k] = sklearn_model_from_param(v, _copy=False)
        else:
            # Otherwise : just a regular param
            dict_args = {}
            if len(rest_param) == 1:
                list_args.append(sklearn_model_from_param(rest_param[-1], _copy=False))
            else:
                list_args.append(rest_param[-1])

        return model_klass(*list_args, **dict_args)

    elif model_node and param[0] != SpecialModels.GraphPipeline:

        ############################
        ### Classical model node ###
        ############################

        rest_param = param[1:]

        # If last attribute is a dict, it is named arguments
        if isinstance(rest_param[-1], dict):
            list_args = list(rest_param[:-1])
            dict_args = rest_param[-1]
        else:
            list_args = list(rest_param)
            dict_args = {}

        return model_klass(
            *sklearn_model_from_param(list_args, _copy=False), **sklearn_model_from_param(dict_args, _copy=False)
        )

    elif isinstance(param, dict):

        ###################
        ### Dictionnary ###
        ###################

        res = param.__class__()
        for k, v in param.items():
            res[k] = sklearn_model_from_param(v, _copy=False)

        return res

    elif isinstance(param, list):

        ############
        ### List ###
        ############

        return [sklearn_model_from_param(v, _copy=False) for v in param]

    elif isinstance(param, tuple):

        #############
        ### Tuple ###
        #############

        return tuple([sklearn_model_from_param(v, _copy=False) for v in param])
    else:
        return param


def filtered_get_params(model, simplify_default=True):

    if not simplify_default:
        return model.get_params(deep=False)
    
    params = model.get_params(deep=False)
    new_params = params.__class__()

    args = inspect.signature(model.__class__)
    for param, value in params.items():
        skip=False
        if param in args.parameters:
            if value == args.parameters[param].default:
                skip=True
        if not skip:
            new_params[param]=value
            
    return new_params


def param_from_sklearn_model(model, simplify_default=True):
    """ convert a sklearn model into a its json representation
    
    Parameters
    ----------
    model : sklearn.BaseEstimator
        the model to convert
        
    simplify_default : boolean, default=True
        if True will simplify the arguments that are identical to the default one
        
    Returns
    -------
    model json representation
    
    
    Example
    -------
    >>> model = RandomForestClassifier(n_estimators=200)
    >>> param_from_sklearn_model(model)
    >>> ('RandomForestClassifier', {'n_estimators': 200})

    
    """
    if isinstance(model, BaseEstimator):
        if  model.__class__.__name__ not in DICO_NAME_KLASS._mapping:
            print(f"You'll need to include your class '{model.__class__.__name__}' into the register to be able to reload it")

        if simplify_default:
            param_dico = {k:param_from_sklearn_model(v, simplify_default=simplify_default) for k,v in filtered_get_params(model, simplify_default=True).items() } 
        else:
            param_dico = {k:param_from_sklearn_model(v, simplify_default=simplify_default) for k,v in model.get_params(deep=False).items() } 
        
        return (model.__class__.__name__, param_dico)
    
    
    elif isinstance(model, dict):
        res = model.__class__() # to keep the same format (dict, OrderedDict)
        for k,v in model.items():
            res[k] = param_from_sklearn_model(v, simplify_default=simplify_default)
    
        return res
    
    elif isinstance(model, list):
        return [param_from_sklearn_model(v, simplify_default=simplify_default) for v in model]
    
    elif isinstance(model, tuple):
        return tuple([param_from_sklearn_model(v, simplify_default=simplify_default) for v in model])
    
    elif isinstance(model, np.number):
        if model.dtype.kind == "i":
            return int(model)

        elif model.dtype.kind == "f":
            return float(model)
        
        else:
            return model
    
    elif isinstance(model, np.bool_):
        return bool(model)
    
    elif isinstance(model, np.str_):
        return str(model)
    
    else:
        return model