# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:30:27 2018

@author: Lionel Massoulard
"""

import copy

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


# def param_from_sklearn_model(model, _simplify_default = False):
#
#    if isinstance(model,Pipeline):
#        return (SpecialModels.Pipeline,{"steps":[(name,param_from_model(step)) for name,step in model.steps]})
#
#    elif isinstance(model,ModelsUnion):
#        return (SpecialModels.ModelsUnion ,{"transformer_list":[(name,param_from_model(step, _simplify_default = _simplify_default)) for name,step in model.transformer_list],
#                                "n_jobs":model.n_jobs,
#                                "transformer_weights":model.transformer_weights
#                                })
#
#    elif isinstance(model, GraphPipeline):
#        return (SpecialModels.GraphPipeline , {n:param_from_model(p) for n,p in model.models.items() } , model.edges)
#
#
#    elif isinstance(model,BaseEstimator) and model.__class__.__name__ in MODEL_REGISTER.dico_name_class:
#        if not _simplify_default:
#            param_dico = {k:param_from_model(v,_simplify_default = _simplify_default) for k,v in model.get_params().items() }
#        else:
#            # Experimental
#            default_params = _get_default_params(model.__class__)
#            param_dico = {}
#            for k,v in model.get_params().items():
#                if not (k in default_params and v == default_params[k]):
#                    param_dico[k] = param_from_model(v, _simplify_default = _simplify_default)
#
#        return (model.__class__.__name__,param_dico)
#        # Ici : peut etre faire un filtre si on a les valeurs par default ?
#
#    elif isinstance(model, (dict,OrderedDict)):
#        res = model.__class__()
#        for k,v in model.items():
#            res[k] = param_from_model(v, _simplify_default = _simplify_default)
#
#        return res
#
#    elif isinstance(model,list):
#        return [param_from_model(v,_simplify_default = _simplify_default) for v in model]
#
#    elif isinstance(model,tuple):
#        return tuple([param_from_model(v,_simplify_default = _simplify_default) for v in model])
#
#    elif isinstance(model,(np.int64,np.int32)):
#        return int(model)
#
#    elif isinstance(model,(np.float64,np.float32)):
#        return float(model)
#
#    else:
#        return model
# In[]
