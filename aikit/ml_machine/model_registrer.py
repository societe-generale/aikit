# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 08:38:10 2018

@author: Lionel Massoulard
"""

from inspect import signature

from aikit.ml_machine import hyper_parameters as hp
from aikit.model_definition import DICO_NAME_KLASS


# In[] : test generic model definition
def get_init_parameters(klass):
    """ get the parameter of the init of a given klass """
    init = getattr(klass.__init__, "deprecated_original", klass.__init__)
    if init is object.__init__:
        return {}

    init_signature = signature(init)
    parameters = [p for p in init_signature.parameters.values() if p.name != "self" and p.kind != p.VAR_KEYWORD]

    return {p.name: p.default for p in parameters}


def singleton(cls):
    """ singleton decorator """
    instances = {}

    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return getinstance


@singleton
class _MODEL_REGISTER(object):
    """ singleton class to store registered models """

    def __init__(self):
        self.reset()

    def reset(self):
        self.hyper_parameters = {}
        self.default_hyper_parameters = {}
        self.init_parameters = {}
        self.informations = {}
        self.all_registered = []

    def register_new_class(self, category, klass, hyper=None, default_hyper=None, **kwargs):

        if not isinstance(klass, type):
            raise TypeError("klass should be klass")

        key = category, klass.__name__

        if key in self.all_registered:
            raise ValueError("%s has already been registered" % str(key))

        self.all_registered.append(key)

        self.init_parameters[key] = get_init_parameters(klass)

        if hyper is not None:
            self.hyper_parameters[key] = hyper
            
        if default_hyper is not None:
            self.default_hyper_parameters[key] = default_hyper

        if kwargs:
            self.informations[key] = {k: v for k, v in kwargs.items()}

        if klass.__name__ not in DICO_NAME_KLASS._mapping:
            raise ValueError(
                "You should also register that klass : %s within the 'simple register' file" % klass.__name__
            )

        return self

    def __repr__(self):
        all_registered = sorted(self.all_registered, key=lambda x: (x[0], x[1]))
        if len(all_registered) == 0:
            return "no klasses yet"

        categories, klasses = zip(*all_registered)

        max_lengh_step = max([len(s) for s in categories])
        result = ["registred klasses:", ""]
        last_cat = None
        for category, name in zip(categories, klasses):
            if last_cat is not None and last_cat != category:
                result.append("")
                last_cat = category
            elif last_cat is None:
                last_cat = category

            result.append("%s : %s" % (category.ljust(max_lengh_step + 1), name))

        return "\n".join(result)


MODEL_REGISTER = _MODEL_REGISTER()

# MODEL_REGISTER.register_new_class(category="toto",klass= RandomForestClassifier,hyper = {"10"})
# MODEL_REGISTER.register_new_class(category="text",klass= CountVectorizerWrapper,hyper = {"10"})
#

# In[]


def register(klass):
    """ decorator to register a new model """

    if not issubclass(klass, _AbstractModelRepresentation):
        raise TypeError("This function is meant to be used with an '_AbstractModelRepresentation'")

    other = {
        k: v
        for k, v in klass.__dict__.items()
        if not k.startswith("_") and k not in ("name", "klass", "custom_hyper", "default_parameters", "category")
    }

    if klass.category is None:
        raise ValueError("I must specify a category for this klass")

    if klass.klass is None:
        raise ValueError("I must specify a klass for this klass")

    MODEL_REGISTER.register_new_class(
        category=klass.category,
        klass=klass.klass,
        hyper=klass.get_hyper_parameter(),
        default_hyper=klass.get_default_hyper_parameter(),
        **other
    )

    return klass


class _AbstractModelRepresentation(object):
    """ Abstract class to store a model/transformer, its hyper-parameters, ... """

    klass = None
    name = None

    custom_hyper = {}
    default_hyper = {}
    default_parameters = {}
    
    hyper = None

    def __init__(self):
        raise ValueError("this class isn't supposed to be instanciated")

    @classmethod
    def get_hyper_parameter(cls):

        if cls.klass is None:
            raise ValueError("I need a klass")

        if cls.hyper is not None:
            return cls.hyper

        all_params = list(get_init_parameters(cls.klass).keys())

        all_hyper = {}
        for p in all_params:

            if p in cls.custom_hyper:
                all_hyper[p] = cls.custom_hyper[p]

            elif p in cls.default_hyper:
                all_hyper[p] = cls.default_hyper[p]

        return hp.HyperCrossProduct(all_hyper)
    
    @classmethod
    def get_default_hyper_parameter(cls):
        if cls.klass is None:
            raise ValueError("I need a klass")
            
        all_params = list(get_init_parameters(cls.klass).keys())
        
        default_hyper_parameters = {}
        for p in all_params:
            if p in cls.default_parameters:
                default_hyper_parameters[p] = cls.default_parameters[p]
            
            elif p in cls.default_default_hyper:
                default_hyper_parameters[p] = cls.default_default_hyper[p]
                
        return default_hyper_parameters    

MODEL_REGISTER = _MODEL_REGISTER()
