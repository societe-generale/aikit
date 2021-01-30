# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 08:38:10 2018

@author: Lionel Massoulard
"""

import networkx as nx
from inspect import signature
from aikit.enums import StepCategories
from aikit.automl.model_generation import hyper_parameters as hp
from aikit.model_definition import DICO_NAME_KLASS
from aikit.tools.graph_helper import has_cycle, iter_graph


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
class _MODEL_REGISTER:
    """ singleton class to store registered models """

    def __init__(self):
        self.reset()

    def reset(self):
        self.hyper_parameters = {}
        self.default_hyper_parameters = {}
        self.init_parameters = {}
        self.informations = {}
        self.is_allowed = {}
        self.all_registered = []

        self.step_dependencies = nx.DiGraph()
        self._drawing_order = {}

    def register_new_class(
            self,
            category,
            klass, hyper=None,
            default_hyper=None,
            is_allowed=None,
            depends_on=None,
            **kwargs
    ):

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

        if is_allowed is not None:
            self.is_allowed[key] = is_allowed

        if kwargs:
            self.informations[key] = {k: v for k, v in kwargs.items()}

        if klass.__name__ not in DICO_NAME_KLASS._mapping:
            raise ValueError(
                "You should also register that klass : %s within the 'simple register' file" % klass.__name__
            )

        self.step_dependencies.add_node(category)
        if depends_on is not None:
            if not isinstance(depends_on, (list, tuple, set)):
                depends_on = (depends_on,)

            for depending_step in depends_on:
                if depending_step not in StepCategories.alls:
                    raise ValueError(f"{depending_step} is not a know step")

                self.step_dependencies.add_edge(depending_step, category)

                if has_cycle(self.step_dependencies):
                    raise ValueError(f"adding this dependency {depending_step} -> {category} create a cycle")

                self._drawing_order = {step: n for n, step in enumerate(iter_graph(self.step_dependencies))}

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


def register(klass):
    """ decorator to register a new model """

    if not issubclass(klass, _AbstractModelRepresentation):
        raise TypeError("This function is meant to be used with an '_AbstractModelRepresentation'")

    other = {
        k: v
        for k, v in klass.__dict__.items()
        if not k.startswith("_") and k not in ("name",
                                               "klass",
                                               "custom_hyper",
                                               "default_parameters",
                                               "category",
                                               "is_allowed",
                                               "depends_on"
                                               )
    }

    if klass.category is None:
        raise ValueError("I must specify a category for this klass")

    if klass.klass is None:
        raise ValueError("I must specify a klass for this klass")

    _MODEL_REGISTER().register_new_class(
        category=klass.category,
        klass=klass.klass,
        hyper=klass.get_hyper_parameter(),
        default_hyper=klass.get_default_hyper_parameter(),
        is_allowed=klass.is_allowed,
        depends_on=klass.depends_on,
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

    @classmethod
    def is_allowed(cls, models_by_steps):
        return True


class ModelRepresentationBase(_AbstractModelRepresentation):
    """ class just to store the default HyperParameters """

    default_hyper = {
        "n_components": hp.HyperRangeFloat(start=0.1, end=1, step=0.05),
        # Forest like estimators
        "n_estimators": hp.HyperComposition(
            [
                (0.75, hp.HyperRangeInt(start=25, end=175, step=25)),
                (0.25, hp.HyperRangeInt(start=200, end=1000, step=100)),
            ]
        ),
        "max_features": hp.HyperComposition(
            [(0.25, ["sqrt", "auto"]), (0.75, hp.HyperRangeBetaFloat(start=0, end=1, alpha=3, beta=1))]
        ),
        "max_depth": hp.HyperChoice([None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 50, 100]),
        "min_samples_split": hp.HyperRangeBetaInt(start=2, end=100, alpha=1, beta=5),
        # Linear model
        "C": hp.HyperLogRangeFloat(start=0.00001, end=10, n=50),
        "alpha": hp.HyperLogRangeFloat(start=0.00001, end=10, n=50),
        # CV
        "analyzer": hp.HyperChoice(["word", "char", "char_wb"]),
        "penalty": ["l1", "l2"],
        "random_state": [123],  # So that every for every model with a random_state attribute, it will be passed and fix

        "drop_used_columns": [True],
        "drop_unused_columns": [True]
    }
    # This dictionnary is used to specify the default hyper-parameters that are used during the random search phase
    # They will be used if :
    # * the model has a paramters among that list
    # * the parameters is not specified within the class (within 'custom_hyper')

    default_default_hyper = {
        "random_state": 123,
        "drop_used_columns": True,
        "drop_unused_columns": True
    }
    # This dictionnary is used to specify the default hyper-parameters that are used during the default model phase
    # They will be used if :
    # * the model has a paramters among that list
    # * the default parameters is not specified within the class (withing 'default_parameters')

    depends_on = ()
