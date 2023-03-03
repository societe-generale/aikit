# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 08:38:10 2018

@author: Lionel Massoulard
"""
import abc
from inspect import signature

import networkx as nx

from .hyper_parameters import HyperCrossProduct
from ..enums import StepCategory
from ..util import CLASS_REGISTRY
from ..util.decorators import singleton
from ..util.graph import has_cycle, iter_graph


def get_init_parameters(klass):
    """ Get the parameter of the __init__ of a given klass """
    init = getattr(klass.__init__, "deprecated_original", klass.__init__)
    if init is object.__init__:
        return {}

    init_signature = signature(init)
    parameters = [p for p in init_signature.parameters.values() if
                  p.name != "self" and p.kind != p.VAR_KEYWORD]

    return {p.name: p.default for p in parameters}


@singleton
class ModelRegistry:
    """ Singleton class to store registered models. """

    def __init__(self):
        self.hyper_parameters = {}
        self.default_hyper_parameters = {}
        self.init_parameters = {}
        self.informations = {}
        self.is_allowed = {}
        self.all_registered = []
        self.step_dependencies = nx.DiGraph()
        self._drawing_order = {}
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

    def register_new_class(self,
                           category,
                           klass, hyper=None,
                           default_hyper=None,
                           is_allowed=None,
                           depends_on=None,
                           **kwargs):

        if not isinstance(klass, type):
            raise TypeError("klass should be a type definition")

        key = category, klass.__name__

        if key in self.all_registered:
            raise ValueError(f"{key} has already been registered")

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

        if klass.__name__ not in CLASS_REGISTRY._mapping:
            raise ValueError(f"You should also register that class: "
                             f"{klass.__name__} within the 'simple register' file")

        self.step_dependencies.add_node(category)
        if depends_on is not None:
            if not isinstance(depends_on, (list, tuple, set)):
                depends_on = (depends_on,)

            for depending_step in depends_on:
                if depending_step not in StepCategory.alls:
                    raise ValueError(f"{depending_step} is not a know step")

                self.step_dependencies.add_edge(depending_step, category)

                if has_cycle(self.step_dependencies):
                    raise ValueError(
                        f"Adding this dependency {depending_step} -> {category} creates a cycle")

                self._drawing_order = {step: n for n, step in enumerate(iter_graph(self.step_dependencies))}

        return self

    def __repr__(self):
        all_registered = sorted(self.all_registered, key=lambda x: (x[0], x[1]))
        if len(all_registered) == 0:
            return "no classes yet"

        categories, klasses = zip(*all_registered)

        max_length_step = max([len(s) for s in categories])
        result = ["Registered classes:", ""]
        last_cat = None
        for category, name in zip(categories, klasses):
            if last_cat is not None and last_cat != category:
                result.append("")
                last_cat = category
            elif last_cat is None:
                last_cat = category

            result.append(f"{category.ljust(max_length_step + 1)} : {name}")

        return "\n".join(result)


MODEL_REGISTRY = ModelRegistry()


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

    MODEL_REGISTRY.register_new_class(
        category=klass.category,
        klass=klass.klass,
        hyper=klass.get_hyper_parameter(),
        default_hyper=klass.get_default_hyper_parameter(),
        is_allowed=klass.is_allowed,
        depends_on=klass.depends_on,
        **other
    )

    return klass


class _AbstractModelRepresentation(metaclass=abc.ABCMeta):
    """ Abstract class to store a model/transformer, its hyper parameters, ... """

    klass = None
    name = None

    custom_hyper = {}
    default_hyper = {}
    default_parameters = {}

    hyper = None

    def __init__(self):
        pass

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

        return HyperCrossProduct(all_hyper)

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


def allow_conditional(model, models_by_steps):
    """ Is a given model allowed to be drawn based on what was already drawn in previous steps.

    Parameters
    ----------
    model : tuple (name, step)
        the model that we want to draw

    models_by_steps : dictionary of keys = steps, and values = Model
        the models already included

    Returns
    -------
    boolean
    """

    model_is_allowed_fun = MODEL_REGISTRY.is_allowed.get(model, None)

    if model_is_allowed_fun is None:
        return True

    return model_is_allowed_fun(models_by_steps)
