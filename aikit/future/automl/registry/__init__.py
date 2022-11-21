# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:54:14 2018

@author: Lionel Massoulard

To register a new model you need to:
 * create a class with register decorator
 * the name isn't important (and the class will never be used). It is just a simplified way to register models.

The class should have:
 * klass: Model class
 * category: something among StepCategories, overall category of the transformer/model
 * type_of_variable: type of variable it is applied on
 * type_of_model: type of model
 * custom_hyper: CAN be used to specify hyperparameter
 * get_hyper_parameter: class method can also be overriden to specify behavior
 * any other class attribute will be saved in "information"
"""
from .._registry import MODEL_REGISTRY

MODEL_REGISTRY.reset()

from . import _models  # noqa
from . import _text  # noqa
from . import _transformers  # noqa
from . import _feature_selection  # noqa
