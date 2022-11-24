import os
from collections import OrderedDict
from copy import deepcopy
from typing import Tuple, List

import pandas as pd

from ._registry import MODEL_REGISTRY
from .hyper_parameters import HyperCrossProduct
from ._steps import get_needed_steps, filter_model_to_keep
from ..enums import ProblemType, StepCategory
from ..util.decorators import enforce_init
from ..util import get_columns_informations, check_column_information, get_columns_by_variable_type, guess_problem_type


@enforce_init
class AutoMlConfig(object):
    """ class to handle the AutoMlConfiguration, it will contain :

    * information about the type of variable
    * information about the type of problem
    * list of steps  to include in the auto-ml
    * list of models to include in the auto-ml
    """

    def __init__(self, X, y, groups=None, name=None):  # noqa
        self.name = name
        self.X = X
        self.y = y
        self.groups = groups

        # Variables to hold property values
        self._problem_type = None
        self._columns_informations = None
        self._columns_block = None
        self._needed_steps = None
        self._models_to_keep = None
        self._models_to_keep_block_search = None
        self._specific_hyper = None

        # Initialize default configuration values
        self.problem_type = None
        self.columns_informations = None
        self.needed_steps = None
        self.models_to_keep = None
        self.models_to_keep_block_search = None
        self.specific_hyper = {}
        self.columns_block = None

    def get_params(self):
        """ Returns parameters for serialization/deserialization purpose. """
        result = {}
        for attr in ("name", "problem_type", "columns_informations", "needed_steps", "models_to_keep"):
            result[attr] = getattr(self, attr)
        return result

    def set_params(self, **params):
        """ Sets parameters. """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @classmethod
    def from_params(cls, X, y, params, groups=None, name=None):  # noqa
        """ Instantiate class and restore params. """
        inst = cls(X=X, y=y, groups=groups, name=name)
        inst.set_params(**params)
        return inst

    # region Problem type

    @property
    def problem_type(self):
        return self._problem_type

    @problem_type.setter
    def problem_type(self, value):
        if value == self._problem_type:
            return

        if value is None:
            self._problem_type = None
            return

        if value not in ProblemType.alls:
            raise ValueError(f"'problem_type' should be among {ProblemType.alls}")

        # If it has move, recompute configuration that depends on problem type
        self.needed_steps = get_needed_steps(self.columns_informations, value)
        self.models_to_keep = filter_model_to_keep(value, block_search_only=False)
        self.guess_models_to_keep_for_block_search(value)
        self._problem_type = value

    @problem_type.deleter
    def problem_type(self):
        self._problem_type = None

    def guess_type_of_problem(self, X=None, y=None):  # noqa
        """ Guess problem type """
        if X is None:
            X = self.X  # noqa
        if y is None:
            y = self.y
        self.problem_type = guess_problem_type(X, y)
        return self.problem_type

    # endregion

    # region Columns information

    def guess_columns_informations(self, X=None):  # noqa
        """ set information about each column """
        if X is None:
            X = self.X  # noqa
        self.columns_informations = get_columns_informations(X)
        return pd.DataFrame(self.columns_informations).T

    @property
    def columns_informations(self):
        return self._columns_informations

    @columns_informations.setter
    def columns_informations(self, new_columns_informations):
        if new_columns_informations is None:
            self._columns_informations = None
            return

        # Warning, updates are not caught
        if not isinstance(new_columns_informations, dict):
            raise TypeError(f"'columns_informations' should be dict-like, instead I got '{new_columns_informations}'")

        cols = None if self.X is None else set(self.X.columns)

        for key, value in new_columns_informations.items():
            if cols is not None and key not in cols:
                raise ValueError(f"column {key} isn't present in dataset")
            check_column_information(key, value)

        self._columns_informations = deepcopy(new_columns_informations)

    @columns_informations.deleter
    def columns_informations(self):
        self._columns_informations = None

    def info(self):
        """ helper to quickly see info of variables """
        if self.X is not None:
            return pd.concat((pd.DataFrame(self.columns_informations).T,
                              self.X.dtypes.rename("type")), axis=1)
        else:
            return pd.DataFrame(self.columns_informations).T

    def update_columns_informations(self, new_columns_informations):
        if not isinstance(new_columns_informations, dict):
            raise TypeError(
                "columns_information should be dict-like, instead I got '%s'" % type(
                    new_columns_informations)
            )

        columns_informations_copy = deepcopy(self.columns_informations)

        for key, value in new_columns_informations.items():
            if key in columns_informations_copy:
                columns_informations_copy[key].update(value)

        self.columns_informations = columns_informations_copy  # Here checks will be done

        return self.columns_informations

    # endregion

    # region Columns blocks

    @property
    def columns_block(self):
        if self._columns_block is None:
            return get_columns_by_variable_type(self.columns_informations)
        return self._columns_block

    @columns_block.setter
    def columns_block(self, new_columns_block):
        if new_columns_block is None:
            self._columns_block = None
            return

        if not isinstance(new_columns_block, dict):
            raise TypeError(f"columns_block should be a dictionary, got {type(new_columns_block)}")

        cols = None if self.X is None else set(self.X.columns)

        for block_name, block_columns in new_columns_block.items():
            if not isinstance(block_name, str):
                raise TypeError(f"keys of columns_block should be string, got {type(block_name)}")

            if not isinstance(block_columns, (tuple, list)):
                raise TypeError(f"values of columns_block should be lists or tuples, got {type(block_columns)}")

            if cols is not None:
                for column in block_columns:
                    if column not in cols:
                        raise ValueError(f"column {column} isn't present in dataset")

            for column in block_columns:
                if column not in self.columns_informations:
                    raise ValueError(f"column {column} isn't present in dataset")

                if not self.columns_informations[column]["ToKeep"]:
                    raise ValueError(f"The column {column} shouldn't be keep and can't be included in a block")

        self._columns_block = deepcopy(new_columns_block)

    @columns_block.deleter
    def columns_block(self):
        self._columns_block = None

    # endregion

    # region Needed steps

    def guess_needed_steps(self):
        if self.columns_informations is None:
            raise ValueError("'columns_information' not set")

        if self.problem_type is None:
            raise ValueError("'problem_type' not set")

        self.needed_steps = get_needed_steps(self.columns_informations, self.problem_type)
        return self.needed_steps

    @property
    def needed_steps(self):
        return self._needed_steps

    @needed_steps.setter
    def needed_steps(self, new_needed_steps):
        if new_needed_steps is None:
            self._needed_steps = new_needed_steps
            return

        if not isinstance(new_needed_steps, (list, tuple)):
            raise TypeError(f"'needed_steps' should be a list or tuple, got {type(new_needed_steps)}")

        new_needed_steps = list(new_needed_steps)
        for step in new_needed_steps:
            if not isinstance(step, dict):
                raise TypeError(f"each step should be a dict, got {type(step)}")

            if "optional" not in step:
                raise ValueError(f"'optional' key not found be in step, got keys: {step.keys()}")

            if not isinstance(step["optional"], bool):
                raise TypeError(f"'optional' should be a boolean, got {type(step['optional'])}")

            if "step" not in step:
                raise ValueError(f"'step' key not found in step, got keys: {step.keys()}")

            if step["step"] not in StepCategory.alls:
                raise ValueError(f"Unknown step: {step['step']}, must be one of ({StepCategory.alls})")

        self._needed_steps = new_needed_steps

    @needed_steps.deleter
    def needed_steps(self):
        self._needed_steps = None

    # endregion

    # region Models to keep

    def guess_models_to_keep(self):
        if self.problem_type is None:
            raise ValueError("you need to set 'type_of_problem' first")
        self.models_to_keep = filter_model_to_keep(self.problem_type, block_search_only=False)
        return self.models_to_keep

    @property
    def models_to_keep(self) -> List[Tuple[str, str]]:
        return self._models_to_keep

    @models_to_keep.setter
    def models_to_keep(self, new_models_to_keep):
        if new_models_to_keep is None:
            self._models_to_keep = new_models_to_keep
            return

        if not isinstance(new_models_to_keep, (list, tuple)):
            raise TypeError(f"new_models_to_keep should be a list, got {type(new_models_to_keep)}")

        for n in new_models_to_keep:
            if not isinstance(n, (tuple, list)):
                raise TypeError(f"models should be tuple or list, got {type(n)}")
            if len(n) != 2:
                raise ValueError(f"models should be of size 2, got {len(n)}")
            if n[0] not in StepCategory.alls:
                raise ValueError(f"first item should be a step category ({StepCategory.alls}), got {n[0]}")
            if n not in MODEL_REGISTRY.all_registered:
                raise ValueError("model should have been registered")

        self._models_to_keep = new_models_to_keep

    @models_to_keep.deleter
    def models_to_keep(self):
        self._models_to_keep = None

    def filter_models(self, **kwargs):
        """ Filters the list of transformers/models that you want to test.
        You can also directly set the 'models_to_keep' attributes.

        Parameters
        ----------
        kwargs:
            A list of named parameters with name as the type of model and the value as the name of the model
            (can be either a str or a list of str)

        Example
        -------
        self.filter_models(Model = 'LGBMClassifier')
        self.filter_models(Model = ['LGBMClassifier','ExtraTreesClassifier'])
        """
        models_dict = OrderedDict()
        for k, v in self.models_to_keep:
            if k in models_dict:
                models_dict[k].append(v)
            else:
                models_dict[k] = [v]

        new_dict = OrderedDict()
        for k, vs in models_dict.items():
            if k in kwargs:
                args = kwargs[k]
                if isinstance(args, str):
                    args = [args]

                if not isinstance(args, (list, tuple)):
                    raise TypeError(f"Argument should be either list or tuple, got {type(args)}")

                for arg in args:
                    if arg not in vs:
                        raise ValueError(f"This model {arg} doesn't exist in original list")
                new_dict[k] = args
            else:
                new_dict[k] = vs

        new_models_to_keep = []
        for k, vs in new_dict.items():
            for v in vs:
                new_models_to_keep.append((k, v))
        self.models_to_keep = new_models_to_keep
        return self

    # endregion

    # region Models to keep for block search

    def guess_models_to_keep_for_block_search(self, problem_type=None):
        if problem_type is None:
            problem_type = self._problem_type

        if problem_type is None:
            raise ValueError(f"'problem_type' not set")

        if self.models_to_keep is None:
            raise ValueError(f"'models_to_keep' not set")

        models_to_keep_block_search = filter_model_to_keep(problem_type, block_search_only=True)
        self.models_to_keep_block_search = [m for m in
                                            models_to_keep_block_search if
                                            m in self.models_to_keep]
        return self.models_to_keep_block_search

    @property
    def models_to_keep_block_search(self):
        return self._models_to_keep_block_search

    @models_to_keep_block_search.setter
    def models_to_keep_block_search(self, new_models_to_keep_block_search):

        if new_models_to_keep_block_search is None:
            self._models_to_keep_block_search = new_models_to_keep_block_search
            return

        if not isinstance(new_models_to_keep_block_search, (list, tuple)):
            raise TypeError(f"new_models_to_keep should be a list or tuple, "
                            f"got {type(new_models_to_keep_block_search)}")

        for n in new_models_to_keep_block_search:
            if not isinstance(n, (tuple, list)):
                raise TypeError(f"model should be a tuple or list, got {n}")
            if len(n) != 2:
                raise ValueError(f"model should be of size 2, got {len(n)}")

            if n[0] not in StepCategory.alls:
                raise ValueError(f"first item should be a step category ({StepCategory.alls}), got {n[0]}")

            if n not in MODEL_REGISTRY.all_registered:
                raise ValueError(f"model should have been registered")

            if n not in self.models_to_keep:
                raise ValueError("model should be 'models_to_keep'")

        self._models_to_keep_block_search = new_models_to_keep_block_search

    # endregion

    # region Specific hyperparameter

    @property
    def specific_hyper(self):
        return self._specific_hyper

    @specific_hyper.setter
    def specific_hyper(self, new_specific_hyper):
        if new_specific_hyper is None or len(new_specific_hyper) == 0:
            self._specific_hyper = new_specific_hyper
            return

        if self.models_to_keep is None:
            raise ValueError("'models_to_keep' should be set")

        new_specific_hyper: dict = deepcopy(new_specific_hyper)
        if not isinstance(new_specific_hyper, dict):
            raise TypeError(f"specific_hyper should be a dict, got {type(new_specific_hyper)}")

        for key, value in new_specific_hyper.items():
            if key not in self.models_to_keep:
                raise ValueError(f"keys of specific_hyper should be within 'models_to_keep', unknown key {key}")

            if isinstance(value, dict):
                new_specific_hyper[key] = HyperCrossProduct(value)

            if not isinstance(new_specific_hyper[key], HyperCrossProduct):
                raise TypeError(f"values of specific_hyper should be dict or HyperCrossProduct, got {type(value)}")

        self._specific_hyper = new_specific_hyper

    @specific_hyper.deleter
    def specific_hyper(self):
        self._specific_hyper = None

    # endregion

    def check_base(self, X=None, y=None):  # noqa
        """ perform a few basic test on the base to make sur no mistake was made """
        if X is None:
            X = self.X  # noqa

        if y is None:
            y = self.y

        if X is not None:
            shape_X = getattr(X, "shape", None)  # noqa
        else:
            shape_X = None  # noqa

        if y is not None:
            shape_y = getattr(y, "shape", None)
        else:
            shape_y = None

        if shape_X is not None and y is not None:
            if shape_X[0] != shape_y[0]:
                raise ValueError(f"X and y don't have the same shape {shape_X[0]} vs {shape_y[0]}")

            if len(shape_y) > 1 and shape_y[1] > 1:
                raise ValueError("Multi-output isn't handled yet")

    def guess_everything(self, X=None, y=None):  # noqa
        self.guess_columns_informations(X)
        self.guess_type_of_problem(X, y)
        self.guess_needed_steps()
        self.guess_models_to_keep()
        self.guess_models_to_keep_for_block_search()
        self.check_base()
        return self

    def __repr__(self):
        res = [f"Problem type: {self.problem_type}"]
        return super(AutoMlConfig, self).__repr__() + os.linesep + os.linesep.join(res)
