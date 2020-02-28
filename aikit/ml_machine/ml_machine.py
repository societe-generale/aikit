# -*- coding: utf-8 -*-
"""
Created on Fri May  4 13:32:38 2018

@author: Lionel Massoulard
"""

import logging

logger = logging.getLogger(__name__)

from copy import deepcopy
from collections import OrderedDict

import gc
import traceback


from datetime import datetime
import itertools
from functools import wraps


import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn.utils.validation import check_random_state

from aikit.tools.helper_functions import dico_keyvalue_filter, diff, intersect, unnest_tuple, _is_number, md5_hash
from aikit.tools.db_informations import (
    get_columns_informations,
    get_var_type_columns_dico,
    guess_type_of_problem,
    get_all_var_type,
)
import aikit.enums as en

import aikit.model_definition

from aikit.cross_validation import create_cv, cross_validation, score_from_params_clustering
from aikit.scorer import SCORERS

from aikit.ml_machine.steps_handling import (
    get_needed_steps,
    filter_model_to_keep,
    modify_var_type_none_to_default,
    modify_var_type_alldefault_to_none,
    create_var_type_from_steps,
)
from aikit.ml_machine.hyper_parameters import HyperMultipleChoice, HyperCrossProduct, HyperComposition

from aikit.ml_machine.ml_machine_registration import MODEL_REGISTER
from aikit.ml_machine.data_persister import SavingType
from aikit.ml_machine.jobs import AbstractJobRunner
from aikit.ml_machine.model_graph import convert_graph_to_code

from aikit.ml_machine import model_graph as mg


def froze_init(cls):
    """ decorator that prevent attribute that are not setted in the init """

    def __setattr__(self, key, value):
        if key[0] != "_":
            if self.__frozen:
                if key not in self.__allowed_attributes:
                    raise TypeError("this can't be setted : %s" % key)
            else:
                self.__allowed_attributes.add(key)

        object.__setattr__(self, key, value)

    def init_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self.__frozen = False
            self.__allowed_attributes = set()
            func(self, *args, **kwargs)
            self.__frozen = True

        return wrapper

    cls.__setattr__ = __setattr__
    cls.__init__ = init_decorator(cls.__init__)

    return cls


@froze_init
class AutoMlConfig(object):
    """ class to handle the AutoMlConfiguration, it will contain :

    * information about the type of variable
    * information about the type of problem
    * list of steps  to include in the auto-ml
    * list of models to include in the auto-ml
    """

    def __init__(self, dfX, y, groups=None, name=None):
        self.name = name

        self.dfX = dfX
        self.y = y
        self.groups = groups

        self.type_of_problem = None
        self.columns_informations = None
        self.needed_steps = None

        self.models_to_keep = None
        self.models_to_keep_block_search = None

        self.specific_hyper = {}

        # self._var_type_columns_dico = None
        # self._default_var_type = None
        self.columns_block = None

    ################################
    ### to save object into json ###
    ################################
    def get_params(self):
        result = {}
        for attr in ("name", "type_of_problem", "columns_informations", "needed_steps", "models_to_keep"):
            result[attr] = getattr(self, attr)

        return result

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @classmethod
    def from_params(cls, params):
        inst = cls()
        inst.set_params(**params)
        return inst

    ############################
    ### columns_informations ###
    ############################
    def guess_columns_informations(self, dfX=None):
        """ set information about each columns """

        if dfX is None:
            dfX = self.dfX

        self.columns_informations = get_columns_informations(dfX)
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
            raise TypeError(
                "'columns_informations' should be dict-like, instead I got '%s'" % type(new_columns_informations)
            )

        if self.dfX is None:
            cols = None
        else:
            cols = set(self.dfX.columns)

        for key, value in new_columns_informations.items():
            if cols is not None and key not in cols:
                raise ValueError("column %s isn't present in dataset" % key)

            if not isinstance(value, dict):
                raise TypeError("value of dictionnary should be dictionnary")

            # Should have 'HasMissing' #
            if "HasMissing" not in value:
                raise ValueError("'HasMissing' should be in the value of the dictionnary")

            if not isinstance(value["HasMissing"], bool):
                raise TypeError("'HasMissing' should be a boolean, instead it is '%s'" % type(value["HasMissing"]))

            # Should have 'TypeOfVariable'
            if "TypeOfVariable" not in value:
                raise ValueError("'TypeOfVariable' should be in the value of the dictionnary")

            if value["TypeOfVariable"] not in en.TypeOfVariables.alls:
                raise ValueError(
                    "Unknown'TypeOfVariable' : %s, it should be among (%s)"
                    % (value["TypeOfVariable"], str(en.TypeOfVariables.alls))
                )

            # Should have 'ToKeep'
            if "ToKeep" not in value:
                raise ValueError("'ToKeep' should be in the value of the dictionnary")

            if not isinstance(value["ToKeep"], bool):
                raise TypeError("'ToKeep' should be a boolean, instead it is '%s'" % type(value["ToKeep"]))

        self._columns_informations = deepcopy(new_columns_informations)
        # self._var_type_columns_dico = get_var_type_columns_dico(self._columns_informations)
        # self._default_var_type =  get_default_var_type(self._columns_informations)

    @columns_informations.deleter
    def columns_informations(self):
        self._columns_informations = None
        
    def infos(self):
        """ helper to quickly see info of variables """
        if self.dfX is not None:
            return pd.concat((pd.DataFrame(self.columns_informations).T,self.dfX.dtypes.rename("type")),axis=1)
        else:
            return pd.DataFrame(self.columns_informations).T

    # @property
    # def var_type_columns_dico(self):
    #    return self._var_type_columns_dico
    #
    # @property
    # def default_var_type(self):
    #    return self._default_var_type
    # Those 2 'attributes' can't be set, they are automatically re-compute when columns_informations changes

    def update_columns_informations(self, new_columns_informations):
        if not isinstance(new_columns_informations, dict):
            raise TypeError(
                "columns_information should be dict-like, instead I got '%s'" % type(new_columns_informations)
            )

        columns_informations_copy = deepcopy(self.columns_informations)

        for key, value in new_columns_informations.items():
            if key in columns_informations_copy:
                columns_informations_copy[key].update(value)

        self.columns_informations = columns_informations_copy  # Here checks will be done

        return self.columns_informations

    #####################
    ### Columns block ###
    #####################
    #    def guess_columns_block(self):
    #        """ guess the different blocks of columns """
    #        self.columns_block = get_var_type_columns_dico(self.columns_informations)

    # By default I'll set one block per type of variable
    # TODO : maybe detect 'BLOCKNAME_%..." pattern

    @property
    def columns_block(self):
        if self._columns_block is None:
            return get_var_type_columns_dico(self.columns_informations)

        return self._columns_block

    @columns_block.setter
    def columns_block(self, new_columns_block):

        if new_columns_block is None:
            self._columns_block = None
            return

        if not isinstance(new_columns_block, dict):
            raise TypeError("columns_block should be a dictionnary, not a %s" % type(new_columns_block))

        if self.dfX is None:
            cols = None
        else:
            cols = list(self.dfX.columns)

        for block_name, block_columns in new_columns_block.items():
            if not isinstance(block_name, str):
                raise TypeError("keys of columns_block should be strings, not %s" % type(block_name))

            if not isinstance(block_columns, (tuple, list)):
                raise TypeError("values of columns_block should be lists or tuples, not %s" % type(block_columns))

            if cols is not None:
                for column in block_columns:
                    if column not in cols:
                        raise ValueError("column %s isn't present in dataset" % column)

            for column in block_columns:
                if column not in self.columns_informations:
                    raise ValueError("column %s isn't present in dataset" % column)

                if not self.columns_informations[column]["ToKeep"]:
                    raise ValueError("The column %s shouldn't be keep and can't be included in a block" % column)

        # TODO : a priori je veux pas de columns dans 2 blocks, mais bon pourquoi pas... En theorie ca pose pas forcement de probleme

        self._columns_block = deepcopy(new_columns_block)

    @columns_block.deleter
    def columns_block(self):
        self._columns_block = None

    #######################
    ### type of problem ###
    #######################
    def guess_type_of_problem(self, dfX=None, y=None):
        """ set the type of problem """

        if dfX is None:
            dfX = self.dfX

        if y is None:
            y = self.y

        self.type_of_problem = guess_type_of_problem(dfX, y)
        return self.type_of_problem

    @property
    def type_of_problem(self):
        return self._type_of_problem

    @type_of_problem.setter
    def type_of_problem(self, new_type_of_problem):
        if new_type_of_problem is None:
            self._type_of_problem = None
            return

        if new_type_of_problem not in en.TypeOfProblem.alls:
            raise ValueError("'type_of_problem' should be among %s" % str(en.TypeOfProblem.alls))

        old_type_of_problem = self._type_of_problem

        if old_type_of_problem != new_type_of_problem:
            # If it has move, I need to re-change the other configurations.
            self.needed_steps = get_needed_steps(self.columns_informations, new_type_of_problem)
            self.models_to_keep = filter_model_to_keep(new_type_of_problem, block_search_only=False)
            self.guess_models_to_keep_block_search(new_type_of_problem)

        self._type_of_problem = new_type_of_problem

    @type_of_problem.deleter
    def type_of_problem(self):
        self._type_of_problem = None

    ######################
    ### Checks on base ###
    ######################
    def check_base(self, dfX=None, y=None):
        """ perform a few basic test on the base to make sur no mistake was made """
        if dfX is None:
            dfX = self.dfX

        if y is None:
            y = self.y

        #######################
        ### Check same size ###
        #######################
        if dfX is not None:
            shapeX = getattr(dfX, "shape", None)
        else:
            shapeX = None

        if y is not None:
            shapey = getattr(y, "shape", None)
        else:
            shapey = None

        if shapeX is not None and y is not None:
            if shapeX[0] != shapey[0]:
                raise ValueError("dfX and y don't have the same shape %d vs %d" % (shapeX[0], shapey[0]))

            if len(shapey) > 1 and shapey[1] > 1:
                raise ValueError("Multi-output isn't handled yet")

    ####################
    ### Needed steps ###
    ####################
    def guess_needed_steps(self):

        if self.columns_informations is None:
            raise ValueError("you need to set 'columns_information' first")

        if self.type_of_problem is None:
            raise ValueError("you need to set 'type_of_problem' first")

        self.needed_steps = get_needed_steps(self.columns_informations, self.type_of_problem)

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
            raise TypeError("'needed_steps' should be a list or tuple")

        new_needed_steps = list(new_needed_steps)
        for step in new_needed_steps:
            if not isinstance(step, dict):
                raise TypeError("each step should be a dict, instead it is %s" % type(step))

            if "optional" not in step:
                raise ValueError("'optional' should be in step")
            if not isinstance(step["optional"], bool):
                raise TypeError("'optional' should be a boolean")

            if "step" not in step:
                raise ValueError("'step' should be in step")
            if step["step"] not in en.StepCategories.alls:
                raise ValueError("Unknown step : %s" % step["step"])

        self._needed_steps = new_needed_steps

    @needed_steps.deleter
    def needed_steps(self):
        self._needed_steps = None

    ######################
    ### models to keep ###
    ######################
    def guess_models_to_keep(self):

        if self.type_of_problem is None:
            raise ValueError("you need to set 'type_of_problem' first")

        self.models_to_keep = filter_model_to_keep(self.type_of_problem, block_search_only=False)

        return self.models_to_keep

    @property
    def models_to_keep(self):
        return self._models_to_keep

    @models_to_keep.setter
    def models_to_keep(self, new_models_to_keep):

        if new_models_to_keep is None:
            self._models_to_keep = new_models_to_keep
            return

        if not isinstance(new_models_to_keep, (list, tuple)):
            raise TypeError("new_models_to_keep should be a list")

        for n in new_models_to_keep:
            if not isinstance(n, (tuple, list)):
                raise TypeError("all models should be tuple")
            if len(n) != 2:
                raise ValueError("all models should be of size 2")

            if n[0] not in en.StepCategories.alls:
                raise ValueError("first item should be among StepCategories")
            if n not in MODEL_REGISTER.all_registered:
                raise ValueError("each item should have been registred")

        self._models_to_keep = new_models_to_keep
        # TODO : ici je ne fais pas de test car je vais surement changer la structure de l'object

    @models_to_keep.deleter
    def models_to_keep(self):
        self._models_to_keep = None

    def filter_models(self, **kwargs):
        """ use that method to filter the list of transformers/models that you want to test

        You can also directly set the 'models_to_keep' attributes

        Parameters
        ----------
        name of params : type of model
        values of params : str or list with the names of the models

        Example
        -------
        self.filter_models(Model = 'LGBMClassifier')
        self.filter_models(Model = ['LGBMClassifier','ExtraTreesClassifier'])


        """
        dico_models = OrderedDict()
        for k, v in self.models_to_keep:
            if k in dico_models:
                dico_models[k].append(v)
            else:
                dico_models[k] = [v]

        new_dico = OrderedDict()
        for k, vs in dico_models.items():
            if k in kwargs:
                args = kwargs[k]
                if isinstance(args, str):
                    args = [args]

                if not isinstance(args, (list, tuple)):
                    raise TypeError("Argument should be either list or tuple, not %s" % type(args))

                for arg in args:
                    if arg not in vs:
                        raise ValueError("This model %s doesn't exist in original list" % arg)
                new_dico[k] = args
            else:
                new_dico[k] = vs

        new_models_to_keep = []
        for k, vs in new_dico.items():
            for v in vs:
                new_models_to_keep.append((k, v))
        self.models_to_keep = new_models_to_keep

        return self

    #######################################
    ### models to keep for block search ###
    #######################################
    def guess_models_to_keep_block_search(self, type_of_problem=None):

        if type_of_problem is None:
            type_of_problem = self._type_of_problem

        if type_of_problem is None:
            raise ValueError("you need to set 'type_of_problem' first")

        if self.models_to_keep is None:
            raise ValueError("models_to_keep need to be setted first")

        models_to_keep_block_search = filter_model_to_keep(type_of_problem, block_search_only=True)
        self.models_to_keep_block_search = [m for m in models_to_keep_block_search if m in self.models_to_keep]

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
            raise TypeError("new_models_to_keep should be a list")

        for n in new_models_to_keep_block_search:
            if not isinstance(n, (tuple, list)):
                raise TypeError("all models should be tuple")
            if len(n) != 2:
                raise ValueError("all models should be of size 2")

            if n[0] not in en.StepCategories.alls:
                raise ValueError("first item should be among StepCategories")

            if n not in MODEL_REGISTER.all_registered:
                raise ValueError("each item should have been registred")

            if n not in self.models_to_keep:
                raise ValueError("each item should be in 'models_to_keep'")

        self._models_to_keep_block_search = new_models_to_keep_block_search

    ################################
    ### Specific HyperParameters ###
    ################################

    @property
    def specific_hyper(self):
        return self._specific_hyper

    @specific_hyper.setter
    def specific_hyper(self, new_specific_hyper):

        if new_specific_hyper is None or len(new_specific_hyper) == 0:
            self._specific_hyper = new_specific_hyper
            return

        if self.models_to_keep is None:
            raise ValueError("Please specify models_to_keep first")

        new_specific_hyper = deepcopy(new_specific_hyper)
        if isinstance(self, dict):
            raise TypeError("specific_hyper should be a dict, instead I got %s" % type(new_specific_hyper))

        for key, value in new_specific_hyper.items():

            if key not in self.models_to_keep:
                raise ValueError("keys of specific_hyper should be within 'models_to_keep' : unknown %s" % str(key))

            if isinstance(value, dict):
                new_specific_hyper[key] = HyperCrossProduct(value)

            if not isinstance(new_specific_hyper[key], HyperCrossProduct):
                raise TypeError(
                    "values of specific_hyper should be dict or HyperCrossProduct, instead I got %s" % type(value)
                )

        self._specific_hyper = new_specific_hyper

    @specific_hyper.deleter
    def specific_hyper(self):
        self._specific_hyper = None

    def guess_everything(self, dfX=None, y=None):
        self.guess_columns_informations(dfX)
        # self.guess_columns_block()
        self.guess_type_of_problem(dfX, y)
        self.guess_needed_steps()
        self.guess_models_to_keep()
        self.guess_models_to_keep_block_search()

        self.check_base()
        return self

    ### Repr ###
    def __repr__(self):

        res = ["type of problem : %s" % self.type_of_problem]

        return super(AutoMlConfig, self).__repr__() + "\n" + "\n".join(res)


# In[]
def _create_all_combinations(all_blocks_to_use, max_number_of_blocks_to_test, max_number_of_blocks_to_remove):
    """ create all the combinations of 'all_blocks_to_use' that have either
    * at most 'max_number_of_blocks_to_test' elements    OR
    * at most 'max_number_of_blocks_to_remove' elements removed
    
    Parameters
    ----------
    max_number_of_blocks_to_test : int, default=1
        will include ALL the models with at most 'max_number_of_blocks_to_test' (all combinations)
        
    max_number_of_blocks_to_remove : int, default=1
        will include ALL the models with at most 'max_number_of_blocks_to_remove' blocks REMOVED (al combinations)
    """

    if max_number_of_blocks_to_test < 1:
        raise ValueError("'max_number_of_blocks_to_test' should be >= 1")

    if max_number_of_blocks_to_remove < 1:
        raise ValueError("'max_number_of_blocks_to_remove' should be >= 1")

    if len(set(all_blocks_to_use)) != len(all_blocks_to_use):
        raise ValueError("'all_blocks_to_use' shouldn't contain duplicate")

    if max_number_of_blocks_to_test >= len(all_blocks_to_use) - 1:
        max_number_of_blocks_to_test = len(all_blocks_to_use) - 1

    if max_number_of_blocks_to_remove >= len(all_blocks_to_use) - 1:
        max_number_of_blocks_to_remove = len(all_blocks_to_use) - 1

    # Create full list of blocks_to_use to be tried
    set_blocks_to_use = set()
    for r in range(1, max_number_of_blocks_to_test + 1):
        for blocks_to_use in itertools.combinations(all_blocks_to_use, r=r):
            if blocks_to_use not in set_blocks_to_use:
                set_blocks_to_use.add(blocks_to_use)

    for r in range(1, max_number_of_blocks_to_remove + 1):
        for blocks_to_remove in itertools.combinations(all_blocks_to_use, r=r):
            blocks_to_use = tuple([b for b in all_blocks_to_use if b not in blocks_to_remove])
            if blocks_to_use not in set_blocks_to_use:
                set_blocks_to_use.add(blocks_to_use)

    return list(set_blocks_to_use)


def random_list_generator(elements, probas=None, random_state=None):
    """ helper to create a generator in a random order
    
    Parameters
    ----------
    elements : list or iterable
        the list of elements
        
    probas : None or list of proba
        if not None, the 'un-normalize' proba to draw each elements
        
    random_state : None, int, or generator
        the random generator
        
    Yield
    -----
    element in random order
    """

    elements = list(elements)

    random_state = check_random_state(random_state)

    if probas is not None:
        probas = list(probas)
        if len(elements) != len(probas):
            raise ValueError("'elements' and 'probas' should have the same length")

        if len(elements) > 0 and np.array(probas).min() <= 0:
            raise ValueError("'probas' should be >0")

    all_indexes = list(range(len(elements)))

    remaining_indexes = all_indexes

    while len(remaining_indexes) > 0:
        if probas is None:
            ind = random_state.choice(remaining_indexes)
        else:
            p = np.array([probas[i] for i in remaining_indexes])
            p = p / p.sum()
            ind = random_state.choice(remaining_indexes, p=p)

        remaining_indexes = [r for r in remaining_indexes if r != ind]

        yield elements[ind]


class RandomModelGenerator(object):
    """ class to generate random model """

    def __init__(self, auto_ml_config, allow_block_selection=True, block_probas=0.9, random_state=None):

        self.auto_ml_config = auto_ml_config

        self.random_probas = None

        self.allow_block_selection = allow_block_selection
        self.block_probas = block_probas

        self.hyper_parameters = {}
        self._hyper_block = None

        self.random_state = random_state

        self.prepare_hyper_parameters()

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, new_random_state):
        self._random_state = check_random_state(new_random_state)
        for k, hyper in self.hyper_parameters.items():
            hyper.random_state = self._random_state

        if self._hyper_block is not None:
            self._hyper_block.random_state = self._random_state

    def prepare_hyper_parameters(self):
        """ create the custum hyper-parameters """
        self.hyper_parameters = {}

        for model in self.auto_ml_config.models_to_keep:

            hyper = MODEL_REGISTER.hyper_parameters[model]
            hyper.random_state = self.random_state
            assert isinstance(hyper, (HyperCrossProduct, HyperComposition))

            other_hyper = self.auto_ml_config.specific_hyper.get(model, None)
            if other_hyper is None:
                self.hyper_parameters[model] = hyper
            else:
                self.hyper_parameters[model] = hyper + other_hyper  # random_state here ?

        return self

    def iterator_default_models(self):
        """ iterator that generate the list of default models to test """

        all_choices_by_steps = []
        for step in self.auto_ml_config.needed_steps:

            if step["optional"]:
                all_choices = [(None, None)]
            else:
                all_choices = [n for n in self.auto_ml_config.models_to_keep if n[0] == step["step"]]

                # Maybe we'd like to remove some choices here for some steps

            all_choices_by_steps.append([(step["step"], c) for c in all_choices])

        all_models_steps = [OrderedDict(m) for m in itertools.product(*all_choices_by_steps)]

        for models_by_steps in all_models_steps:

            # Blocks
            blocks_to_use = tuple(self.auto_ml_config.columns_block.keys())  # keep all blocks

            # Hyper
            hyper_parameters_by_step = {}
            for step_name, model_name in models_by_steps.items():
                if model_name[0] is not None:
                    default_parameters = MODEL_REGISTER.default_hyper_parameters.get(model_name, {})
#                    default_parameters = MODEL_REGISTER.informations.get(model_name, {}).get("default_parameters", {})
                    # If default_parameters present in register use it, otherwise use {} (and so will go back to default parameter of the model)
                    hyper_parameters_by_step[(step_name, model_name)] = default_parameters

            # Call the 'draw_random_graph' method but with pre-setted params
            simplified_Graph, all_models_params, blocks_to_use = self.draw_random_graph(
                blocks_to_use=blocks_to_use,
                models_by_steps=models_by_steps,
                hyper_parameters_by_step=hyper_parameters_by_step,
            )

            yield simplified_Graph, all_models_params, blocks_to_use

    def iterate_block_search(self, max_number_of_blocks_to_test=1, max_number_of_blocks_to_remove=1, random_order=True):
        """ iterator that generate the list of models to test when we are searching for blocks
        The iteration order can be random, in that case the order is drawn by a law depending on the size of the blocks to use
        
        Parameters
        ----------
        max_number_of_blocks_to_test : int, default=1
            will include ALL the models with at most 'max_number_of_blocks_to_test' (all combinations)
            
        max_number_of_blocks_to_remove : int, default=1
            will include ALL the models with at most 'max_number_of_blocks_to_remove' blocks REMOVED (al combinations)
            
        random_order: boolean, default=True
            if True will iterate in a random order

        Yields
        ------
        simplified_Graph, all_models_params, blocks_to_use
        """
        if random_order:
            block_search_models = list(
                self._iterate_block_search_models(
                    max_number_of_blocks_to_test=max_number_of_blocks_to_test,
                    max_number_of_blocks_to_remove=max_number_of_blocks_to_remove,
                )
            )

            N = len(self.auto_ml_config.columns_block.keys())

            probas = []
            for _, _, blocks_to_use in block_search_models:
                p = 1 / min(len(blocks_to_use), N - len(blocks_to_use))
                probas.append(p)

            yield from random_list_generator(block_search_models, probas=probas, random_state=self.random_state)

        else:
            yield from self._iterate_block_search_models(
                max_number_of_blocks_to_test=max_number_of_blocks_to_test,
                max_number_of_blocks_to_remove=max_number_of_blocks_to_remove,
            )

    def _iterate_block_search_models(self, max_number_of_blocks_to_test=1, max_number_of_blocks_to_remove=1):
        """ iterator that generate the list of models to test when we are searching for blocks
        
        Parameters
        ----------
        max_number_of_blocks_to_test : int, default=1
            will include ALL the models with at most 'max_number_of_blocks_to_test' (all combinations)
            
        max_number_of_blocks_to_remove : int, default=1
            will include ALL the models with at most 'max_number_of_blocks_to_remove' blocks REMOVED (al combinations)
            
        Yields
        ------
        simplified_Graph, all_models_params, blocks_to_use
        """

        all_blocks_to_use = tuple(self.auto_ml_config.columns_block.keys())  # keep all blocks
        if len(all_blocks_to_use) <= 1:
            return  # No models

        list_blocks_to_use = _create_all_combinations(
            all_blocks_to_use,
            max_number_of_blocks_to_test=max_number_of_blocks_to_test,
            max_number_of_blocks_to_remove=max_number_of_blocks_to_remove,
        )

        for blocks_to_use in list_blocks_to_use:

            needed_steps_filtered, columns_informations_filtered, all_columns_keep = self._filter_based_on_blocks(
                blocks_to_use
            )

            all_choices_by_steps = []
            for step in needed_steps_filtered:

                if step["optional"]:
                    all_choices = [(None, None)]
                else:
                    all_choices = [n for n in self.auto_ml_config.models_to_keep_block_search if n[0] == step["step"]]

                    # Peut etre qu'on veut enlever des choix ici pour certain steps

                all_choices_by_steps.append([(step["step"], c) for c in all_choices])

            all_models_steps = [OrderedDict(m) for m in itertools.product(*all_choices_by_steps)]

            for models_by_steps in all_models_steps:

                hyper_parameters_by_step = {}
                for step_name, model_name in models_by_steps.items():
                    if model_name[0] is not None:
                        default_parameters = MODEL_REGISTER.informations.get(model_name, {}).get(
                            "default_parameters", {}
                        )
                        # If default_parameters present in register use it, otherwise use {} (and so will go back to default parameter of the model)
                        hyper_parameters_by_step[(step_name, model_name)] = default_parameters

                simplified_Graph, all_models_params, blocks_to_use = self.draw_random_graph(
                    blocks_to_use=blocks_to_use,
                    models_by_steps=models_by_steps,
                    hyper_parameters_by_step=hyper_parameters_by_step,
                )

                yield simplified_Graph, all_models_params, blocks_to_use

    ###########################
    ### Everything Together ###
    ###########################

    def _filter_based_on_blocks(self, blocks_to_use):
        """ helper function to refilter what is needed when knowing which blocks are used """
        ######################
        ### Filter columns ###
        ######################
        temp_kept_columns = []
        for b in blocks_to_use:
            temp_kept_columns += self.auto_ml_config.columns_block[b]

        columns_informations_filtered = dico_keyvalue_filter(
            self.auto_ml_config.columns_informations, lambda k, v: (k in temp_kept_columns and v["ToKeep"])
        )

        all_columns_keep = sorted(columns_informations_filtered.keys()) == sorted(
            self.auto_ml_config.columns_informations
        )

        #        else:
        #            columns_informations_filtered = self.auto_ml_config.columns_informations
        #

        ####################
        ### Filter steps ###
        ####################
        needed_steps_filtered_temp = get_needed_steps(
            columns_informations_filtered, self.auto_ml_config.type_of_problem
        )
        steps_in_config = [
            s2["step"] for s2 in self.auto_ml_config.needed_steps
        ]  # If user remove some step... I need to retake the intersection
        needed_steps_filtered = [s for s in needed_steps_filtered_temp if s["step"] in steps_in_config]

        return needed_steps_filtered, columns_informations_filtered, all_columns_keep

    def draw_random_graph(self, blocks_to_use=None, models_by_steps=None, hyper_parameters_by_step=None):
        """ draw a random model graph and its parameters """

        ###################
        ### Draw blocks ###
        ###################
        if blocks_to_use is None:
            if self.allow_block_selection:
                blocks_to_use = self._draw_blocks()
            else:
                blocks_to_use = tuple(sorted(self.auto_ml_config.columns_block.keys()))

        needed_steps_filtered, columns_informations_filtered, all_columns_keep = self._filter_based_on_blocks(
            blocks_to_use
        )

        ###################
        ### Draw models ###
        ###################
        if models_by_steps is None:
            models_by_steps = self._draw_random_model_by_step(
                needed_steps=needed_steps_filtered,
                models_to_keep=self.auto_ml_config.models_to_keep,
                random_probas=self.random_probas,
            )

        #############################
        ### Draw Hyper-parameters ###
        #############################
        if hyper_parameters_by_step is None:
            hyper_parameters_by_step = self._draw_hyperparameters(
                models_by_steps
            )  # , specific_hyper = self.auto_ml_config.specific_hyper)

        #        ######################################
        #        ### Re-create var-type columns dico ##
        #        ######################################
        #        if not no_columns_filtered:
        #            var_type_columns_dico_filtered = OrderedDict([(k,intersect(v, kept_columns)) for k,v in self.auto_ml_config.var_type_columns_dico.items()])
        #            var_type_columns_dico_filtered = OrderedDict([(k,v) for k,v in var_type_columns_dico_filtered.items() if len(v) > 0])
        #
        #
        #        else:
        #            var_type_columns_dico_filtered = self.auto_ml_config.var_type_columns_dico

        ########################
        ### VariableType Cat ###
        ########################
        remaining_var_type = get_all_var_type(columns_informations_filtered)  # All var type still present

        var_type_by_steps = create_var_type_from_steps(models_by_steps)  # Retrieve 'saved' var type from register
        var_type_by_steps = modify_var_type_none_to_default(
            var_type_by_steps, remaining_var_type
        )  # Modify None to remaining_var_type + intersect other with remaining var_type

        if all_columns_keep:
            var_type_by_steps = modify_var_type_alldefault_to_none(var_type_by_steps, remaining_var_type)

        var_type_columns_dico = get_var_type_columns_dico(
            columns_informations_filtered
        )  # 'columns for each type of variable

        ###############################
        ### Now assemble everything ###
        ###############################

        Graph, simplified_Graph, all_models_params = self._create_graph(
            var_type_by_steps=var_type_by_steps,
            hyper_parameters_by_step=hyper_parameters_by_step,
            var_type_columns_dico=var_type_columns_dico,
        )

        return simplified_Graph, all_models_params, blocks_to_use

    ######################
    ### Graph Creation ###
    ######################

    def _create_graph(self, var_type_by_steps, hyper_parameters_by_step, var_type_columns_dico):
        """ create the complete graph """

        ### Complete Graph (None transformer INCLUDE) ###
        Graph, new_steps = mg.create_graphical_representation(var_type_by_steps)
        # mg.graphviz_modelgraph(Graph)

        if len(new_steps) > 0:
            hyper_parameters_by_step = deepcopy(hyper_parameters_by_step)
            for step, vtype in new_steps.items():
                var_type_by_steps[step] = vtype
                hyper_parameters_by_step[step] = {}

        ### Simplified Graph (None transformers excluded ###
        simplified_Graph = mg.simplify_none_node(Graph)
        # mg.graphviz_modelgraph(simplified_Graph)

        ###  Add the selectors
        simplified_Graph, all_models_params = mg.add_columns_selector(
            Graph=simplified_Graph,
            var_type_node_dico=var_type_by_steps,
            var_type_columns_dico=var_type_columns_dico,
            all_models_params=hyper_parameters_by_step,
        )

        # mg.graphviz_modelgraph(simplified_Graph)
        return Graph, simplified_Graph, all_models_params

    #    def draw_random_graph_OLD(self):
    #        """ draw a random model graph and its parameters """
    #        models_by_steps = self._draw_models()
    #
    #        if self.allow_block_selection:
    #            blocks_to_use   = self._draw_blocks()
    #        else:
    #            blocks_to_use = None
    #
    #        hyper_parameters_by_step = self._draw_hyperparameters(models_by_steps)
    #
    #        Graph, simplified_Graph, all_models_params, blocks_to_use = self._create_graph(models_by_steps,hyper_parameters_by_step, blocks_to_use)
    #
    #        return simplified_Graph, all_models_params , blocks_to_use

    #############
    ### Steps ###
    #############
    def _draw_random_model_by_step(self, needed_steps, models_to_keep, log_unif=True, random_probas=None):
        """
        for each step draw a random model among the possible models,

        Parameters
        ----------
        * needed_steps : list of dictionnary representing each step, each one with a 'step' key and an 'optional' key

        * models_to_keep : list of models that we want to draw within

        * log_unif : bool, default = True
            if True (and random_probas is not None), will draw a model with probability proportionnal to 'log(1 + hyperparameter.size)'

        * random_probas : dictionnary of proba array for each step or None
            if None will use uniform (or log_uniform) otherwise will draw according to that probability


        """
        # TODO : specify a random_state + save random state ?
        # TODO : allow conditionnal probas to draw steps  => create a Graphical Proba Model to handle implication
        models_by_steps = OrderedDict()
        for step in needed_steps:

            all_choices = [n for n in models_to_keep if n[0] == step["step"]]
            if step["optional"]:
                all_choices.append((None, None))
                # TODO : put that into a Constant

            if len(all_choices) == 0:
                logger.info("I'll skip this step : %s" % step)
                continue

            if random_probas is None:
                p = None
                if log_unif:
                    all_hypers = [MODEL_REGISTER.hyper_parameters.get(model_name, None) for model_name in all_choices]
                    all_sizes = np.array([10 if h is None else h.size for h in all_hypers])
                    # TODO : les 'nothing' de taille 1 c'est pas bon, il vont jamais etre tire

                    # Attention : dans le cas ou il a des hyper-parametre custum la taille n'est plus bonne !
                    p = np.log1p(all_sizes)
                    p /= p.sum()

                # TODO : ici il faut faire un choix plutot en 'log uniform' sur le nombre de combinaison possibles
                # Pb, on connait pas vraiment le nombre ?
                # TODO : proxy avec une methode 'nb' sur les hyper-parameters ?

            else:
                default_p = 1 / len(all_choices)
                p = np.array([random_probas.get((step["step"], c), default_p) for c in all_choices])
                p /= np.sum(p)

            ii = np.arange(len(all_choices))
            chosen_class = all_choices[ii[self.random_state.choice(ii, size=1, p=p)[0]]]

            models_by_steps[step["step"]] = chosen_class
            # all_current_steps[chosen_class] = var_type

        return models_by_steps

    def _draw_hyperparameters(self, all_steps):  # ,specific_hyper = None):
        """ draw the random hyper-parameters """
        # TODO : specify a random state

        all_hyper = {}
        for step_name, model_name in all_steps.items():
            if model_name[0] is not None:
                hyper = self.hyper_parameters.get(
                    model_name, None
                )  # draw from self.hyper_parameters (which might have been updated by the 'specific_hyper' )
                all_hyper[(step_name, model_name)] = hyper

        def custom_draw(hyper):
            if hyper:
                return hyper.get_rand()
            else:
                return {}

        all_models_params = {n: custom_draw(hyper=all_hyper[n]) for n in all_hyper.keys()}

        return all_models_params

    #    def _draw_hyperparameters_OLD(self, models_by_steps):
    #        """ draw the hyperparameters for each model in each step """
    #        ### model by model => draw a random set of hyper-parameters
    #        hyper_parameters_by_step = self._draw_random_hyperparameters(list(models_by_steps.items()))
    #        return hyper_parameters_by_step

    def _draw_blocks(self):
        """ draw the block of model to keep """
        if len(self.auto_ml_config.columns_block.keys()) == 1:
            return tuple(sorted(self.auto_ml_config.columns_block.keys()))

        if self._hyper_block is None:
            self._hyper_block = HyperMultipleChoice(
                tuple(sorted(self.auto_ml_config.columns_block.keys())),
                min_number=1,
                proba_choice=self.block_probas,
                random_state=self.random_state,
            )

        return self._hyper_block.get_rand()


# In[]


# from sklearn.metrics.scorer import SCORERS
# Remark : on peut peut etre faire une copie local de ce dictionnaire pour rajouter nos propres objects


@froze_init
class JobConfig(object):
    """ small helper class to store a job configuration

    Attributes
    -----------

    * cv : CrossValidation to use

    * scoring : list of scorer

    * main_scorer : main scorer (used for the baseline)

    * score_base_line : base line of the main_scorer

    * allow_approx_cv : if True will do an approximate cv (faster)

    """

    def __init__(self):

        self.cv = None
        self.scoring = None

        self.score_base_line = None

        self.start_with_default = True  # if True, will start with default models
        self.do_blocks_search = True  # if True, will add in the queue model aiming at searching which block add values
        self.allow_approx_cv = False  # if True, will do 'approximate cv'

        self.main_scorer = None
        self.guiding_scorer = None

        self.additional_scoring_function = None

    ########################
    ### Cross Validation ###
    ########################
    def guess_cv(self, auto_ml_config, n_splits=10):
        if auto_ml_config.type_of_problem == en.TypeOfProblem.CLASSIFICATION:
            cv = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

        elif auto_ml_config.type_of_problem == en.TypeOfProblem.CLUSTERING:
            cv = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=123)

        else:
            cv = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=123)

        self.cv = cv
        return cv

    @property
    def cv(self):
        return self._cv

    @cv.setter
    def cv(self, new_cv):
        if new_cv is None:
            self._cv = new_cv
            return

        if new_cv is not None and not isinstance(new_cv, int):
            if not hasattr(new_cv, "split") or isinstance(new_cv, str):
                raise ValueError(
                    "Expected cv as an integer, cross-validation "
                    "object (from sklearn.model_selection) "
                    "or an iterable. Got %s." % new_cv
                )
        self._cv = new_cv

    @cv.deleter
    def cv(self):
        self._cv = None

    # TODO : on pourrait utiliser 2 CVs ... pour comparer 2 choses a la fois
    # Ex : 'Adverse CV' + 'Regular CV' ?
    # Ou tout bettement plusieurs CV pour averager a la fin les perfs

    ###############
    ### Metrics ###
    ###############
    def guess_scoring(self, auto_ml_config):
        if auto_ml_config.type_of_problem == en.TypeOfProblem.CLASSIFICATION:
            self.scoring = ["accuracy", "log_loss_patched", "avg_roc_auc", "f1_macro"]

        elif auto_ml_config.type_of_problem == en.TypeOfProblem.CLUSTERING:
            self.scoring = ["silhouette", "calinski_harabaz", "davies_bouldin"]

        else:
            self.scoring = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]

        return self.scoring

    @property
    def scoring(self):
        return self._scoring

    @scoring.setter
    def scoring(self, new_scoring):

        if new_scoring is None:
            self._scoring = new_scoring
            return

        if not isinstance(new_scoring, (list, tuple)):
            new_scoring = [new_scoring]

        for scoring in new_scoring:
            if isinstance(scoring, str):
                if scoring not in SCORERS:
                    raise ValueError("I don't know that scorer : %s" % scoring)
            else:
                raise NotImplementedError("for now I can only use pre-defined scorer, entered as string")
                # Rmk : on peut enlever cette restriction
                # Il faut verifier que c'est un scorer (mais bon le cross_val s'en chargera)

        self._scoring = new_scoring

    @scoring.deleter
    def scoring(self):
        self._scoring = None

    #######################
    ### Score Base Line ###
    #######################
    @property
    def score_base_line(self):
        return self._score_base_line

    @score_base_line.setter
    def score_base_line(self, new_score_base_line):
        if new_score_base_line is None:
            self._score_base_line = new_score_base_line
            return

        if pd.isnull(new_score_base_line):
            self._base_line = None

        else:
            if not _is_number(new_score_base_line):
                raise TypeError("base_line should a be a number, instead I got %s" % type(new_score_base_line))

            self._score_base_line = new_score_base_line

    @score_base_line.deleter
    def score_base_line(self):
        self._score_base_line = None

    ###################
    ### Main Scorer ###
    ###################
    @property
    def main_scorer(self):
        if self._main_scorer is None and self._scoring is not None:
            return self._scoring[0]
        else:
            return self._main_scorer

    @main_scorer.setter
    def main_scorer(self, new_main_scorer):

        if new_main_scorer is None:
            self._main_scorer = new_main_scorer
            return

        if new_main_scorer not in self._scoring:
            raise ValueError("main_scorer should be among 'scoring', %s" % new_main_scorer)

        self._main_scorer = new_main_scorer

        self._scoring = [self._main_scorer] + [s for s in self._scoring if s != self._main_scorer]
        # Put main_scorer first, rmk : cross_val_score_V2 will use FIRST scorer to determine when to stop

    @main_scorer.deleter
    def main_scorer(self):
        self._main_scorer = None

    ####################################
    ###  Addtionnal Scoring Function ###
    ####################################

    @property
    def additional_scoring_function(self):
        return self._additional_scoring_function

    @additional_scoring_function.setter
    def additional_scoring_function(self, new_additional_scoring_function):

        if new_additional_scoring_function is None:
            self._additional_scoring_function = new_additional_scoring_function
            return

        if not callable(new_additional_scoring_function):
            raise TypeError("'additional_scoring_function' should be callable")

        self._additional_scoring_function = new_additional_scoring_function

    @additional_scoring_function.deleter
    def additional_scoring_function(self):
        self._additional_scoring_function = None

    ############
    ### Repr ###
    ############
    def __repr__(self):
        res = [
            "cv              : %s" % self.cv.__repr__(),
            "scoring         : %s" % str(self.scoring),
            "score_base_line : %s" % str(self.score_base_line),
            "main_scorer     : %s" % str(self.main_scorer),
        ]

        return super(JobConfig, self).__repr__() + "\n" + "\n".join(res)


# In[]


class JobManagerQueue(object):
    """ Queue like object that generate the model that the ml machine
    This is the queue of the controller
    """

    def __init__(self, auto_ml_config, job_config, auto_ml_guider, data_persister, random_state=None):

        self.auto_ml_config = auto_ml_config
        self.job_config = job_config

        self.auto_ml_guider = auto_ml_guider

        self.data_persister = data_persister

        self._last_metric_threshold = None
        self.random_model_generator = None

        self.random_state = random_state

        self.prepare()

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, new_random_state):
        self._random_state = check_random_state(new_random_state)
        if self.random_model_generator is not None:
            self.random_model_generator.random_state = self._random_state

    def prepare(self):
        self.random_model_generator = RandomModelGenerator(
            auto_ml_config=self.auto_ml_config, random_state=self.random_state
        )

        if self.job_config.start_with_default:
            self._default_iterator = self.random_model_generator.iterator_default_models()
            self._default_iterator_empty = False

        else:
            self._default_iterator = None  # Will never be used
            self._default_iterator_empty = True

        if self.job_config.do_blocks_search:
            self._block_search_iterator = self.random_model_generator.iterate_block_search(random_order=True)
            self._block_search_iterator_empty = False
        else:
            self._block_search_iterator = None
            self._block_search_iterator_empty = True

    def remove(self):
        """ main method that will generate a new job to be given to ml machine """
        # nb_models_done = self.auto_ml_guider.get_nb_models_done()
        # proba_exploration = #####

        job_type = None

        #########################################################
        ### Try to remove something from the default iterator ###
        #########################################################
        if not self._default_iterator_empty:

            # Here : maybe look if something tells if this phase is not already done
            if self.data_persister.exists("phase", path="infos", write_type=SavingType.json):
                phase = self.data_persister.read("phase", path="infos", write_type=SavingType.json)
            else:
                phase = "default"
                self.data_persister.write(phase, "phase", path="infos", write_type=SavingType.json)

            if phase == "default":
                try:
                    iter_next = next(self._default_iterator)
                except StopIteration:
                    iter_next = None

                if iter_next is None:
                    logger.info("default_iterator is empty...")
                    job_type = None
                    self._default_iterator_empty = True

                    phase = "random"
                    self.data_persister.write(phase, "phase", path="infos", write_type=SavingType.json)

                else:
                    job_type = "default"

            else:
                job_type = None
                iter_next = None

        #################################################################
        ###  Try to remove something from the block search iterator   ###
        #################################################################
        if job_type is None and not self._block_search_iterator_empty:

            p_block_search = 0.2
            if self.random_state.rand(1)[0] <= p_block_search:

                try:
                    iter_next = next(self._block_search_iterator)
                except StopIteration:
                    iter_next = None

                if iter_next is None:
                    logger.info("block_search iterator is empty...")
                    self._block_search_iterator_empty = True
                    job_type = None
                else:
                    job_type = "block_search"

        if job_type is None:
            # It means this is not a default model ...
            if self.auto_ml_guider is None:
                # no guider => I'll use exploration
                job_type = "exploration"
            else:
                p_exploration = self.auto_ml_guider.find_exploration_proba()

                logger.debug("exploration proba : %2.2f%%" % (100 * p_exploration))

                if self.random_state.rand(1)[0] <= p_exploration:
                    job_type = "exploration"
                else:
                    job_type = "guided"

        if job_type == "exploration":

            logger.info("I'll create a random model...")

            ###############################################
            ### Exploration job : Draw one random model ###
            ###############################################
            Graph, all_models_params, blocks_to_use = self.random_model_generator.draw_random_graph()

            temp = convert_graph_to_code(Graph, all_models_params, also_returns_mapping=True)
            model_json_code = temp["json_code"]
            name_mapping = temp["name_mapping"]

            job_id = md5_hash(model_json_code)  # Or increment an id ? avec un SharedInteger

            json_param = {
                "Graph": {"nodes": list(Graph.nodes), "edges": list(Graph.edges)},
                "all_models_params": all_models_params,
                "blocks_to_use": blocks_to_use,
                "job_id": job_id,
            }

        elif job_type == "guided":

            logger.info("I'll create a guided model...")
            ################################################################
            ### Guided job : draw several models and guess which is best ###
            ################################################################
            all_params1 = []
            all_params2 = []
            all_names_mapping = []

            for nb in range(100):
                # I'll draw 100 random models
                Graph, all_models_params, blocks_to_use = self.random_model_generator.draw_random_graph()

                temp = convert_graph_to_code(Graph, all_models_params, also_returns_mapping=True)
                model_json_code = temp["json_code"]
                name_mapping = temp["name_mapping"]

                job_id = md5_hash(model_json_code)  # Or increment an id ? avec un SharedInteger

                # TODO, ici on peut peut etre verifier rapidement que le job n'existe pas deja...

                json_param = {
                    "Graph": {"nodes": list(Graph.nodes), "edges": list(Graph.edges)},
                    "all_models_params": all_models_params,
                    "blocks_to_use": blocks_to_use,
                    "job_id": job_id,
                }

                all_params1.append(json_param)
                all_params2.append(model_json_code)

                all_names_mapping.append(name_mapping)

            self.auto_ml_guider.fit_metric_model()
            # On refit le 'Guider'

            # Applique le Guider pour avoir une estimation de la moyenne + variance
            metric_prediction, metric_variance_prediction = self.auto_ml_guider.predict_metric(all_params1)

            if metric_prediction is None or metric_variance_prediction is None:
                ### Should not append ###
                # => I'll just take the first random model
                ii = 0

            else:

                ####################
                ### What to look ###
                ####################

                # 1) Mean + 2 * Std
                benchmark = metric_prediction + 2 * np.sqrt(metric_variance_prediction)

                # 2) Proba new >= best =>
                # (Mean - Best) / Std

                # 3) E( New * (1[new >= best]) )

                # Rmk : si on utilise des rang, best presque 1

                # ii = np.argmax(benchmark)

                def softmax(benchmark, T=1):
                    ss = np.std(benchmark)
                    if ss == 0:
                        return 1 / len(benchmark) * np.ones(len(benchmark), dtype=np.float32)
                    else:
                        nbenchmark = (benchmark - np.mean(benchmark)) / ss
                        exp_nbenchmark = np.exp(nbenchmark / T)
                        return exp_nbenchmark / exp_nbenchmark.sum()

                probas = softmax(benchmark, T=0.1)
                # Rmk : on va prendre a peu pres le plus best avec une proba 1/4 (suivant les tests)
                # On peut aussi faire une heuristic en suppossant benchmark uniform (pas tres loin de la verite vu qu'on a fitter un rank...)
                # TODO : on peut faire descendre la temperature en court de route.... a peu pres equivalent à gérer l'explortion...
                probas[pd.isnull(probas)] = 0.0
                probas[np.isinf(probas)] = 0.0

                if probas.sum() == 0:
                    ii = np.random.choice(len(probas), size=1)[0]
                else:
                    probas = probas / probas.sum()
                    ii = np.random.choice(len(probas), size=1, p=probas)[0]
                # Comme ca je prend pas l'argmax, mais quelque chose d'un peu plus exploratoir
                # ... peut etre que argmax ca marcherait mieux (surement plus petite variation autour du meilleurs model)

                # On peut aussi virer les trucs vraiment pas bon...
                # Ou tirer au hasard parmis les meilleurs ..
                # etc

                # => a revoir

            model_json_code = all_params2[ii]
            json_param = all_params1[ii]
            name_mapping = all_names_mapping[ii]

        elif job_type in ("default", "block_search"):

            if job_type == "default":
                logger.info("I'll create a default model...")
            else:
                logger.info("I'll create a block-search model...")

            ###########################
            ### This is a defalt job ###
            ###########################

            Graph, all_models_params, blocks_to_use = iter_next

            temp = convert_graph_to_code(Graph, all_models_params, also_returns_mapping=True)
            model_json_code = temp["json_code"]
            name_mapping = temp["name_mapping"]

            job_id = md5_hash(model_json_code)  # Or increment an id ? avec un SharedInteger
            # TODO : check if we have already done that model... which can happen in the controller re-booted...

            json_param = {
                "Graph": {"nodes": list(Graph.nodes), "edges": list(Graph.edges)},
                "all_models_params": all_models_params,
                "blocks_to_use": blocks_to_use,
                "job_id": job_id,
            }

        else:
            raise NotImplementedError("I don't know this job_type %s, please check the code" % job_type)

            ########
            # Test #
            ########

        #            temp_df = self.auto_ml_guider.result_reader.params_to_df(all_params1)
        #            temp_df["metric_prediction"] = metric_prediction
        #            temp_df["metric_std_prediction"] = np.sqrt(metric_variance_prediction)
        #            temp_df["benchmark"] = benchmark
        #
        #            condboxplot("Model","metric_prediction",data = temp_df)
        #            conddistplot("Model","metric_prediction",data = temp_df)
        #            condboxplot("Model","metric_std_prediction",data = temp_df)
        #
        #            condboxplot("DimensionReduction","metric_std_prediction",data = temp_df.loc[temp_df["Model"] == "RandomForestClassifier",:])
        #
        #            condboxplot("DimensionReduction","metric_prediction",data = temp_df.loc[temp_df["Model"] == "RandomForestClassifier",:])
        #            plt.cla()
        #            condboxplot("DimensionReduction","metric_prediction",data = temp_df.loc[temp_df["Model"] == "LogisticRegression",:])
        #
        #            condboxplot("DimensionReduction","metric_prediction",data = temp_df.loc[temp_df["Model"] == "RandomForestClassifier",:])
        #            conddistplot("DimensionReduction","metric_prediction",data = temp_df.loc[temp_df["Model"] == "RandomForestClassifier",:])
        #
        #            condboxplot("hasblock_TEXT","metric_prediction",data = temp_df.loc[temp_df["Model"] == "RandomForestClassifier",:])
        ##            temp_df = pd.DataFrame({"model":models,"benchmark":benchmark})
        ##            temp_df.groupby("model")["benchmark"].max()
        ##            temp_df.groupby("model")["benchmark"].mean()
        #
        #            ii = temp_df["metric_prediction"] >= 0.85
        ########

        #########################
        ### Approx or Full CV ###
        #########################
        job_param = {"model_json": model_json_code, "job_type": job_type, "cv_type": "full"}  # default

        if self.job_config.allow_approx_cv:

            ### The model is a GraphPipeline ###
            if model_json_code[0] == en.SpecialModels.GraphPipeline:

                # I'll loop throught the nodes to figure out which one uses y, the what that don't won't be cross-validated
                nodes_not_to_crossvalidate = set()
                for node in json_param["Graph"]["nodes"]:
                    infos = MODEL_REGISTER.informations.get(node[1], None)
                    if infos is not None:
                        if not infos.get("use_y", True):
                            nodes_not_to_crossvalidate.add(name_mapping[node])

                job_param["cv_type"] = "approximate"
                job_param["nodes_not_to_crossvalidate"] = nodes_not_to_crossvalidate

        ####################
        ### Cv threshold ###
        ####################

        if self.auto_ml_guider is None:
            metric_threshold = None
        else:
            metric_threshold = self.auto_ml_guider.find_metric_threshold()

            if self._last_metric_threshold is not None:
                metric_threshold = max(metric_threshold, self._last_metric_threshold)
                # I do that so that, the metric threshold ALWAYS increases which make more sense.
                # Otherwise it could decrease because it is a quantile which can vary

            self._last_metric_threshold = metric_threshold

        if self.job_config.score_base_line is not None:
            job_param["stopping_round"] = 0
            job_param["stopping_threshold"] = self.job_config.score_base_line

            if metric_threshold is not None:
                job_param["stopping_threshold"] = max(metric_threshold, job_param["stopping_threshold"])
                # Attention : si le scorer est une loss ca marche pas !

        else:

            if metric_threshold is not None:
                job_param["stopping_round"] = 0
                job_param["stopping_threshold"] = metric_threshold
            else:
                job_param["stopping_round"] = None
                job_param["stopping_threshold"] = None

        # Rmk : on peut aussi ne pas faire de CV au depart, en mettant ici un stopping_threshold trop haut quoiqu'il arrive

        job_param["job_creation_time"] = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
        # Plutot dans 'json_param'

        logger.info("stopping_threshold : %s" % str(job_param["stopping_threshold"]))
        logger.info("cv_type : %s" % str(job_param["cv_type"]))

        self.data_persister.write(data=json_param, key=job_id, path="param", write_type=SavingType.json)
        self.data_persister.write(data=job_param, key=job_id, path="job_param", write_type=SavingType.json)

        return job_id


# In[]
class MlJobRunner(AbstractJobRunner):
    """ job class that handle the fit of a model and its cv """

    def __init__(self, dfX, y, groups, auto_ml_config, job_config, data_persister, seed=None):

        self.dfX = dfX
        self.y = y
        self.groups = groups

        self.auto_ml_config = auto_ml_config
        self.job_config = job_config

        super(MlJobRunner, self).__init__(
            data_persister=data_persister,
            gc_collect_freq=1,  # call gc every iteration
            draw_random_for_queue=True,  # draw randomly in queue
            max_queue_waiting_time=600,  # wait at most 10 seconds if input queue is empty
            max_done_queue_waiting_time=np.inf,  # wait forever if exit queue is full (which never happend this the exit queue doesn't have a max_size)
            input_queue_sleeping_time=1,
            done_queue_sleeping_time=1,
            seed=seed,
        )

    def prepare(self):
        ##########
        ### CV ###
        ##########
        self.cv = create_cv(
            cv=self.job_config.cv,
            y=self.y,
            classifier=self.auto_ml_config.type_of_problem == en.TypeOfProblem.CLASSIFICATION,
            shuffle=True,
            random_state=123,
        )

        ##############
        ### Scorer ###
        ##############
        self.scorers = OrderedDict()
        for i, s in enumerate(self.job_config.scoring):
            if isinstance(s, str):
                self.scorers[s] = SCORERS[s]
            else:
                self.scorers["scorer_%d" % i] = SCORERS[s]

    def do_job(self, job_id, job_param):
        """ main function that does one cross-validation job """
        model_json_code = job_param["model_json"]  ### Json of the model to test

        stopping_round = job_param.get("stopping_round", None)
        stopping_threshold = job_param.get("stopping_threshold", None)

        cv_type = job_param.get("cv_type", "full")
        nodes_not_to_crossvalidate = job_param.get("nodes_not_to_crossvalidate", None)

        if cv_type not in ("approximate", "full"):
            raise NotImplementedError("I don't know this type of cv %s" % cv_type)

        # 1) create sklearn model
        skmodel = aikit.model_definition.sklearn_model_from_param(model_json_code)

        if self.auto_ml_config.type_of_problem == en.TypeOfProblem.CLASSIFICATION:
            method = "predict_proba"
        elif self.auto_ml_config.type_of_problem == en.TypeOfProblem.CLUSTERING:
            method = "fit_predict"
        else:
            method = "predict"

        # 2)run CV
        error = False
        try:
            if cv_type == "full":
                approximate_cv = False
            elif cv_type == "approximate":
                approximate_cv = True
            else:
                raise NotImplementedError("I don't know this type of cv %s" % cv_type)

            if self.auto_ml_config.type_of_problem == en.TypeOfProblem.CLUSTERING:
                cv_result, yhat = score_from_params_clustering(
                    skmodel, X=self.dfX, scoring=self.scorers, return_predict=True, method=method
                )
            else:
                cv_result, yhat = cross_validation(
                    skmodel,
                    X=self.dfX,
                    y=self.y,
                    groups=self.groups,
                    cv=self.cv,
                    scoring=self.scorers,
                    return_predict=True,
                    method=method,
                    stopping_round=stopping_round,
                    stopping_threshold=stopping_threshold,
                    nodes_not_to_crossvalidate=nodes_not_to_crossvalidate,
                    approximate_cv=approximate_cv,
                )

            # TODO : suivant la CV on va pas tout le temps retourner un yhat
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if isinstance(e, MemoryError):
                gc.collect()

            trace_back = traceback.format_exc()
            error_repr = repr(e)
            error = True

        # 3) Save result
        if not error:
            ##############################
            ## saving when not an error ##
            ##############################

            if self.auto_ml_config.type_of_problem != en.TypeOfProblem.CLUSTERING:
                test_metric = 100 * cv_result["test_%s" % self.job_config.main_scorer].mean()
                train_metric = 100 * cv_result["train_%s" % self.job_config.main_scorer].mean()

                logger.info("train %s : %2.2f%%" % (self.job_config.main_scorer, train_metric))
                logger.info("test  %s : %2.2f%%" % (self.job_config.main_scorer, test_metric))

            self.data_persister.write(data=cv_result, key=job_id, path="result", write_type=SavingType.csv)

            if self.job_config.additional_scoring_function is not None:
                additional_result = self.job_config.additional_scoring_function(cv_result, yhat, self.y, self.groups)
                self.data_persister.write(
                    data=additional_result, key=job_id, path="additional_result", write_type=SavingType.json
                )

            if self.auto_ml_config.type_of_problem == en.TypeOfProblem.CLUSTERING:
                self.data_persister.write(data=yhat, key=job_id, path="labels", write_type=SavingType.csv)

            return error, (cv_result, yhat)
        else:
            #######################
            ## saving when ERROR ##
            #######################

            logger.warning("error on job_id %s" % job_id)
            logger.warning(error_repr)

            logger.error(trace_back)

            self.data_persister.write(data=error_repr, key=job_id, path="error", write_type=SavingType.txt)
            return error, error_repr


class MlJobManager(AbstractJobRunner):
    """ manager job that fill the job queue with job_id """

    def __init__(self, auto_ml_config, job_config, auto_ml_guider, data_persister, seed=None):

        self.auto_ml_guider = auto_ml_guider
        self.auto_ml_config = auto_ml_config
        self.job_config = job_config

        super(MlJobManager, self).__init__(
            data_persister=data_persister,
            gc_collect_freq=10,
            draw_random_for_queue=None,
            max_queue_waiting_time=-1,
            max_done_queue_waiting_time=np.inf,
            seed=seed,
        )

    def get_job_queue(self):
        return JobManagerQueue(
            auto_ml_config=self.auto_ml_config,
            job_config=self.job_config,
            auto_ml_guider=self.auto_ml_guider,
            data_persister=self.data_persister,
            random_state=self.random_state,
        )

    def get_job_done(self):
        ### job done queue => should be job_queue
        return self.data_persister.new_queue(
            path="job_queue", write_type=SavingType.json, random=False, max_queue_size=10
        )

    def do_job(self, job_id, job_param):
        return self  # nothing to do... I just needed to draw the job !


# In[] : Result Analyser


def generic_unest(p):
    """ generic function to unnest a dictionnary """

    def key_repr(k):
        """ unnest key and represent them as strings """
        return "_".join(unnest_tuple(k))

    result = {}
    for k, v in p.items():
        if isinstance(v, dict):
            nested_result = generic_unest(v)
            for k2, v2 in nested_result.items():
                result[key_repr(k) + "__" + key_repr(k2)] = v2
        else:
            result[key_repr(k)] = v
    return result


class AutoMlResultReader(object):
    """ helper to read the results of an AutoMl experiment """

    def __init__(self, data_persister):
        self.data_persister = data_persister

        # Preparation
        self._all_results_key = None
        self._load_all_results_cache = None

        self._all_params_key = None
        self._load_all_params_cache = None

        self._all_error_key = None
        self._load_all_errors_cache = None

    ###############
    ### Results ###
    ###############

    def load_all_results(self, aggregate=True):
        """ load the DataFrame with all the results """
        ########################
        ### Retrive all keys ###
        ########################
        all_results_key = sorted(self.data_persister.alls(path="result", write_type=SavingType.csv))

        # If same keys => return direct
        # TODO : do something more robust : everytime something is added recompute what is needed...
        if self._all_results_key is not None and all_results_key == self._all_results_key:
            return self._load_all_results_cache
        else:
            self._all_results_key = all_results_key

        #######################
        ### Load everything ###
        #######################
        all_results = []
        for key in all_results_key:
            df = self.data_persister.read_from_cache(path="result", key=key, write_type=SavingType.csv)
            df["job_id"] = key

            all_results.append(df)

        # Attention => Beaucoup trop long de tout recharcher des qu'on a un truc nouveau <= ###

        #########################
        ### Concat everything ###
        #########################
        if len(all_results) == 0:
            df_result = pd.DataFrame(columns=["job_id"])  # Empty DataFrame with one column : job_id

        else:

            df_result = pd.concat(all_results, axis=0)
            if aggregate:
                gp = df_result.groupby("job_id")
                df_result = gp.mean()
                df_result["NB"] = gp.count().iloc[:, 0].values
                df_result = df_result.reset_index()

            ########################
            ### Re-order columns ###
            ########################
            cols = list(df_result.columns)
            cols = (
                intersect(["job_id"], cols)
                + sorted(diff(cols, ["job_id", "time", "time_score", "NB"]))
                + intersect(["time", "time_score", "NB"], cols)
            )

            assert sorted(cols) == sorted(list(df_result.columns))

            df_result = df_result.loc[:, cols]

        self._load_all_results_cache = df_result

        return df_result

    ##################
    ### Parameters ###
    ##################

    @staticmethod
    def auto_ml_unnest_param(p):
        """ unnest a param dictionnary,
        skipping 'ColumnsSelector' and 'columns_to_use'

        """
        res = {}
        for k, v in p["all_models_params"].items():
            if k[1][1] != "ColumnsSelector":
                for k2, v2 in v.items():
                    if k2 != "columns_to_use":
                        res["__".join(k[1]) + "__" + k2] = v2

                res[k[1][0]] = k[1][1]

        for b in p["blocks_to_use"]:
            res["hasblock_%s" % b] = 1

        # Rmk : les autres choses mise dans les params ne sont pas ressorties
        # Si on met le 'timestamps' ou le type de job , ...

        res["job_id"] = p["job_id"]
        return res

    @classmethod
    def params_to_df(cls, all_params):
        """ transform a dictionnary of param into a DataFrame

        this DataFrame will be the entry point of the Meta-Model that tries to predict the benchmark
        DO NOT include columns that aren't relevant to the models

        """
        if len(all_params) == 0:
            return pd.DataFrame(columns=["job_id"])  # Empty DataFrame with one column : job_id

        ##############
        ### Unnest ###
        ##############
        all_new_params = [cls.auto_ml_unnest_param(p) for p in all_params]

        ##############
        ### Concat ###
        ##############
        df_params = pd.DataFrame(all_new_params)

        ###########################################
        ### Fill Missing Value for step columns ###
        ###########################################
        for cat in en.StepCategories.alls:
            if cat in list(df_params.columns):
                df_params.loc[df_params[cat].isnull(), cat] = "--nothing--"

        ########################################
        ### Fill Missing Value for block use ###
        ########################################
        for c in df_params.columns:
            if c.startswith("hasblock_"):
                df_params.loc[df_params[c].isnull(), c] = 0

        ########################
        ### Re-order columns ###
        ########################
        def reorder_param_col(cols):
            return sorted(cols, key=lambda x: tuple(x.split("__")))

        cols = list(df_params.columns)
        cols_hasblock = sorted([c for c in cols if c.startswith("hasblock_")])

        cols = intersect(["job_id"], cols) + cols_hasblock + reorder_param_col(diff(cols, ["job_id"] + cols_hasblock))
        # 1) job_id
        # 2) hasblock columns
        # 3) then model and hyper-parameters

        assert sorted(cols) == sorted(list(df_params.columns))
        df_params = df_params.loc[:, cols]

        return df_params

    def load_all_other_params(self):
        """ load all the other params """
        all_params1_key = sorted(self.data_persister.alls(path="job_param", write_type=SavingType.json))

        if len(all_params1_key) == 0:
            return pd.DataFrame(columns=["job_id"])

        all_params = []
        for key in all_params1_key:
            param = self.data_persister.read_from_cache(path="job_param", key=key, write_type=SavingType.json)
            param["job_id"] = key
            all_params.append({k: v for k, v in param.items() if k != "model_json"})

        df = pd.DataFrame(all_params)
        cols = list(df.columns)
        cols = intersect(["job_id"], cols) + sorted(diff(cols, ["job_id"]))

        if "job_creation_time" in cols:
            df = df.sort_values(by="job_creation_time")

        return df.loc[:, cols]

    def load_all_params(self):
        """ load all the params """
        #########################
        ### Retrieve all keys ###
        #########################
        all_params1_key = sorted(self.data_persister.alls(path="param", write_type=SavingType.json))
        # all_params2_key = data_persister.alls(path = "job_param" , write_type = SavingType.json)

        if self._all_params_key is not None and self._all_params_key == all_params1_key:
            return self._load_all_params_cache
        else:
            self._all_params_key = all_params1_key

        #######################
        ### Load Everything ###
        #######################
        all_params = []
        for key in all_params1_key:
            param = self.data_persister.read_from_cache(path="param", key=key, write_type=SavingType.json)
            param["job_id"] = key
            all_params.append(param)

        df_params = self.params_to_df(all_params)

        self._load_all_params_cache = df_params

        return df_params

    ##############
    ### Errors ###
    ##############

    def load_all_errors(self):
        """ load all errors """
        ########################
        ### Retrive all keys ###
        ########################
        all_error_key = sorted(self.data_persister.alls(path="error", write_type=SavingType.txt))

        if self._all_error_key is not None and self._all_error_key == all_error_key:
            return self._load_all_errors_cache
        else:
            self._all_error_key = all_error_key

        #######################
        ### Load Everything ###
        #######################
        all_errors = []
        for key in all_error_key:
            msg = self.data_persister.read_from_cache(path="error", key=key, write_type=SavingType.txt)
            all_errors.append({"error_msg": msg, "job_id": key, "has_error": True})

        ###################
        ### Concatenate ###
        ###################
        if len(all_errors) == 0:
            df_error = pd.DataFrame(columns=["job_id"])  # Empty DataFrame with one column : job_id
        else:
            df_error = pd.DataFrame(all_errors).loc[:, ["job_id", "error_msg", "has_error"]]

        self._load_all_errors_cache = df_error

        return df_error

    ######################
    ###  Other result  ###
    ######################
    def load_additional_results(self):
        """ load the things saved in 'additional_result' """
        all_params1_key = sorted(self.data_persister.alls(path="additional_result", write_type=SavingType.json))

        if len(all_params1_key) == 0:
            return pd.DataFrame(columns=["job_id"])  # empty DataFrame with 'job_id' columns

        all_params = []
        for key in all_params1_key:
            param = self.data_persister.read_from_cache(path="additional_result", key=key, write_type=SavingType.json)
            if param is not None:
                param["job_id"] = key
                all_params.append(param)

        df = pd.DataFrame(all_params)
        cols = list(df.columns)
        cols = intersect(["job_id"], cols) + sorted(diff(cols, ["job_id"]))

        return df.loc[:, cols]


# In[]
