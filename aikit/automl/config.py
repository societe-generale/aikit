import pandas as pd
import sklearn
from copy import deepcopy
from collections import OrderedDict

from aikit.tools.db_informations import (
    get_columns_informations,
    get_var_type_columns_dico,
    guess_type_of_problem,
    get_all_var_type,
)

from aikit.ml_machine.steps_handling import (
    get_needed_steps,
    filter_model_to_keep,
    modify_var_type_none_to_default,
    modify_var_type_alldefault_to_none,
    create_var_type_from_steps,
)

from aikit.ml_machine.hyper_parameters import HyperMultipleChoice, HyperCrossProduct, HyperComposition
from aikit.ml_machine.ml_machine_registration import MODEL_REGISTER
from aikit.scorer import SCORERS
from aikit import enums


class AutoMlConfig:
    """ class to handle the AutoMlConfiguration, it will contain :

    * information about the type of variable
    * information about the type of problem
    * list of steps  to include in the auto-ml
    * list of models to include in the auto-ml

    TODO:
    - why everything is saved as OrderedDict ?
    - chain updates in setters have to be reviewed
    """

    def __init__(self):
        self.type_of_problem = None
        self.columns_informations = None
        self.needed_steps = None

        self.models_to_keep = None
        self.models_to_keep_block_search = None

        self.specific_hyper = {}
        self.columns_block = None

    def get_params(self):
        result = {}
        for attr in ("type_of_problem", "columns_informations", "needed_steps", "models_to_keep", "models_to_keep_block_search", "specific_hyper", "columns_block"):
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

    def guess_everything(self, dfX, y):
        self.check_dimensions(dfX, y)
        self.columns_informations = get_columns_informations(dfX)
        self.columns_block = get_var_type_columns_dico(self.columns_informations)
        self.type_of_problem = guess_type_of_problem(dfX, y)
        self.needed_steps = get_needed_steps(self.columns_informations, self.type_of_problem)
        self.models_to_keep = filter_model_to_keep(self.type_of_problem, block_search_only=False)
        models_to_keep_block_search = filter_model_to_keep(self.type_of_problem, block_search_only=True)
        self.models_to_keep_block_search = [m for m in models_to_keep_block_search if m in self.models_to_keep]
        return self

    def check_dimensions(self, dfX, y):
        """ perform a few basic test on the base to make sur no mistake was made """
        if dfX is None or y is None:
            return

        if dfX.shape[0] != y.shape[0]:
            raise ValueError("dfX and y don't have the same shape %d vs %d" % (dfX.shape[0], y.shape[0]))

        if len(y.shape) > 1 and y.shape[1] > 1:
            raise ValueError("Multi-output isn't handled yet")

    ############################
    ### columns_informations ###
    ############################
    @property
    def columns_informations(self):
        return self._columns_informations

    @columns_informations.setter
    def columns_informations(self, values):

        if values is None:
            self._columns_informations = None
            return

        # Warning, updates are not caught
        if not isinstance(values, dict):
            raise TypeError(
                "'columns_informations' should be dict-like, instead I got '%s'" % type(values)
            )

        for key, value in values.items():
            if not isinstance(value, dict):
                raise TypeError("value of dictionary should be dictionary")

            # Should have 'HasMissing' #
            if "HasMissing" not in value:
                raise ValueError("'HasMissing' should be in the value of the dictionary")

            if not isinstance(value["HasMissing"], bool):
                raise TypeError("'HasMissing' should be a boolean, instead it is '%s'" % type(value["HasMissing"]))

            # Should have 'TypeOfVariable'
            if "TypeOfVariable" not in value:
                raise ValueError("'TypeOfVariable' should be in the value of the dictionary")

            if value["TypeOfVariable"] not in enums.TypeOfVariables.alls:
                raise ValueError(
                    "Unknown'TypeOfVariable' : %s, it should be among (%s)"
                    % (value["TypeOfVariable"], str(enums.TypeOfVariables.alls))
                )

            # Should have 'ToKeep'
            if "ToKeep" not in value:
                raise ValueError("'ToKeep' should be in the value of the dictionary")

            if not isinstance(value["ToKeep"], bool):
                raise TypeError("'ToKeep' should be a boolean, instead it is '%s'" % type(value["ToKeep"]))

        self._columns_informations = deepcopy(values)

    @columns_informations.deleter
    def columns_informations(self):
        self._columns_informations = None

    #####################
    ### Columns block ###
    #####################
    @property
    def columns_block(self):
        return self._columns_block

    @columns_block.setter
    def columns_block(self, new_columns_block):

        if new_columns_block is None:
            self._columns_block = None
            return

        if not isinstance(new_columns_block, dict):
            raise TypeError("columns_block should be a dictionary, not a %s" % type(new_columns_block))

        for block_name, block_columns in new_columns_block.items():
            if not isinstance(block_name, str):
                raise TypeError("keys of columns_block should be strings, not %s" % type(block_name))

            if not isinstance(block_columns, (tuple, list)):
                raise TypeError("values of columns_block should be lists or tuples, not %s" % type(block_columns))

            for column in block_columns:
                if column not in self.columns_informations:
                    raise ValueError("column %s isn't present in dataset" % column)

                if not self.columns_informations[column]["ToKeep"]:
                    raise ValueError("The column %s shouldn't be keep and can't be included in a block" % column)

        self._columns_block = deepcopy(new_columns_block)

    @columns_block.deleter
    def columns_block(self):
        self._columns_block = None

    @property
    def type_of_problem(self):
        return self._type_of_problem

    #######################
    ### type of problem ###
    #######################
    @type_of_problem.setter
    def type_of_problem(self, value):
        if value is None:
            self._type_of_problem = None
            return

        if value not in enums.TypeOfProblem.alls:
            raise ValueError("'type_of_problem' should be among %s" % str(enums.TypeOfProblem.alls))

        self._type_of_problem = value

    @type_of_problem.deleter
    def type_of_problem(self):
        self._type_of_problem = None

    ####################
    ### Needed steps ###
    ####################
    @property
    def needed_steps(self):
        return self._needed_steps

    @needed_steps.setter
    def needed_steps(self, values):

        if values is None:
            self._needed_steps = values
            return

        if not isinstance(values, (list, tuple)):
            raise TypeError("'needed_steps' should be a list or tuple")

        new_needed_steps = list(values)
        for step in values:
            if not isinstance(step, dict):
                raise TypeError("each step should be a dict, instead it is %s" % type(step))

            if "optional" not in step:
                raise ValueError("'optional' should be in step")
            if not isinstance(step["optional"], bool):
                raise TypeError("'optional' should be a boolean")

            if "step" not in step:
                raise ValueError("'step' should be in step")
            if step["step"] not in enums.StepCategories.alls:
                raise ValueError("Unknown step : %s" % step["step"])

        self._needed_steps = values

    @needed_steps.deleter
    def needed_steps(self):
        self._needed_steps = None

    ######################
    ### models to keep ###
    ######################
    @property
    def models_to_keep(self):
        return self._models_to_keep

    @models_to_keep.setter
    def models_to_keep(self, values):

        if values is None:
            self._models_to_keep = values
            return

        if not isinstance(values, (list, tuple)):
            raise TypeError("new_models_to_keep should be a list")

        for n in values:
            if not isinstance(n, (tuple, list)):
                raise TypeError("all models should be tuple")
            if len(n) != 2:
                raise ValueError("all models should be of size 2")

            if n[0] not in enums.StepCategories.alls:
                raise ValueError("first item should be among StepCategories")
            if n not in MODEL_REGISTER.all_registered:
                raise ValueError("each item should have been registred")

        self._models_to_keep = values

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
    @property
    def models_to_keep_block_search(self):
        return self._models_to_keep_block_search

    @models_to_keep_block_search.setter
    def models_to_keep_block_search(self, values):

        if values is None:
            self._models_to_keep_block_search = values
            return

        if not isinstance(values, (list, tuple)):
            raise TypeError("new_models_to_keep should be a list")

        for n in values:
            if not isinstance(n, (tuple, list)):
                raise TypeError("all models should be tuple")
            if len(n) != 2:
                raise ValueError("all models should be of size 2")

            if n[0] not in enums.StepCategories.alls:
                raise ValueError("first item should be among StepCategories")

            if n not in MODEL_REGISTER.all_registered:
                raise ValueError("each item should have been registred")

            if n not in self.models_to_keep:
                raise ValueError("each item should be in 'models_to_keep'")

        self._models_to_keep_block_search = values

    ################################
    ### Specific HyperParameters ###
    ################################
    @property
    def specific_hyper(self):
        return self._specific_hyper

    @specific_hyper.setter
    def specific_hyper(self, values):

        if values is None or len(values) == 0:
            self._specific_hyper = values
            return

        if self.models_to_keep is None:
            raise ValueError("Please specify models_to_keep first")

        values = deepcopy(values)
        if isinstance(self, dict):
            raise TypeError("specific_hyper should be a dict, instead I got %s" % type(values))

        for key, value in values.items():

            if key not in self.models_to_keep:
                raise ValueError("keys of specific_hyper should be within 'models_to_keep' : unknown %s" % str(key))

            if isinstance(value, dict):
                values[key] = HyperCrossProduct(value)

            if not isinstance(values[key], HyperCrossProduct):
                raise TypeError(
                    "values of specific_hyper should be dict or HyperCrossProduct, instead I got %s" % type(value)
                )

        self._specific_hyper = values

    @specific_hyper.deleter
    def specific_hyper(self):
        self._specific_hyper = None

    ###############
    ### Helpers ###
    ###############
    def is_regression(self):
        return self._type_of_problem == enums.TypeOfProblem.REGRESSION

    def is_classification(self):
        return self._type_of_problem == enums.TypeOfProblem.CLASSIFICATION

    def is_clustering(self):
        return self._type_of_problem == enums.TypeOfProblem.CLUSTERING

    def get_predict_method_name(self):
        if self.is_classification():
            return "predict_proba"
        elif self.is_clustering():
            return "fit_predict"
        else:
            return "predict"

    ############
    ### Repr ###
    ############
    def __repr__(self):
        res = ["type of problem : %s" % self.type_of_problem]
        return super(AutoMlConfig, self).__repr__() + "\n" + "\n".join(res)

    def serialize(self):
        return self.get_params()

    @classmethod
    def deserialize(cls, params):
        return cls.from_params(params)



class JobConfig:
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
        if auto_ml_config.type_of_problem == enums.TypeOfProblem.CLASSIFICATION:
            self.cv = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
        elif auto_ml_config.type_of_problem == enums.TypeOfProblem.CLUSTERING:
            self.cv = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=123)
        else:
            self.cv = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=123)

    @property
    def cv(self):
        return self._cv

    @cv.setter
    def cv(self, value):
        if value is None:
            self._cv = None
            return

        if value is not None and not isinstance(value, int):
            if not hasattr(value, "split") or isinstance(value, str):
                raise ValueError(
                    "Expected cv as an integer, cross-validation "
                    "object (from sklearn.model_selection) "
                    "or an iterable. Got %s." % value
                )
        self._cv = value

    @cv.deleter
    def cv(self):
        self._cv = None

    ###############
    ### Metrics ###
    ###############
    def guess_scoring(self, auto_ml_config):
        if auto_ml_config.type_of_problem == enums.TypeOfProblem.CLASSIFICATION:
            self.scoring = ["accuracy", "log_loss_patched", "avg_roc_auc", "f1_macro"]

        elif auto_ml_config.type_of_problem == enums.TypeOfProblem.CLUSTERING:
            self.scoring = ["silhouette", "calinski_harabasz", "davies_bouldin"]

        else:
            self.scoring = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]

        return self.scoring

    @property
    def scoring(self):
        return self._scoring

    @scoring.setter
    def scoring(self, values):

        if values is None:
            self._scoring = None
            return

        if not isinstance(values, (list, tuple)):
            values = [values]

        for scoring in values:
            if isinstance(scoring, str):
                if scoring not in SCORERS:
                    raise ValueError("I don't know that scorer : %s" % scoring)
            else:
                raise NotImplementedError("for now I can only use pre-defined scorer, entered as string")

        self._scoring = values

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
    def score_base_line(self, value):
        if value is None or pd.isnull(value):
            self._score_base_line = None
            return

        if not pd.api.types.is_number(value):
            raise TypeError("base_line should a be a number, instead I got %s" % type(value))

        self._score_base_line = value

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
    def main_scorer(self, value):
        if value is None:
            self._main_scorer = None
            return

        if value not in self._scoring:
            raise ValueError("main_scorer should be among 'scoring', %s" % value)

        self._main_scorer = value
        self._scoring = [self._main_scorer] + [s for s in self._scoring if s != self._main_scorer]

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
    def additional_scoring_function(self, value):

        if value is None:
            self._additional_scoring_function = None
            return

        if not callable(value):
            raise TypeError("'additional_scoring_function' should be callable")

        self._additional_scoring_function = value

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

