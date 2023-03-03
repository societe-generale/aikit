# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 14:25:29 2018

@author: Lionel Massoulard
"""
import pytest

import numpy as np

try:
    from aikit.ml_machine.ml_machine_registration import LGBMClassifier_Model
except ImportError:
    LGBMClassifier_Model = None

from aikit.ml_machine.ml_machine_registration import (CountVectorizer_TextEncoder,
                                                      TruncatedSVD_DimensionReduction,
                                                      Text_TruncatedSVD_DimensionReduction,
                                                      KMeansTransformer_DimensionReduction,
                                                      MODEL_REGISTER,
                                                      register,
                                                      ModelRepresentationBase,
                                                      TypeOfVariables,
                                                      StepCategories
                                                      )

from aikit.model_definition import DICO_NAME_KLASS


@pytest.mark.skipif(LGBMClassifier_Model is None, reason="lightgbm is not installed")
def test_LGBMClassifier_Model():
    hyper_gen = LGBMClassifier_Model.get_hyper_parameter()
    all_hypers = [hyper_gen.get_rand() for _ in range(1000)]
    for hyper in all_hypers:
        if hyper["bagging_freq"] == 0:
            assert hyper["bagging_fraction"] == 1.0
            assert hyper["boosting_type"] != "rf"
        else:
            assert hyper["bagging_fraction"] < 1.0


def test_CountVectorizer_TextEncoder():
    hyper_gen = CountVectorizer_TextEncoder.get_hyper_parameter()
    all_hypers = [hyper_gen.get_rand() for _ in range(1000)]
    for hyper in all_hypers:
        if hyper["analyzer"] == "word":
            assert hyper["ngram_range"] == 1


def test_KMeansTransformer_DimensionReduction():
    hyper_gen = KMeansTransformer_DimensionReduction.get_hyper_parameter()
    all_hypers = [hyper_gen.get_rand() for _ in range(100)]
    for hyper in all_hypers:
        assert "result_type" in hyper
        assert "drop_used_columns" in hyper
        assert "drop_unused_columns" in hyper


def test_Text_TruncatedSVD_DimensionReduction():
    hyper_gen = Text_TruncatedSVD_DimensionReduction.get_hyper_parameter()
    all_hypers = [hyper_gen.get_rand() for _ in range(100)]
    for hyper in all_hypers:
        assert "n_components" in hyper
        assert isinstance(hyper["n_components"], int)


def test_TruncatedSVD_DimensionReduction():
    hyper_gen = TruncatedSVD_DimensionReduction.get_hyper_parameter()
    all_hypers = [hyper_gen.get_rand() for _ in range(100)]
    for hyper in all_hypers:
        assert "n_components" in hyper
        assert isinstance(hyper["n_components"], float)



def test_hyper_init():
    np.random.seed(123)
    for model, hyper in MODEL_REGISTER.hyper_parameters.items():

        klass = DICO_NAME_KLASS[model[1]]
        klass(**hyper.get_rand())


def test_register():

    class TestCategoriesEncoder(object):
        pass

    DICO_NAME_KLASS.add_klass(TestCategoriesEncoder)

    @register
    class LGBMCategoriesEncoder_CatEncoder(ModelRepresentationBase):
        klass = TestCategoriesEncoder
        category = StepCategories.CategoryEncoder

        type_of_variable = (TypeOfVariables.CAT, )

        custom_hyper = {}

        type_of_model = None

        use_y = False

        use_for_block_search = True

        testing_other_param = "this_is_a_test"

        depends_on = (StepCategories.Model, ) # This models needs to be drawn AFTER the Step.Categories.Model,

        @classmethod
        def is_allowed(cls, models_by_steps):

            if models_by_steps[StepCategories.Model] == (StepCategories.Model, 'LGBMClassifier'):
                return True
            else:
                return False


    assert (StepCategories.Model, StepCategories.CategoryEncoder) in MODEL_REGISTER.step_dependencies.edges
    assert MODEL_REGISTER._drawing_order[StepCategories.Model] < MODEL_REGISTER._drawing_order[StepCategories.CategoryEncoder]

    key = (StepCategories.CategoryEncoder, "TestCategoriesEncoder")
    assert key in MODEL_REGISTER.informations

    assert isinstance(MODEL_REGISTER.informations[key], dict)
    assert MODEL_REGISTER.informations[key]["type_of_variable"] == (TypeOfVariables.CAT,)
    assert MODEL_REGISTER.informations[key]["type_of_model"] is None
    assert MODEL_REGISTER.informations[key]["use_y"] is False
    assert MODEL_REGISTER.informations[key]["use_for_block_search"] is True
    assert MODEL_REGISTER.informations[key]["testing_other_param"] == "this_is_a_test"

