# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 14:25:29 2018

@author: Lionel Massoulard
"""
import pytest

import numpy as np

from aikit.future.automl import MODEL_REGISTRY
from aikit.future.automl._registry import register
from aikit.future.automl.registry._base import ModelRepresentationBase
from aikit.future.automl.registry._text import CountVectorizerTextEncoder
from aikit.future.automl.registry._transformers import KMeansTransformerDimensionReduction, \
    TextTruncatedSVDDimensionReduction, TruncatedSVDDimensionReduction
from aikit.future.enums import StepCategory, VariableType
from aikit.future.util import CLASS_REGISTRY

try:
    from aikit.future.automl.registry._models import LGBMClassifierModel
except ImportError:
    LGBMClassifierModel = None


@pytest.mark.skipif(LGBMClassifierModel is None, reason="lightgbm is not installed")
def test_lgbm_classifier_model():
    hyper_gen = LGBMClassifierModel.get_hyper_parameter()
    all_hypers = [hyper_gen.get_rand() for _ in range(1000)]
    for hyper in all_hypers:
        if hyper["bagging_freq"] == 0:
            assert hyper["bagging_fraction"] == 1.0
            assert hyper["boosting_type"] != "rf"
        else:
            assert hyper["bagging_fraction"] < 1.0


def test_count_vectorizer_text_encoder():
    hyper_gen = CountVectorizerTextEncoder.get_hyper_parameter()
    all_hypers = [hyper_gen.get_rand() for _ in range(1000)]
    for hyper in all_hypers:
        if hyper["analyzer"] == "word":
            assert hyper["ngram_range"] == 1


def test_kmeans_transformer_dimension_reduction():
    hyper_gen = KMeansTransformerDimensionReduction.get_hyper_parameter()
    all_hypers = [hyper_gen.get_rand() for _ in range(100)]
    for hyper in all_hypers:
        assert "result_type" in hyper
        assert "drop_used_columns" in hyper
        assert "drop_unused_columns" in hyper


def test_text_truncated_svd_dimension_reduction():
    hyper_gen = TextTruncatedSVDDimensionReduction.get_hyper_parameter()
    all_hypers = [hyper_gen.get_rand() for _ in range(100)]
    for hyper in all_hypers:
        assert "n_components" in hyper
        assert isinstance(hyper["n_components"], int)


def test_truncated_svd_dimension_reduction():
    hyper_gen = TruncatedSVDDimensionReduction.get_hyper_parameter()
    all_hypers = [hyper_gen.get_rand() for _ in range(100)]
    for hyper in all_hypers:
        assert "n_components" in hyper
        assert isinstance(hyper["n_components"], float)


def test_hyper_init():
    np.random.seed(123)
    for model, hyper in MODEL_REGISTRY.hyper_parameters.items():
        klass = CLASS_REGISTRY[model[1]]
        try:
            klass(**hyper.get_rand())
        except:
            raise


def test_register():
    class TestCategoriesEncoder(object):
        pass

    CLASS_REGISTRY.add_klass(TestCategoriesEncoder)

    @register
    class LGBMCategoriesEncoderCatEncoder(ModelRepresentationBase):
        klass = TestCategoriesEncoder
        category = StepCategory.CategoryEncoder
        type_of_variable = (VariableType.CAT,)
        custom_hyper = {}
        type_of_model = None
        use_y = False
        use_for_block_search = True
        testing_other_param = "this_is_a_test"
        depends_on = (StepCategory.Model,)  # This models needs to be drawn AFTER the Step.Categories.Model,

        @classmethod
        def is_allowed(cls, models_by_steps):
            if models_by_steps[StepCategory.Model] == (StepCategory.Model, 'LGBMClassifier'):
                return True
            else:
                return False

    assert (StepCategory.Model, StepCategory.CategoryEncoder) in MODEL_REGISTRY.step_dependencies.edges
    assert MODEL_REGISTRY._drawing_order[StepCategory.Model] < MODEL_REGISTRY._drawing_order[
        StepCategory.CategoryEncoder]

    key = (StepCategory.CategoryEncoder, "TestCategoriesEncoder")
    assert key in MODEL_REGISTRY.informations

    assert isinstance(MODEL_REGISTRY.informations[key], dict)
    assert MODEL_REGISTRY.informations[key]["type_of_variable"] == (VariableType.CAT,)
    assert MODEL_REGISTRY.informations[key]["type_of_model"] is None
    assert MODEL_REGISTRY.informations[key]["use_y"] is False
    assert MODEL_REGISTRY.informations[key]["use_for_block_search"] is True
    assert MODEL_REGISTRY.informations[key]["testing_other_param"] == "this_is_a_test"
