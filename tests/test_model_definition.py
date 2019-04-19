# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:40:18 2018

@author: Lionel Massoulard
"""

import copy


from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from aikit.model_definition import sklearn_model_from_param

from aikit.transformers.base import BoxCoxTargetTransformer, KMeansTransformer, TruncatedSVDWrapper, NumImputer
from aikit.transformers.categories import NumericalEncoder
from aikit.transformers.text import CountVectorizerWrapper
from aikit.models.stacking import StackerClassifier

from aikit.pipeline import GraphPipeline


class Test_sklearn_model_from_param:
    def test_random_forest(self):
        #####################
        ### Random Forest ###
        #####################

        param1 = ("RandomForestClassifier", {"n_estimators": 100, "criterion": "entropy"})
        param1_c = copy.deepcopy(param1)

        model1 = sklearn_model_from_param(param1)

        assert isinstance(model1, RandomForestClassifier)
        assert model1.n_estimators == 100

        assert param1 == param1_c  # verif that param was not modified inside function

    def test_logistic_regression(self):
        ###########################
        ### Logistic Regression ###
        ###########################
        from sklearn.linear_model import LogisticRegression

        param2 = ("LogisticRegression", {"C": 10})
        param2_c = copy.deepcopy(param2)

        model2 = sklearn_model_from_param(param2)

        assert isinstance(model2, LogisticRegression)
        assert model2.C == 10

        assert param2 == param2_c  # verif that param was not modified inside function

    def test_graph_pipeline(self):
        #####################
        ### GraphPipeline ###
        #####################

        param3 = (
            "GraphPipeline",
            {
                "models": {
                    "svd": ("TruncatedSVDWrapper", {"n_components": 2}),
                    "logit": ("LogisticRegression", {"C": 10}),
                },
                "edges": [("svd", "logit")],
            },
        )

        param3_c = copy.deepcopy(param3)

        model3 = sklearn_model_from_param(param3)

        assert isinstance(model3, GraphPipeline)
        assert isinstance(model3.models["logit"], LogisticRegression)
        assert isinstance(model3.models["svd"], TruncatedSVDWrapper)

        assert param3 == param3_c

    def test_graph_pipeline_list(self):
        #####################
        ### GraphPipeline ###
        #####################

        # Test when inputs are list and not tuples

        param4 = [
            "GraphPipeline",
            {
                "edges": [["encoder", "imputer", "rf"], ["vect", "svd", "rf"]],
                "models": {
                    "encoder": [
                        "NumericalEncoder",
                        {
                            "columns_to_use": ["^BLOCK_", "^NUMBERTOKEN_", "^DATETOKEN_", "^CURRENCYTOKEN_"],
                            "regex_match": True,
                        },
                    ],
                    "imputer": ["NumImputer", {}],
                    "rf": ["RandomForestClassifier", {"n_estimators": 500}],
                    "svd": ["TruncatedSVDWrapper", {"n_components": 200}],
                    "vect": [
                        "CountVectorizerWrapper",
                        {
                            "analyzer": "char",
                            "columns_to_use": ["STRINGLEFTOF", "STRINGABOVEOF"],
                            "ngram_range": [1, 4],
                        },
                    ],
                },
            },
        ]

        param4_c = copy.deepcopy(param4)

        model4 = sklearn_model_from_param(param4)

        assert isinstance(model4, GraphPipeline)
        assert isinstance(model4.models["encoder"], NumericalEncoder)
        assert isinstance(model4.models["imputer"], NumImputer)
        assert isinstance(model4.models["vect"], CountVectorizerWrapper)
        assert isinstance(model4.models["svd"], TruncatedSVDWrapper)
        assert isinstance(model4.models["rf"], RandomForestClassifier)

        assert param4 == param4_c

    def test_boxcox_target_transformer(self):

        ## syntax 1 ##

        params = ("BoxCoxTargetTransformer", ("RandomForestClassifier", {}))

        params_c = copy.deepcopy(params)

        model = sklearn_model_from_param(params_c)
        assert isinstance(model, BoxCoxTargetTransformer)
        assert isinstance(model.model, RandomForestClassifier)
        assert params == params_c

        ## syntax 2 ##
        params = ("BoxCoxTargetTransformer", ("RandomForestClassifier", {}), {"ll": 10})

        params_c = copy.deepcopy(params)

        model = sklearn_model_from_param(params_c)
        assert isinstance(model, BoxCoxTargetTransformer)
        assert isinstance(model.model, RandomForestClassifier)
        assert model.ll == 10
        assert params == params_c

        ## syntax 3 ##
        params = ("BoxCoxTargetTransformer", {"model": ("RandomForestClassifier", {}), "ll": 10})

        params_c = copy.deepcopy(params)

        model = sklearn_model_from_param(params_c)

        assert isinstance(model, BoxCoxTargetTransformer)
        assert isinstance(model.model, RandomForestClassifier)
        assert model.ll == 10
        assert params == params_c

    def boxcox_and_graphpipeline(self):

        params = (
            "GraphPipeline",
            {
                "edges": [("NumericalEncoder", "BoxCoxTargetTransformer")],
                "models": {
                    "BoxCoxTargetTransformer": (
                        "BoxCoxTargetTransformer",
                        (
                            "GraphPipeline",
                            {
                                "edges": [("KMeansTransformer", "RandomForestClassifier")],
                                "models": {
                                    "KMeansTransformer": ("KMeansTransformer", {"n_clusters": 10}),
                                    "RandomForestClassifier": ("RandomForestClassifier", {"n_estimators": 10}),
                                },
                            },
                        ),
                        {"ll": 10},
                    ),
                    "NumericalEncoder": ("NumericalEncoder", {}),
                },
            },
        )

        params_c = copy.deepcopy(params)

        model = sklearn_model_from_param(params_c)

        assert isinstance(model, GraphPipeline)
        assert len(model.models) == 2

        assert "NumericalEncoder" in model.models
        assert isinstance(model.models["NumericalEncoder"], NumericalEncoder)

        assert "BoxCoxTargetTransformer" in model.models
        assert isinstance(model.models["BoxCoxTargetTransformer"], BoxCoxTargetTransformer)

        assert isinstance(model.models["BoxCoxTargetTransformer"].model, GraphPipeline)

        assert set(model.models["BoxCoxTargetTransformer"].model.models.keys()) == {
            "KMeansTransformer",
            "RandomForestClassifier",
        }

        assert isinstance(model.models["BoxCoxTargetTransformer"].model.models["KMeansTransformer"], KMeansTransformer)
        assert isinstance(
            model.models["BoxCoxTargetTransformer"].model.models["RandomForestClassifier"], RandomForestClassifier
        )

        assert params == params_c

    def test_stacking_classifier(self):

        params = (
            "StackerClassifier",
            {
                "models": [("RandomForestClassifier", {}), ("ExtraTreesClassifier", {})],
                "cv": 5,
                "blender": ("LogisticRegression", {}),
            },
        )

        params_c = copy.deepcopy(params)

        model = sklearn_model_from_param(params_c)

        assert isinstance(model, StackerClassifier)
        assert len(model.models) == 2
        assert isinstance(model.models[0], RandomForestClassifier)
        assert isinstance(model.models[1], ExtraTreesClassifier)
        assert isinstance(model.blender, LogisticRegression)
        assert model.cv == 5
