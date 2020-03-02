# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:40:18 2018

@author: Lionel Massoulard
"""

import copy
import json

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from aikit.model_definition import sklearn_model_from_param, param_from_sklearn_model, filtered_get_params

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

        param = ("RandomForestClassifier", {"n_estimators": 150, "criterion": "entropy"})
        param_c = copy.deepcopy(param)

        model = sklearn_model_from_param(param)

        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 150

        assert param == param_c  # verif that param was not modified inside function
        
        param_reverse = param_from_sklearn_model(model)
        assert param_reverse == param

    def test_logistic_regression(self):
        ###########################
        ### Logistic Regression ###
        ###########################
        from sklearn.linear_model import LogisticRegression

        param = ("LogisticRegression", {"C": 10})
        param_c = copy.deepcopy(param)

        model = sklearn_model_from_param(param)

        assert isinstance(model, LogisticRegression)
        assert model.C == 10

        assert param == param_c  # verif that param was not modified inside function

        param_reverse = param_from_sklearn_model(model)
        assert param_reverse == param

    def test_graph_pipeline(self):
        #####################
        ### GraphPipeline ###
        #####################

        param = (
            "GraphPipeline",
            {
                "models": {
                    "svd": ("TruncatedSVDWrapper", {"n_components": 3}),
                    "logit": ("LogisticRegression", {"C": 10}),
                },
                "edges": [("svd", "logit")],
            },
        )

        param_c = copy.deepcopy(param)

        model = sklearn_model_from_param(param)

        assert isinstance(model, GraphPipeline)
        assert isinstance(model.models["logit"], LogisticRegression)
        assert isinstance(model.models["svd"], TruncatedSVDWrapper)
        assert model.models["svd"].n_components==3

        assert param == param_c
        
        param_reverse = param_from_sklearn_model(model)
        assert param_reverse == param


    def test_graph_pipeline_list(self):
        #####################
        ### GraphPipeline ###
        #####################

        # Test when inputs are list and not tuples

        param = (
            "GraphPipeline",
            {
                "edges": [["encoder", "imputer", "rf"], ["vect", "svd", "rf"]],
                "models": {
                    "encoder": (
                        "NumericalEncoder",
                        {
                            "columns_to_use": ["^BLOCK_", "^NUMBERTOKEN_", "^DATETOKEN_", "^CURRENCYTOKEN_"],
                            "regex_match": True,
                        },
                    ),
                    "imputer": ("NumImputer", {}),
                    "rf": ("RandomForestClassifier", {"n_estimators": 500}),
                    "svd": ("TruncatedSVDWrapper", {"n_components": 200}),
                    "vect": (
                        "CountVectorizerWrapper",
                        {
                            "analyzer": "char",
                            "columns_to_use": ["STRINGLEFTOF", "STRINGABOVEOF"],
                            "ngram_range": [1, 4],
                        },
                    ),
                },
            },
        )

        param_c = copy.deepcopy(param)

        model = sklearn_model_from_param(param)

        assert isinstance(model, GraphPipeline)
        assert isinstance(model.models["encoder"], NumericalEncoder)
        assert isinstance(model.models["imputer"], NumImputer)
        assert isinstance(model.models["vect"], CountVectorizerWrapper)
        assert isinstance(model.models["svd"], TruncatedSVDWrapper)
        assert isinstance(model.models["rf"], RandomForestClassifier)

        assert param == param_c
        
        param_reverse = param_from_sklearn_model(model)
        assert param_reverse == param
        
    def test_boxcox_target_transformer(self):

        ## syntax 1 ##

        param = ("BoxCoxTargetTransformer", ("RandomForestClassifier", {}))

        param_c = copy.deepcopy(param)

        model = sklearn_model_from_param(param_c)
        assert isinstance(model, BoxCoxTargetTransformer)
        assert isinstance(model.model, RandomForestClassifier)
        assert param == param_c
        param_reverse = param_from_sklearn_model(model)  # rmk : difference from param because the RandomForest isn't explicitely passed with a named attribute
        assert param_reverse[0] == param[0]
        
        ## syntax 2 ##
        params = ("BoxCoxTargetTransformer", ("RandomForestClassifier", {}), {"ll": 10})

        params_c = copy.deepcopy(params)

        model = sklearn_model_from_param(params_c)
        assert isinstance(model, BoxCoxTargetTransformer)
        assert isinstance(model.model, RandomForestClassifier)
        assert model.ll == 10
        assert params == params_c
        param_reverse = param_from_sklearn_model(model) # rmk : difference from param because the RandomForest isn't explicitely passed with a named attribute

        assert param_reverse[0] == param[0]

        ## syntax 3 ##
        params = ("BoxCoxTargetTransformer", {"model": ("RandomForestClassifier", {}), "ll": 10})

        params_c = copy.deepcopy(params)

        model = sklearn_model_from_param(params_c)

        assert isinstance(model, BoxCoxTargetTransformer)
        assert isinstance(model.model, RandomForestClassifier)
        assert model.ll == 10
        assert params == params_c
        param_reverse = param_from_sklearn_model(model) # rmk : difference from param because the RandomForest isn't explicitely passed with a named attribute
        assert param_reverse == params
        
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
        
        param_reverse = param_from_sklearn_model(model) # rmk : difference from param because the RandomForest isn't explicitely passed with a named attribute

        assert param_reverse[0] == params[0]

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
        param_reverse = param_from_sklearn_model(model) # rmk : difference from param because the RandomForest isn't explicitely passed with a named attribute

        assert param_reverse == params


def test_filtered_get_params():
    forest = RandomForestClassifier(n_estimators=250)
    assert RandomForestClassifier().get_params()["n_estimators"] != 250
    assert filtered_get_params(forest) == {"n_estimators":250}
    
    forest = RandomForestClassifier(n_estimators=250, max_depth=None)
    assert filtered_get_params(forest) == {"n_estimators":250}

    
    model = BoxCoxTargetTransformer(RandomForestClassifier(n_estimators=250), ll=0)
    fparams = filtered_get_params(model)
    
    assert "ll" not in fparams
    assert "model" in fparams
    
    
    model = BoxCoxTargetTransformer(RandomForestClassifier(n_estimators=250), ll=1)
    assert BoxCoxTargetTransformer(RandomForestClassifier()).get_params()["ll"] != 1
    fparams = filtered_get_params(model)
    
    assert "ll" in fparams
    assert fparams["ll"] == 1
    assert "model" in fparams


def test_param_from_sklearn_model():
    # simple RandomForest
    model = RandomForestClassifier(n_estimators=250)
    assert RandomForestClassifier().get_params()["n_estimators"] != 250
    assert param_from_sklearn_model(model, simplify_default=True) == ('RandomForestClassifier', {'n_estimators': 250})
    param = param_from_sklearn_model(model, simplify_default=False)
    assert isinstance(param, tuple)
    assert len(param) == 2
    assert param[0] == "RandomForestClassifier"
    
    assert isinstance(sklearn_model_from_param(param_from_sklearn_model(model)), model.__class__)
    s = json.dumps(param) # check that it can be json serialized
    assert isinstance(s, str)
    
    assert isinstance(sklearn_model_from_param(param_from_sklearn_model(model)), model.__class__)
    
    # Composition model : BoxCoxTargetTransformer of RandomForestClassifier
    model = BoxCoxTargetTransformer(RandomForestClassifier(n_estimators=250), ll=0)
    param = param_from_sklearn_model(model, simplify_default=True)
    assert param == ('BoxCoxTargetTransformer',
                                   {'model': ('RandomForestClassifier', {'n_estimators': 250})})
    
    assert isinstance(sklearn_model_from_param(param_from_sklearn_model(model)), model.__class__)
    s = json.dumps(param) # check that it can be json serialized
    assert isinstance(s, str)
    
    
    # Composition model : BoxCoxTargetTransformer of RandomForestClassifier
    model = BoxCoxTargetTransformer(RandomForestClassifier(n_estimators=250), ll=1)
    param = param_from_sklearn_model(model, simplify_default=True)
    assert param == ('BoxCoxTargetTransformer',
                                   {'ll':1, 'model': ('RandomForestClassifier', {'n_estimators': 250})})
    s = json.dumps(param) # check that it can be json serialized
    assert isinstance(s, str)

    
    assert isinstance(sklearn_model_from_param(param_from_sklearn_model(model)), model.__class__)
    
    # Pipeline
    model = Pipeline([("enc", NumericalEncoder()), ("forest", RandomForestClassifier(n_estimators=250))])
    param = param_from_sklearn_model(model, simplify_default=True)
    assert param == ('Pipeline',
                                   {'steps': [('enc', ('NumericalEncoder', {})),
                                              ('forest', ('RandomForestClassifier', {'n_estimators': 250}))]})
   
    assert isinstance(sklearn_model_from_param(param_from_sklearn_model(model)), model.__class__)
    s = json.dumps(param) # check that it can be json serialized
    assert isinstance(s, str)


    # GraphPipeline
    model = GraphPipeline(models={"enc":NumericalEncoder(),"forest":RandomForestClassifier(n_estimators=250)},
                          edges=[("enc","forest")]
                          )
    
    param = param_from_sklearn_model(model, simplify_default=True)
    assert param == ('GraphPipeline',
             {'models': {'enc': ('NumericalEncoder', {}),
               'forest': ('RandomForestClassifier', {'n_estimators': 250})},
              'edges': [('enc', 'forest')]
              })

    assert isinstance(sklearn_model_from_param(param_from_sklearn_model(model)), model.__class__)
    
    
    # GraphPipeline with verbose = True
    model = GraphPipeline(models={"enc":NumericalEncoder(),"forest":RandomForestClassifier(n_estimators=250)},
                          edges=[("enc","forest")],
                          verbose=True
                          )
    
    param = param_from_sklearn_model(model, simplify_default=True)
    assert param == ('GraphPipeline',
             {'models': {'enc': ('NumericalEncoder', {}),
               'forest': ('RandomForestClassifier', {'n_estimators': 250})},
              'edges': [('enc', 'forest')],
              'verbose':True
              })

    s = json.dumps(param) # check that it can be json serialized
    assert isinstance(s, str)

    model2 = sklearn_model_from_param(param_from_sklearn_model(model))
    assert model2.verbose is True
    assert isinstance(model2, model.__class__)

    # GraphPipeline + composition
    model = GraphPipeline(models={"enc":NumericalEncoder(),
                                  "forest":BoxCoxTargetTransformer(RandomForestClassifier(n_estimators=250), ll=1)},
                          edges=[("enc","forest")]
                          )

    param = param_from_sklearn_model(model, simplify_default=True)
    assert param == ('GraphPipeline',
         {'edges': [('enc', 'forest')],
          'models': {'enc': ('NumericalEncoder', {}),
           'forest': ('BoxCoxTargetTransformer',
            {'ll': 1, 'model': ('RandomForestClassifier', {'n_estimators': 250})})}})

    assert isinstance(sklearn_model_from_param(param_from_sklearn_model(model)), model.__class__)
    s = json.dumps(param) # check that it can be json serialized
    assert isinstance(s, str)


