# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:45:32 2020

@author: Lionel Massoulard
"""

import networkx as nx
import pandas as pd

from aikit.ml_machine.ml_machine import AutoMlConfig, RandomModelGenerator
from aikit.ml_machine.model_graph import convert_graph_to_code
from aikit.model_definition import sklearn_model_from_param

from aikit.model_registration import DICO_NAME_KLASS


def get_default_pipeline(dfX, y, final_model=None):
    """ create a default GraphPipeline for a given model

    Paramterers
    -----------
    dfX : pd.DataFrame
        the training data

    y : array like
        the target

    final_model : None or model instance
        if not None the model at the end the pipeline to use

    Returns
    -------
    a full pipeline to be fitted

    """
    if not isinstance(dfX, pd.DataFrame):
        dfX = pd.DataFrame(dfX)

    auto_ml_config = AutoMlConfig(dfX, y)
    auto_ml_config.guess_everything(dfX, y)
    

    if ('Model','RandomForestClassifier') in auto_ml_config.models_to_keep:
        auto_ml_config.filter_models(Model='RandomForestClassifier')
    else:
        auto_ml_config.filter_models(Model='RandomForestRegressor')

    if len([m for m in auto_ml_config.models_to_keep if m[0] == "Model"]) != 1:
        raise ValueError("I couldn't find a default model")
    
    generator = RandomModelGenerator(auto_ml_config=auto_ml_config)
    iterator = generator.iterator_default_models()

    Graph, all_models_params, blocks_to_use  = next(iterator)    # Retrieve first default model
    
    if final_model is not None:
        if not hasattr(final_model, "fit"):
            raise ValueError("'final_model' should have a 'fit' method")
        # Modify final model by what whas given by the user
        node = None
        for node in Graph.nodes:
            if node[0] == 'Model':
                break
        assert node is not None
        new_node = ('Model',('Model',final_model.__class__.__name__))
        
        Graph = nx.relabel_nodes(Graph,{node:new_node})
        del all_models_params[node]
        all_models_params[new_node] = final_model.get_params()
    
        
        DICO_NAME_KLASS.add_klass(final_model.__class__)

    json_code = convert_graph_to_code(Graph, all_models_params, also_returns_mapping=True)

    model = sklearn_model_from_param(json_code["json_code"])
    
    return model