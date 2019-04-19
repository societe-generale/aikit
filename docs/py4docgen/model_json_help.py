# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 13:47:46 2018

@author: Lionel Massoulard
"""

from aikit.graph_pipeline import GraphPipeline
from aikit.transformers import CountVectorizerWrapper,TruncatedSVDWrapper
from sklearn.linear_model import LogisticRegression



gpipeline = GraphPipeline(models = {"vect" : CountVectorizerWrapper(analyzer="char",ngram_range=(1,4)),
                                        "svd"  : TruncatedSVDWrapper(n_components=400) ,
                                        "logit" : LogisticRegression(class_weight="balanced")},
                               edges = [("vect","svd","logit")]
                               )

json_object = ("GraphPipeline", {"models": {"vect" : ("CountVectorizerWrapper"  , {"analyzer":"char","ngram_range":(1,4)} ),
                             "svd"  : ("TruncatedSVDWrapper"     , {"n_components":400}) ,
                             "logit": ("LogisticRegression" , {"class_weight":"balanced"}) },
                  "edges":[("vect","svd","logit")]
                  })
                        
from aikit.model_generation import sklearn_model_from_param

sklearn_model_from_param(json_object)

from aikit.json_helper import save_json

from aikit.json_helper import save_json
save_json(json_object, fname ="model.json")