# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 17:50:14 2018

@author: Lionel Massoulard
"""




from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from aikit.pipeline import GraphPipeline
from aikit.transformers import CountVectorizerWrapper, TruncatedSVDWrapper
from aikit.transformers_categories import NumericalEncoder

gpipeline = GraphPipeline(models = {"vect" : CountVectorizerWrapper(analyzer="char",ngram_range=(1,4), columns_to_use=["text1","text2"]),
                                    "cat"  : NumericalEncoder(columns_to_use=["cat1","cat2"]) , 
                                    "rf"   : RandomForestClassifier(n_estimators=100)}  ,
                               edges = [("vect","rf"),("cat","rf")]
                               )


gpipeline = GraphPipeline(models = {"encoder":NumericalEncoder(columns_to_use = ["cat1","cat2"]),
                                "imputer": NumImputer(),
                                "vect": CountVectorizerWrapper(analyzer="word",columns_to_use=["cat1","cat2"]),
                                "svd":TruncatedSVDWrapper(n_components=50),
                                "rf":RandomForestClassifier(n_estimators=100)
                                    },
                    edges = [("encoder","imputer","rf"),("vect","svd","rf")] )



gpipeline_mix3 = GraphPipeline(models = {"encoder" : NumericalEncoder(columns_to_use = ["cat1","cat2"],
                                         "imputer" : NumImputer(),
                                         "vect"    : CountVectorizerWrapper(analyzer="word",columns_to_use = ["text1","text2"],
                                        "svd"      : TruncatedSVDWrapper(n_components=50)
                                         "rf"      : RandomForestClassifier(n_estimators=500),
                                         
                                        },
                        edges = [("encoder","imputer","rf"),("vect","rf"),("vect","svd","rf")] )



    gpipeline = GraphPipeline(models = [( "vect" , CountVectorizerWrapper(analyzer="char",ngram_range=(1,4)) ),
                                        ( "svd"   , TruncatedSVDWrapper(n_components=400) ), 
                                        ( "logit" , LogisticRegression(class_weight="balanced") )] )