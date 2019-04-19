 # -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 22:33:30 2018

@author: Lionel Massoulard
"""


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from sklearn.models import StratifiedKFold
from aikit.models import OutSamplerTransformer, StackerClassifier
from aikit.pipeline import GraphPipeline

# In[]

stacker = StackerClassifier( models = [RandomForestClassifier() , LGBMClassifier(), LogisticRegression()],
                             cv = 10,
                             blender = LogisticRegression()
                            )

from sklearn.models.stacking import StratifiedKFold, OutSamplerTransformer

from aikit.pipeline import GraphPipeline
from aikit.transformers import PassThrough

cv = StratifiedKFold(10, shuffle=True, random_state=123)

stacker = GraphPipeline(models = {
    "rf":OutSamplerTransformer(RandomForestClassifier() , cv = cv),
    "lgbm":OutSamplerTransformer(LGBMClassifier() , cv = cv),
    "logit":OutSamplerTransformer(LogisticRegression(), cv = cv),
    "blender":LogisticRegression()
    }, edges = [("rf","blender"),("lgbm","blender"),("logit","blender")])


stacker = GraphPipeline(models = {
    "rf"   : OutSamplerTransformer(RandomForestClassifier() , cv = cv),
    "lgbm" : OutSamplerTransformer(LGBMClassifier() , cv = cv),
    "logit": OutSamplerTransformer(LogisticRegression(), cv = cv),
    "pass" : PassThrough(),
    "blender":LogisticRegression()
    }, edges = [("rf","blender"),
                ("lgbm","blender"),
                ("logit","blender"),
                ("pass", "blender")
                ])


 # In[]
from aikit.transformers import NumImputer, CountVectorizerWrapper, NumericalEncoder

stacker = GraphPipeline(models = {
    "enc"  : NumericalEncoder(),
    "imp"  : NumImputer(),
    "rf"   : OutSamplerTransformer(RandomForestClassifier() , cv = cv),
    "lgbm" : OutSamplerTransformer(LGBMClassifier() , cv = cv),
    "logit": OutSamplerTransformer(LogisticRegression(), cv = cv),
    "blender":LogisticRegression()
    }, edges = [("enc","imp"),
                ("imp","rf","blender"),
                ("imp","lgbm","blender"),
                ("imp","logit","blender")
                ])



stacker = GraphPipeline(models = {
    "enc"  : NumericalEncoder(columns_to_use= ["cat1","cat2","num1","num2"]),
    "imp"  : NumImputer(),
    "cv"   : CountVectorizerWrapper(columns_to_use = ["text1","text2"]),
    "logit": OutSamplerTransformer(LogisticRegression(), cv = cv),
    "lgbm" : OutSamplerTransformer(LGBMClassifier() , cv = cv),
    "blender":LogisticRegression()
    }, edges = [("enc","imp","lgbm","blender"),
                ("cv","logit","blender")
                ])

# In[]
from sklearn.preprocessing import OneHotEncoder
class NumericalEncoder(OneHotEncoder):
    pass
class NumImputer(OneHotEncoder):
    pass

stacker = GraphPipeline(models = {
    "enc"  : NumericalEncoder(),
    "imp"  : NumImputer(),
    "rf"   : OutSamplerTransformer( RandomForestClassifier(class_weight = "auto"), cv = 10),
    "scaling":LogisticRegression()
    }, edges = [('enc','imp','rf','scaling')]
)

# In[]
import numpy as np
X = np.random.randn(100,10)
y = 1*(np.random.randn(100)>0)

from importlib import reload
import aikit.models.stacking
reload(aikit.models.stacking)
from aikit.models.stacking import OutSamplerTransformer

model = OutSamplerTransformer(RandomForestClassifier())
model.fit(X,y)

assert model.get_feature_names() == ["RandomForestClassifier__1"]

# In[]
y = np.array(["a","b","c"])[np.random.randint(0,3,100)]

model = OutSamplerTransformer(RandomForestClassifier())
model.fit(X,y)

assert model.get_feature_names() == ['RandomForestClassifier__a','RandomForestClassifier__b','RandomForestClassifier__c']

