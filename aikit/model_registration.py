# -*- coding: utf-8 -*-
"""
Created on Mon May  7 14:25:48 2018

@author: Lionel Massoulard
"""

# TODO : here registrer everything needed in that file as well.. so that I can deliver it without delivering everything else


class _DICO_NAME_KLASS(object):
    """ simple class to handle dictionnary of klass - name """

    def __init__(self):
        self._mapping = {}

    def add_klass(self, klass):
        self._mapping[klass.__name__] = klass

    def __getitem__(self, klass_name):
        return self._mapping[klass_name]

    def __repr__(self):
        result = ["registered klasses :"] + [s for s in sorted(self._mapping.keys())]
        return "\n".join(result)

    def get(self, key, default=None):
        return self._mapping.get(key, default)


DICO_NAME_KLASS = _DICO_NAME_KLASS()

# In[]

# TODO : il faut aussi enregister certain function un peu particuliere, type :
# "StratifiedKFold", "KFold"

from sklearn.pipeline import Pipeline
from aikit.pipeline import GraphPipeline

from aikit.transformers import ColumnsSelector  # ,ModelsUnion
from aikit.models import OutSamplerTransformer, StackerClassifier, StackerRegressor

from aikit.transformers import FeaturesSelectorClassifier, FeaturesSelectorRegressor, TruncatedSVDWrapper, PassThrough
from aikit.transformers import PCAWrapper
from aikit.transformers import BoxCoxTargetTransformer, NumImputer, KMeansTransformer, CdfScaler
from aikit.transformers import Word2VecVectorizer, CountVectorizerWrapper, Char2VecVectorizer
from aikit.transformers import TextNltkProcessing, TextDefaultProcessing, TextDigitAnonymizer
from aikit.transformers import TargetEncoderClassifier, TargetEncoderEntropyClassifier, TargetEncoderRegressor
from aikit.transformers import NumericalEncoder

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso

try:
    import lightgbm
except ImportError:
    lightgbm = None
    print("I wont be able to import LightGBM")
from aikit.models import DBSCANWrapper, KMeansWrapper, AgglomerativeClusteringWrapper


############################
## Special 'transformers' ##
############################
DICO_NAME_KLASS.add_klass(PassThrough)
DICO_NAME_KLASS.add_klass(Pipeline)
DICO_NAME_KLASS.add_klass(GraphPipeline)
DICO_NAME_KLASS.add_klass(ColumnsSelector)

####################
## Stacking tools ##
####################
DICO_NAME_KLASS.add_klass(OutSamplerTransformer)
DICO_NAME_KLASS.add_klass(StackerRegressor)
DICO_NAME_KLASS.add_klass(StackerClassifier)

##################
## Transformers ##
##################

# Selector
DICO_NAME_KLASS.add_klass(FeaturesSelectorClassifier)
DICO_NAME_KLASS.add_klass(FeaturesSelectorRegressor)

# Text Vectorizer
DICO_NAME_KLASS.add_klass(CountVectorizerWrapper)
DICO_NAME_KLASS.add_klass(Word2VecVectorizer)
DICO_NAME_KLASS.add_klass(Char2VecVectorizer)

# Text preprocessor
DICO_NAME_KLASS.add_klass(TextNltkProcessing)
DICO_NAME_KLASS.add_klass(TextDefaultProcessing)
DICO_NAME_KLASS.add_klass(TextDigitAnonymizer)

DICO_NAME_KLASS.add_klass(TruncatedSVDWrapper)
DICO_NAME_KLASS.add_klass(PCAWrapper)
DICO_NAME_KLASS.add_klass(BoxCoxTargetTransformer)
DICO_NAME_KLASS.add_klass(NumImputer)
DICO_NAME_KLASS.add_klass(CdfScaler)


DICO_NAME_KLASS.add_klass(KMeansTransformer)

# Cat Encoder
DICO_NAME_KLASS.add_klass(NumericalEncoder)
DICO_NAME_KLASS.add_klass(TargetEncoderClassifier)
DICO_NAME_KLASS.add_klass(TargetEncoderEntropyClassifier)
DICO_NAME_KLASS.add_klass(TargetEncoderRegressor)

############
## Models ##
############

# Classifier
DICO_NAME_KLASS.add_klass(RandomForestClassifier)
DICO_NAME_KLASS.add_klass(ExtraTreesClassifier)
DICO_NAME_KLASS.add_klass(LogisticRegression)
DICO_NAME_KLASS.add_klass(Lasso)
if lightgbm is not None:
    DICO_NAME_KLASS.add_klass(lightgbm.LGBMClassifier)


# Regressor
DICO_NAME_KLASS.add_klass(RandomForestRegressor)
DICO_NAME_KLASS.add_klass(ExtraTreesRegressor)
DICO_NAME_KLASS.add_klass(Ridge)
if lightgbm is not None:
    DICO_NAME_KLASS.add_klass(lightgbm.LGBMRegressor)

# Clusterer
DICO_NAME_KLASS.add_klass(KMeansWrapper)
DICO_NAME_KLASS.add_klass(DBSCANWrapper)
DICO_NAME_KLASS.add_klass(AgglomerativeClusteringWrapper)
