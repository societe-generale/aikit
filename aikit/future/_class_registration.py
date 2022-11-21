from sklearn.pipeline import Pipeline
from aikit.pipeline import GraphPipeline

from aikit.transformers import ColumnsSelector
from aikit.models import OutSamplerTransformer, StackerClassifier, StackerRegressor, KMeansWrapper, DBSCANWrapper, \
    AgglomerativeClusteringWrapper

from aikit.transformers import FeaturesSelectorClassifier, FeaturesSelectorRegressor, TruncatedSVDWrapper, PassThrough
from aikit.transformers import PCAWrapper
from aikit.transformers import BoxCoxTargetTransformer, NumImputer, KMeansTransformer, CdfScaler
from aikit.transformers import Word2VecVectorizer, CountVectorizerWrapper, Char2VecVectorizer
from aikit.transformers import TextNltkProcessing, TextDefaultProcessing, TextDigitAnonymizer
from aikit.transformers import TargetEncoderClassifier, TargetEncoderEntropyClassifier, TargetEncoderRegressor
from aikit.transformers import NumericalEncoder

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso

from .util import CLASS_REGISTRY

try:
    import nltk
except ImportError:
    nltk = None
    print("NLTK not available, AutoML won't run with NLTK transformers")

try:
    import gensim
except ImportError:
    gensim = None
    print("Gensim not available, AutoML won't run with Gensim models")

try:
    import lightgbm
except ImportError:
    lightgbm = None
    print("LightGBM not available, AutoML won't run with LightGBM models")

# Pipelines
CLASS_REGISTRY.add_klass(PassThrough)
CLASS_REGISTRY.add_klass(Pipeline)
CLASS_REGISTRY.add_klass(GraphPipeline)
CLASS_REGISTRY.add_klass(ColumnsSelector)

# Stacking tools
CLASS_REGISTRY.add_klass(OutSamplerTransformer)
CLASS_REGISTRY.add_klass(StackerRegressor)
CLASS_REGISTRY.add_klass(StackerClassifier)

# Feature selection
CLASS_REGISTRY.add_klass(FeaturesSelectorClassifier)
CLASS_REGISTRY.add_klass(FeaturesSelectorRegressor)

# Text vectorizers
CLASS_REGISTRY.add_klass(CountVectorizerWrapper)
if gensim is not None:
    CLASS_REGISTRY.add_klass(Word2VecVectorizer)
    CLASS_REGISTRY.add_klass(Char2VecVectorizer)

# Text preprocessors
if nltk is not None:
    CLASS_REGISTRY.add_klass(TextNltkProcessing)
CLASS_REGISTRY.add_klass(TextDefaultProcessing)
CLASS_REGISTRY.add_klass(TextDigitAnonymizer)

# Transformers
CLASS_REGISTRY.add_klass(TruncatedSVDWrapper)
CLASS_REGISTRY.add_klass(PCAWrapper)
CLASS_REGISTRY.add_klass(BoxCoxTargetTransformer)
CLASS_REGISTRY.add_klass(NumImputer)
CLASS_REGISTRY.add_klass(CdfScaler)
CLASS_REGISTRY.add_klass(KMeansTransformer)

# Category encoders
CLASS_REGISTRY.add_klass(NumericalEncoder)
CLASS_REGISTRY.add_klass(TargetEncoderClassifier)
CLASS_REGISTRY.add_klass(TargetEncoderEntropyClassifier)
CLASS_REGISTRY.add_klass(TargetEncoderRegressor)

# Classifiers
CLASS_REGISTRY.add_klass(RandomForestClassifier)
CLASS_REGISTRY.add_klass(ExtraTreesClassifier)
CLASS_REGISTRY.add_klass(LogisticRegression)
CLASS_REGISTRY.add_klass(Lasso)
if lightgbm is not None:
    CLASS_REGISTRY.add_klass(lightgbm.LGBMClassifier)

# Regressors
CLASS_REGISTRY.add_klass(RandomForestRegressor)
CLASS_REGISTRY.add_klass(ExtraTreesRegressor)
CLASS_REGISTRY.add_klass(Ridge)
if lightgbm is not None:
    CLASS_REGISTRY.add_klass(lightgbm.LGBMRegressor)

# Clustering
CLASS_REGISTRY.add_klass(KMeansWrapper)
CLASS_REGISTRY.add_klass(DBSCANWrapper)
CLASS_REGISTRY.add_klass(AgglomerativeClusteringWrapper)
