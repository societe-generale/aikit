# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:54:14 2018

@author: Lionel Massoulard


to register a new model you need to :

 * create a class with register decorator
 * the name isn't important (and the class will never be used). It is just a way to easily register things

The class should have :

 * klass : Model class
 * category : something among StepCategories, overall category of the transfomer/model

 * type_of_variable : type of variable it is applyied on

 * type_of_model : type of model

 * custom_hyper : CAN be used to specify hyper-parameters
 * get_hyper_parameter classmethod can also be overrided to specify behavior

 * any other class atribut will be saved in "informations"


"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso

try:
    import lightgbm
except ImportError:
    lightgbm = None
    print("I wont be able to run AutoML on LighGBM models, please import lightgbm")

try:
    import nltk
except ImportError:
    nltk = None
    print("I wont be able to run AutoML with NLTK transformers, please install nltk")

try:
    import gensim
except ImportError:
    gensim = None
    print("I wont be able to run AutoML with Word2Vec transformer, please install gensim")


from scipy.stats import reciprocal
from scipy.stats import randint as sp_randint

from aikit.enums import StepCategories, TypeOfVariables, TypeOfProblem

from aikit.transformers import FeaturesSelectorClassifier, FeaturesSelectorRegressor, TruncatedSVDWrapper, CdfScaler
from aikit.transformers import PCAWrapper
from aikit.transformers import CountVectorizerWrapper, Word2VecVectorizer, Char2VecVectorizer
from aikit.transformers import TextNltkProcessing, TextDefaultProcessing, TextDigitAnonymizer

from aikit.transformers import BoxCoxTargetTransformer, NumImputer, KMeansTransformer
from aikit.transformers import TargetEncoderClassifier, TargetEncoderRegressor
from aikit.transformers import NumericalEncoder

from aikit.models import DBSCANWrapper, KMeansWrapper, AgglomerativeClusteringWrapper

# from aikit.simple_model_registration import DICO_NAME_KLASS
from aikit.ml_machine.model_registrer import _AbstractModelRepresentation, register, MODEL_REGISTER

assert MODEL_REGISTER  # To shutup python 'not used warning'

from aikit.ml_machine import hyper_parameters as hp


# In[] : Models
MODEL_REGISTER.reset()


class ModelRepresentationBase(_AbstractModelRepresentation):
    """ class just to store the default HyperParameters """

    default_hyper = {
        "n_components": hp.HyperRangeFloat(start=0.1, end=1, step=0.05),
        # Forest like estimators
        "n_estimators": hp.HyperComposition(
            [
                (0.75, hp.HyperRangeInt(start=25, end=175, step=25)),
                (0.25, hp.HyperRangeInt(start=200, end=1000, step=100)),
            ]
        ),
        "max_features": hp.HyperComposition(
            [(0.25, ["sqrt", "auto"]), (0.75, hp.HyperRangeBetaFloat(start=0, end=1, alpha=3, beta=1))]
        ),
        "max_depth": hp.HyperChoice([None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 50, 100]),
        "min_samples_split": hp.HyperRangeBetaInt(start=2, end=100, alpha=1, beta=5),
        # Linear model
        "C": hp.HyperLogRangeFloat(start=0.00001, end=10, n=50),
        "alpha": hp.HyperLogRangeFloat(start=0.00001, end=10, n=50),
        # CV
        "analyzer": hp.HyperChoice(["word", "char", "char_wb"]),
        "penalty": ["l1", "l2"],
        "random_state": [123],  # So that every for every model with a random_state attribute, it will be passed and fix
        
        "drop_used_columns":[True],
        "drop_unused_columns":[True]
    }
    # This dictionnary is used to specify the default hyper-parameters that are used during the random search phase
    # They will be used if :
    # * the model has a paramters among that list
    # * the parameters is not specified within the class (within 'custom_hyper')
    
    
    
    default_default_hyper = {
            "random_state":123,
            "drop_used_columns":True,
            "drop_unused_columns":True
            }
    # This dictionnary is used to specify the default hyper-parameters that are used during the default model phase
    # They will be used if :
    # * the model has a paramters among that list
    # * the default parameters is not specified within the class (withing 'default_parameters')


### Linear
@register
class Ridge_Model(ModelRepresentationBase):
    klass = Ridge

    category = StepCategories.Model
    type_of_variable = None

    # is_regression = True

    type_of_model = TypeOfProblem.REGRESSION

    use_y = True


@register
class Lasso_Model(ModelRepresentationBase):
    klass = Lasso

    category = StepCategories.Model
    type_of_variable = None

    # is_regression = True

    type_of_model = TypeOfProblem.REGRESSION

    use_y = True


@register
class LogisticRegression_Model(ModelRepresentationBase):
    klass = LogisticRegression

    category = StepCategories.Model
    type_of_variable = None

    # is_regression = False

    type_of_model = TypeOfProblem.CLASSIFICATION

    use_y = True


### Random Forest
@register
class RandomForestClassifier_Model(ModelRepresentationBase):

    klass = RandomForestClassifier
    category = StepCategories.Model
    type_of_variable = None

    custom_hyper = {"criterion": ("gini", "entropy")}

    # is_regression = False

    type_of_model = TypeOfProblem.CLASSIFICATION

    default_parameters = {"n_estimators": 100}

    use_y = True

    use_for_block_search = lightgbm is None  # use RandomForest only if LightGBM is not installed


@register
class RandomForestRegressor_Model(ModelRepresentationBase):
    klass = RandomForestRegressor
    category = StepCategories.Model
    type_of_variable = None

    custom_hyper = {"criterion": ("mse", "mae")}

    # is_regression = True

    type_of_model = TypeOfProblem.REGRESSION

    default_parameters = {"n_estimators": 100}

    use_y = True

    use_for_block_search = lightgbm is None  # use RandomForest only if LightGBM is not installed


### Extra Trees
@register
class ExtraTreesClassifier_Model(ModelRepresentationBase):

    klass = ExtraTreesClassifier
    category = StepCategories.Model
    type_of_variable = None

    custom_hyper = {"criterion": ("gini", "entropy")}

    # is_regression = False

    type_of_model = TypeOfProblem.CLASSIFICATION

    default_parameters = {"n_estimators": 100}

    use_y = True


@register
class ExtraTreesRegressor_Model(ModelRepresentationBase):
    klass = ExtraTreesRegressor
    category = StepCategories.Model
    type_of_variable = None

    custom_hyper = {"criterion": ("mse", "mae")}

    # is_regression = True

    type_of_model = TypeOfProblem.REGRESSION

    default_parameters = {"n_estimators": 100}

    use_y = True


### LGBM


class LGBM_HyperParameter(object):
    """ special definition of hyper-parameters of LGBMModel(s) """

    @classmethod
    def get_hyper_parameter(cls):
        """ specific function to handle dependency between hyper-parameters : bagging_fraction AND bagging_freq """
        res = hp.HyperComposition(
            [
                ##################
                ### No Bagging ###
                ##################
                # * bagging_freq == 0
                # * bagging_fraction  == 1.0
                # * no random forest here : 'booting_type' != 'rf'
                (
                    0.5,
                    hp.HyperCrossProduct(
                        {
                            "boosting_type": ["gbdt", "dart"],
                            "learning_rate": hp.HyperLogRangeFloat(0.0001, 0.1),
                            "max_depth": hp.HyperChoice(
                                [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 50, 100]
                            ),
                            "n_estimators": hp.HyperComposition(
                                [
                                    (0.50, hp.HyperRangeInt(start=25, end=175, step=25)),
                                    (0.25, hp.HyperRangeInt(start=200, end=900, step=100)),
                                    (0.25, hp.HyperRangeInt(start=1000, end=10000, step=100)),
                                ]
                            ),
                            "colsample_bytree": hp.HyperRangeBetaFloat(
                                start=0.1, end=1, alpha=3, beta=1
                            ),  # Mean = 0.75
                            "min_child_samples": hp.HyperRangeInt(2, 50),
                            "num_leaves": hp.HyperRangeInt(10, 200),
                            "bagging_fraction": [1.0],
                            "bagging_freq": [0],
                            "n_jobs": [1],
                        }
                    ),
                ),
                ###############
                ### Bagging ###
                ###############
                # * bagging_freq = 1
                # * bagging_fraction < 1
                (
                    0.5,
                    hp.HyperCrossProduct(
                        {
                            "boosting_type": ["rf", "gbdt", "dart"],
                            "learning_rate": hp.HyperLogRangeFloat(0.0001, 0.1),
                            "max_depth": hp.HyperChoice(
                                [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 50, 100]
                            ),
                            "n_estimators": hp.HyperComposition(
                                [
                                    (0.50, hp.HyperRangeInt(start=25, end=175, step=25)),
                                    (0.25, hp.HyperRangeInt(start=200, end=900, step=100)),
                                    (0.25, hp.HyperRangeInt(start=1000, end=10000, step=100)),
                                ]
                            ),
                            "colsample_bytree": hp.HyperRangeBetaFloat(
                                start=0.1, end=1, alpha=3, beta=1
                            ),  # Mean = 0.75
                            "min_child_samples": hp.HyperRangeInt(2, 50),
                            "num_leaves": hp.HyperRangeInt(10, 200),
                            "bagging_fraction": hp.HyperRangeBetaFloat(start=0.1, end=1, alpha=3, beta=1),
                            "bagging_freq": [1],
                            "n_jobs": [1],
                        }
                    ),
                ),
            ]
        )
        return res


if lightgbm is not None:

    @register
    class LGBMClassifier_Model(LGBM_HyperParameter, ModelRepresentationBase):
        klass = lightgbm.LGBMClassifier
        category = StepCategories.Model
        type_of_variable = None

        # is_regression = False

        type_of_model = TypeOfProblem.CLASSIFICATION

        use_y = True

        use_for_block_search = True

    @register
    class LGBMRegressor_Model(LGBM_HyperParameter, ModelRepresentationBase):
        klass = lightgbm.LGBMRegressor
        category = StepCategories.Model
        type_of_variable = None

        # is_regression = True

        type_of_model = TypeOfProblem.REGRESSION

        use_y = True

        use_for_block_search = True


# In[] : Selectors


@register
class RegressorFeaturesSelector_Selection(ModelRepresentationBase):
    klass = FeaturesSelectorRegressor

    category = StepCategories.FeatureSelection
    type_of_variable = None

    custom_hyper = {"selector_type": ("default", "forest", "linear")}
    # is_regression = True

    type_of_model = TypeOfProblem.REGRESSION

    use_y = True


@register
class ClassifierFeaturesSelector_Selection(ModelRepresentationBase):
    klass = FeaturesSelectorClassifier

    category = StepCategories.FeatureSelection

    type_of_variable = None

    custom_hyper = {"selector_type": ("default", "forest", "linear")}

    # is_regression = False

    type_of_model = TypeOfProblem.CLASSIFICATION

    use_y = True


# In[] : Text Encoder
@register
class CountVectorizer_TextEncoder(ModelRepresentationBase):
    klass = CountVectorizerWrapper
    category = StepCategories.TextEncoder
    type_of_variable = TypeOfVariables.TEXT

    use_y = False  # It means in 'approx cv mode' CV wont be done

    @classmethod
    def get_hyper_parameter(cls):
        ### Specific function to handle the fact that I don't want ngram != 1 IF analyzer = word ###
        res = hp.HyperComposition(
            [
                (
                    0.5,
                    hp.HyperCrossProduct(
                        {
                            "ngram_range": 1,
                            "analyzer": "word",
                            "min_df": [1, 0.001, 0.01, 0.05],
                            "max_df": [0.999, 0.99, 0.95],
                            "tfidf": [True, False],
                        }
                    ),
                ),
                (
                    0.5,
                    hp.HyperCrossProduct(
                        {
                            "ngram_range": hp.HyperRangeBetaInt(
                                start=1, end=5, alpha=2, beta=1
                            ),  # 1 = 1.5% ; 2 = 12% ; 3 = 25% ; 4 = 37% ; 5 = 24%
                            "analyzer": hp.HyperChoice(("char", "char_wb")),
                            "min_df": [1, 0.001, 0.01, 0.05],
                            "max_df": [0.999, 0.99, 0.95],
                            "tfidf": [True, False],
                        }
                    ),
                ),
            ]
        )

        return res

    use_for_block_search = True


if gensim is not None:

    @register
    class Word2VecVectorizer_TextEncoder(ModelRepresentationBase):

        klass = Word2VecVectorizer
        category = StepCategories.TextEncoder
        type_of_variable = TypeOfVariables.TEXT

        custom_hyper = {
            "size": hp.HyperRangeInt(50, 300, step=10),
            "window": [3, 5, 7],
            "same_embedding_all_columns": [True, False],
            "text_preprocess": [None, "default", "digit", "nltk"],
        }

        type_of_model = None
        # is_regression = None

        use_y = False


if gensim is not None:

    @register
    class Char2VecVectorizer_TextEncoder(ModelRepresentationBase):

        klass = Char2VecVectorizer
        category = StepCategories.TextEncoder
        type_of_variable = TypeOfVariables.TEXT

        custom_hyper = {
            "size": hp.HyperRangeInt(50, 300, step=10),
            "window": [3, 5, 7],
            "ngram": hp.HyperRangeInt(2, 6),
            "same_embedding_all_columns": [True, False],
            "text_preprocess": [None, "default", "digit", "nltk"],
        }

        type_of_model = None

        use_y = False


if nltk is not None:

    @register
    class TextNltkProcessing_TextPreprocessor(ModelRepresentationBase):
        klass = TextNltkProcessing
        category = StepCategories.TextPreprocessing
        type_of_variable = TypeOfVariables.TEXT

        type_of_model = None

        use_y = False


@register
class TextNltkProcessing_DefaultPreprocessor(ModelRepresentationBase):
    klass = TextDefaultProcessing
    category = StepCategories.TextPreprocessing
    type_of_variable = TypeOfVariables.TEXT

    type_of_model = None

    use_y = False


@register
class TextNltkProcessing_DigitAnonymizer(ModelRepresentationBase):
    klass = TextDigitAnonymizer
    category = StepCategories.TextPreprocessing
    type_of_variable = TypeOfVariables.TEXT

    type_of_model = None

    use_y = False


# In[] : Category Encoder


@register
class NumericalEncoder_CatEncoder(ModelRepresentationBase):
    klass = NumericalEncoder
    category = StepCategories.CategoryEncoder

    type_of_variable = (TypeOfVariables.CAT, )

    custom_hyper = {"encoding_type": ["dummy", "num"], "min_nb_observations": hp.HyperRangeInt(2, 20)}

    type_of_model = None

    use_y = False

    use_for_block_search = True


@register
class TargetEncoderClassifier_CatEncoder(ModelRepresentationBase):

    klass = TargetEncoderClassifier
    category = StepCategories.CategoryEncoder

    type_of_variable = (TypeOfVariables.CAT, )

    custom_hyper = {
        "cv": [None, 2, 5, 10],
        "noise_level": hp.HyperComposition([(0.5, [None]), (0.5, hp.HyperRangeFloat(0, 1))]),
        "smoothing_min": hp.HyperRangeFloat(0, 10),
        "smoothing_value": hp.HyperRangeFloat(0, 10),
    }

    # is_regression = False

    type_of_model = TypeOfProblem.CLASSIFICATION

    use_y = True


@register
class TargetEncoderRegressor_CatEncoder(ModelRepresentationBase):

    klass = TargetEncoderRegressor
    category = StepCategories.CategoryEncoder

    type_of_variable = (TypeOfVariables.CAT, TypeOfVariables.NUM)

    custom_hyper = {
        "cv": [None, 2, 5, 10],
        "noise_level": hp.HyperComposition([(0.5, [None]), (0.5, hp.HyperRangeFloat(0, 1))]),
        "smoothing_min": hp.HyperRangeFloat(0, 10),
        "smoothing_value": hp.HyperRangeFloat(0, 10),
    }

    # is_regression = True

    type_of_model = TypeOfProblem.REGRESSION

    use_y = True


# In[] : Imputer
@register
class NumImputer_Inputer(ModelRepresentationBase):
    klass = NumImputer
    category = StepCategories.MissingValueImputer
    type_of_variable = None

    type_of_model = None
    use_y = False

    use_for_block_search = True


# In[]
@register
class TruncatedSVD_DimensionReduction(ModelRepresentationBase):
    klass = TruncatedSVDWrapper
    category = StepCategories.DimensionReduction

    type_of_variable = None

    type_of_model = None
    use_y = False

    custom_hyper = {"drop_used_columns": [True, False]}


@register
class PCA_Model(ModelRepresentationBase):
    klass = PCAWrapper
    category = StepCategories.DimensionReduction

    type_of_variable = None

    use_y = False

    custom_hyper = {"n_components": sp_randint(2, 30)}

    # @classmethod
    # def get_hyper_parameter(cls):
    #     return hp.HyperRandomVariable(cls.custom_hyper)


@register
class Text_TruncatedSVD_DimensionReduction(ModelRepresentationBase):
    klass = TruncatedSVDWrapper
    category = StepCategories.TextDimensionReduction

    custom_hyper = {"n_components": hp.HyperRangeInt(10, 500, step=5)}

    type_of_variable = TypeOfVariables.TEXT

    type_of_model = None
    use_y = False

    custom_hyper = {"drop_used_columns": [True, False]}


@register
class KMeansTransformer_DimensionReduction(ModelRepresentationBase):
    klass = KMeansTransformer
    category = StepCategories.DimensionReduction

    custom_hyper = {
        "result_type": ("probability", "distance", "inv_distance", "log_distance", "cluster"),
        "temperature": hp.HyperLogRangeFloat(start=0.01, end=2, n=100),
    }
    type_of_model = None
    use_y = False
    type_of_variable = None

    custom_hyper = {"drop_used_columns": [True, False]}


@register
class CdfScaler_Scaler(ModelRepresentationBase):
    klass = CdfScaler
    category = StepCategories.Scaling

    custom_hyper = {"distribution": ("auto-nonparam", "auto-param"), "output_distribution": ("normal", "uniform")}

    use_y = False
    type_of_model = None

    type_of_variable = None


# In[] : target Modifier
@register
class BoxCoxTargetTransformer_TargetModifier(ModelRepresentationBase):

    klass = BoxCoxTargetTransformer
    category = StepCategories.TargetTransformer

    type_of_variable = None

    # is_regression = True

    type_of_model = TypeOfProblem.REGRESSION

    custom_hyper = {"ll": hp.HyperComposition([(0.1, [0]), (0.9, hp.HyperRangeFloat(0, 2))])}

    use_y = True


# In[] : clusterer


@register
class Kmeans_Model(ModelRepresentationBase):

    klass = KMeansWrapper
    category = StepCategories.Model

    type_of_variable = None  # TypeOfVariables.NUM

    # is_regression = False

    type_of_model = TypeOfProblem.CLUSTERING

    custom_hyper = {"n_clusters": sp_randint(2, 20)}

    use_y = False

    # @classmethod
    # def get_hyper_parameter(cls):
    #     return hp.HyperRandomVariable(cls.custom_hyper)


@register
class AgglomerativeClustering_Model(ModelRepresentationBase):

    klass = AgglomerativeClusteringWrapper
    category = StepCategories.Model

    type_of_variable = None  # TypeOfVariables.NUM

    # is_regression = False

    type_of_model = TypeOfProblem.CLUSTERING

    custom_hyper = {"n_clusters": sp_randint(2, 20)}

    use_y = False

    # @classmethod
    # def get_hyper_parameter(cls):
    #     return hp.HyperRandomVariable(cls.custom_hyper)


@register
class DBSCAN_Model(ModelRepresentationBase):

    klass = DBSCANWrapper
    category = StepCategories.Model

    type_of_variable = None  # TypeOfVariables.NUM

    # is_regression = False

    type_of_model = TypeOfProblem.CLUSTERING

    custom_hyper = {
        "eps": reciprocal(1e-5, 1),
        "metric": ["minkowski"],
        "leaf_size": sp_randint(10, 100),
        "min_samples": sp_randint(1, 100),
        "p": sp_randint(1, 20),
        "scale_eps": [True],
    }

    use_y = False

    # @classmethod
    # def get_hyper_parameter(cls):
    #     return hp.HyperRandomVariable(cls.custom_hyper)
