from scipy.stats import randint as sp_randint
from sklearn.linear_model import LinearRegression

from aikit.transformers import NumericalEncoder, TargetEncoderClassifier, TargetEncoderRegressor, NumImputer, \
    TruncatedSVDWrapper, PCAWrapper, KMeansTransformer, CdfScaler, BoxCoxTargetTransformer
from ._base import ModelRepresentationBase
from ..hyper_parameters import HyperComposition, HyperRangeInt, HyperLogRangeFloat, HyperRangeFloat
from .._registry import register
from ...enums import StepCategory, ProblemType, VariableType


@register
class NumericalEncoderCatEncoder(ModelRepresentationBase):
    klass = NumericalEncoder
    category = StepCategory.CategoryEncoder
    type_of_variable = (VariableType.CAT,)
    custom_hyper = {
        "encoding_type": ["dummy", "num"],
        "min_nb_observations": HyperRangeInt(2, 20)
    }
    type_of_model = None
    use_y = False
    use_for_block_search = True


@register
class TargetEncoderClassifierCatEncoder(ModelRepresentationBase):
    klass = TargetEncoderClassifier
    category = StepCategory.CategoryEncoder
    type_of_variable = (VariableType.CAT,)
    custom_hyper = {
        "cv": [None, 2, 5, 10],
        "noise_level": HyperComposition([(0.5, [None]), (0.5, HyperRangeFloat(0, 1))]),
        "smoothing_min": HyperRangeFloat(0, 10),
        "smoothing_value": HyperRangeFloat(0, 10),
    }
    type_of_model = ProblemType.CLASSIFICATION
    use_y = True


@register
class TargetEncoderRegressorCatEncoder(ModelRepresentationBase):
    klass = TargetEncoderRegressor
    category = StepCategory.CategoryEncoder
    type_of_variable = (VariableType.CAT, VariableType.NUM)
    custom_hyper = {
        "cv": [None, 2, 5, 10],
        "noise_level": HyperComposition([(0.5, [None]), (0.5, HyperRangeFloat(0, 1))]),
        "smoothing_min": HyperRangeFloat(0, 10),
        "smoothing_value": HyperRangeFloat(0, 10),
    }
    type_of_model = ProblemType.REGRESSION
    use_y = True


@register
class NumImputerImputer(ModelRepresentationBase):
    klass = NumImputer
    category = StepCategory.MissingValueImputer
    type_of_variable = None
    type_of_model = None
    use_y = False
    use_for_block_search = True


@register
class TruncatedSVDDimensionReduction(ModelRepresentationBase):
    klass = TruncatedSVDWrapper
    category = StepCategory.DimensionReduction
    type_of_variable = None
    type_of_model = None
    use_y = False
    custom_hyper = {"drop_used_columns": [True, False]}


@register
class PCAModel(ModelRepresentationBase):
    klass = PCAWrapper
    category = StepCategory.DimensionReduction
    type_of_variable = None
    use_y = False
    custom_hyper = {"n_components": sp_randint(2, 30)}  # TODO: seed ?


@register
class TextTruncatedSVDDimensionReduction(ModelRepresentationBase):
    klass = TruncatedSVDWrapper
    category = StepCategory.TextDimensionReduction
    custom_hyper = {
        "n_components": HyperRangeInt(10, 500, step=5),
        "drop_used_columns": [True, False],
        "column_prefix": ["textSVD"]  # so that SVD column don't have same names
    }
    type_of_variable = VariableType.TEXT
    type_of_model = None
    use_y = False


@register
class KMeansTransformerDimensionReduction(ModelRepresentationBase):
    klass = KMeansTransformer
    category = StepCategory.DimensionReduction
    custom_hyper = {
        "result_type": ("probability", "distance", "inv_distance", "log_distance", "cluster"),
        "temperature": HyperLogRangeFloat(start=0.01, end=2, n=100),
        "drop_used_columns": [True, False]
    }
    type_of_model = None
    use_y = False
    type_of_variable = None


@register
class CdfScalerScaler(ModelRepresentationBase):
    klass = CdfScaler
    category = StepCategory.Scaling
    custom_hyper = {
        "distribution": ("auto-nonparam", "auto-param"),
        "output_distribution": ("normal", "uniform")
    }
    use_y = False
    type_of_model = None
    type_of_variable = None


@register
class BoxCoxTargetTransformerTargetModifier(ModelRepresentationBase):
    klass = BoxCoxTargetTransformer
    category = StepCategory.TargetTransformer
    type_of_variable = None
    type_of_model = ProblemType.REGRESSION
    custom_hyper = {
        "ll": HyperComposition([(0.1, [0]), (0.9, HyperRangeFloat(0, 2))]),
        "model": (LinearRegression(),)
    }
    use_y = True
