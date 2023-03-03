from aikit.transformers import FeaturesSelectorRegressor, FeaturesSelectorClassifier
from ._base import ModelRepresentationBase
from .._registry import register
from ...enums import StepCategory, ProblemType


@register
class RegressorFeaturesSelectorSelection(ModelRepresentationBase):
    klass = FeaturesSelectorRegressor
    category = StepCategory.FeatureSelection
    type_of_variable = None
    custom_hyper = {"selector_type": ("default", "forest", "linear")}
    type_of_model = ProblemType.REGRESSION
    use_y = True


@register
class ClassifierFeaturesSelectorSelection(ModelRepresentationBase):
    klass = FeaturesSelectorClassifier
    category = StepCategory.FeatureSelection
    type_of_variable = None
    custom_hyper = {"selector_type": ("default", "forest", "linear")}
    type_of_model = ProblemType.CLASSIFICATION
    use_y = True
