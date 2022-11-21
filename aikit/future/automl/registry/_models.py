from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from scipy.stats import randint as sp_randint, reciprocal

from aikit.models import KMeansWrapper, AgglomerativeClusteringWrapper, DBSCANWrapper
from .._hyper_parameters import HyperComposition, HyperRangeInt, HyperRangeBetaFloat, HyperChoice, \
    HyperLogRangeFloat, HyperCrossProduct
from ...enums import StepCategory, ProblemType
from ._base import ModelRepresentationBase
from .._registry import register

try:
    import lightgbm
except ImportError:
    lightgbm = None
    print("LightGBM not available, AutoML won't run with LightGBM models")


@register
class RidgeModel(ModelRepresentationBase):
    klass = Ridge
    category = StepCategory.Model
    type_of_variable = None
    type_of_model = ProblemType.REGRESSION
    use_y = True


@register
class LassoModel(ModelRepresentationBase):
    klass = Lasso
    category = StepCategory.Model
    type_of_variable = None
    type_of_model = ProblemType.REGRESSION
    use_y = True


@register
class LogisticRegressionModel(ModelRepresentationBase):
    klass = LogisticRegression
    category = StepCategory.Model
    type_of_variable = None
    type_of_model = ProblemType.CLASSIFICATION
    use_y = True


@register
class RandomForestClassifierModel(ModelRepresentationBase):
    klass = RandomForestClassifier
    category = StepCategory.Model
    type_of_variable = None
    custom_hyper = {"criterion": ("gini", "entropy")}
    type_of_model = ProblemType.CLASSIFICATION
    default_parameters = {"n_estimators": 100}
    use_y = True
    use_for_block_search = lightgbm is None  # use RandomForest only if LightGBM is not installed


@register
class RandomForestRegressorModel(ModelRepresentationBase):
    klass = RandomForestRegressor
    category = StepCategory.Model
    type_of_variable = None
    custom_hyper = {"criterion": ("mse", "mae")}
    type_of_model = ProblemType.REGRESSION
    default_parameters = {"n_estimators": 100}
    use_y = True
    use_for_block_search = lightgbm is None  # use RandomForest only if LightGBM is not installed


@register
class ExtraTreesClassifierModel(ModelRepresentationBase):
    klass = ExtraTreesClassifier
    category = StepCategory.Model
    type_of_variable = None
    custom_hyper = {"criterion": ("gini", "entropy")}
    type_of_model = ProblemType.CLASSIFICATION
    default_parameters = {"n_estimators": 100}
    use_y = True


@register
class ExtraTreesRegressorModel(ModelRepresentationBase):
    klass = ExtraTreesRegressor
    category = StepCategory.Model
    type_of_variable = None
    custom_hyper = {"criterion": ("mse", "mae")}
    type_of_model = ProblemType.REGRESSION
    default_parameters = {"n_estimators": 100}
    use_y = True


class LightGBMHyperParameter(object):
    """ special definition of hyperparameters of LightGBM models """

    @classmethod
    def get_hyper_parameter(cls):
        """ specific function to handle dependency between hyperparameters: bagging_fraction AND bagging_freq """
        res = HyperComposition(
            [
                # No bagging
                #   * bagging_freq == 0
                #   * bagging_fraction  == 1.0
                #   * no random forest here : 'booting_type' != 'rf'
                (
                    0.5,
                    HyperCrossProduct(
                        {
                            "boosting_type": ["gbdt", "dart"],
                            "learning_rate": HyperLogRangeFloat(0.0001, 0.1),
                            "max_depth": HyperChoice(
                                [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 50, 100]
                            ),
                            "n_estimators": HyperComposition(
                                [
                                    (0.50, HyperRangeInt(start=25, end=175, step=25)),
                                    (0.25, HyperRangeInt(start=200, end=900, step=100)),
                                    (0.25, HyperRangeInt(start=1000, end=10000, step=100)),
                                ]
                            ),
                            "colsample_bytree": HyperRangeBetaFloat(
                                start=0.1, end=1, alpha=3, beta=1
                            ),  # Mean = 0.75
                            "min_child_samples": HyperRangeInt(2, 50),
                            "num_leaves": HyperRangeInt(10, 200),
                            "bagging_fraction": [1.0],
                            "bagging_freq": [0],
                            "n_jobs": [1],
                        }
                    ),
                ),

                # Bagging
                #   * bagging_freq = 1
                #   * bagging_fraction < 1
                (
                    0.5,
                    HyperCrossProduct(
                        {
                            "boosting_type": ["rf", "gbdt", "dart"],
                            "learning_rate": HyperLogRangeFloat(0.0001, 0.1),
                            "max_depth": HyperChoice(
                                [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 50, 100]
                            ),
                            "n_estimators": HyperComposition(
                                [
                                    (0.50, HyperRangeInt(start=25, end=175, step=25)),
                                    (0.25, HyperRangeInt(start=200, end=900, step=100)),
                                    (0.25, HyperRangeInt(start=1000, end=10000, step=100)),
                                ]
                            ),
                            "colsample_bytree": HyperRangeBetaFloat(
                                start=0.1, end=1, alpha=3, beta=1
                            ),  # Mean = 0.75
                            "min_child_samples": HyperRangeInt(2, 50),
                            "num_leaves": HyperRangeInt(10, 200),
                            "bagging_fraction": HyperRangeBetaFloat(start=0.1, end=1, alpha=3, beta=1),
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
    class LGBMClassifierModel(LightGBMHyperParameter, ModelRepresentationBase):
        klass = lightgbm.LGBMClassifier
        category = StepCategory.Model
        type_of_variable = None
        type_of_model = ProblemType.CLASSIFICATION
        use_y = True
        use_for_block_search = True

    @register
    class LGBMRegressorModel(LightGBMHyperParameter, ModelRepresentationBase):
        klass = lightgbm.LGBMRegressor
        category = StepCategory.Model
        type_of_variable = None
        type_of_model = ProblemType.REGRESSION
        use_y = True
        use_for_block_search = True


@register
class KMeansModel(ModelRepresentationBase):
    klass = KMeansWrapper
    category = StepCategory.Model
    type_of_variable = None  # TypeOfVariables.NUM
    type_of_model = ProblemType.CLUSTERING
    custom_hyper = {"n_clusters": sp_randint(2, 20)}  # TODO: seed ?
    use_y = False


@register
class AgglomerativeClusteringModel(ModelRepresentationBase):
    klass = AgglomerativeClusteringWrapper
    category = StepCategory.Model
    type_of_variable = None  # TypeOfVariables.NUM
    type_of_model = ProblemType.CLUSTERING
    custom_hyper = {"n_clusters": sp_randint(2, 20)}  # TODO: seed ?
    use_y = False


@register
class DBSCANModel(ModelRepresentationBase):
    klass = DBSCANWrapper
    category = StepCategory.Model
    type_of_variable = None  # TypeOfVariables.NUM
    type_of_model = ProblemType.CLUSTERING
    custom_hyper = {
        "eps": reciprocal(1e-5, 1),  # TODO: seed ?
        "metric": ["minkowski"],
        "leaf_size": sp_randint(10, 100),  # TODO: seed ?
        "min_samples": sp_randint(1, 100),  # TODO: seed ?
        "p": sp_randint(1, 20),  # TODO: seed ?
        "scale_eps": [True],
    }
    use_y = False
