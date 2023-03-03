from .._registry import _AbstractModelRepresentation
from ..hyper_parameters import HyperRangeFloat, HyperComposition, HyperRangeInt, HyperRangeBetaFloat, HyperChoice, \
    HyperRangeBetaInt, HyperLogRangeFloat


class ModelRepresentationBase(_AbstractModelRepresentation):
    """ Store default hyperparameters """

    # This dictionary is used to specify the default hyperparameters that are used during the random search phase
    # They will be used if :
    #   * the model has a parameters among that list
    #   * the parameters is not specified within the class (within 'custom_hyper')
    default_hyper = {
        "n_components": HyperRangeFloat(start=0.1, end=1, step=0.05),
        # Forest like estimators
        "n_estimators": HyperComposition(
            [
                (0.75, HyperRangeInt(start=25, end=175, step=25)),
                (0.25, HyperRangeInt(start=200, end=1000, step=100)),
            ]
        ),
        "max_features": HyperComposition(
            [(0.25, ["sqrt", "auto"]), (0.75, HyperRangeBetaFloat(start=0, end=1, alpha=3, beta=1))]
        ),
        "max_depth": HyperChoice([None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 50, 100]),
        "min_samples_split": HyperRangeBetaInt(start=2, end=100, alpha=1, beta=5),
        # Linear model
        "C": HyperLogRangeFloat(start=0.00001, end=10, n=50),
        "alpha": HyperLogRangeFloat(start=0.00001, end=10, n=50),
        # CV
        "analyzer": HyperChoice(["word", "char", "char_wb"]),
        "penalty": ["l1", "l2"],
        # So that every for every model with a random_state attribute
        "random_state": [123],
        # Default behavior for built-in column selectors
        "drop_used_columns": [True],
        "drop_unused_columns": [True]
    }

    # This dictionary is used to specify the default hyperparameters that are used during the default model phase
    # They will be used if :
    #   * the model has a parameters among that list
    #   * the default parameters is not specified within the class (withing 'default_parameters')
    default_default_hyper = {
        "random_state": 123,
        "drop_used_columns": True,
        "drop_unused_columns": True
    }

    # Model dependencies
    depends_on = ()
