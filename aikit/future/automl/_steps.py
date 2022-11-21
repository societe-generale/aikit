from ._registry import MODEL_REGISTRY
from ..enums import ProblemType, VariableType, StepCategory


def get_needed_steps(df_information, type_of_problem):
    """ Returns the needed steps 'apriori' needed for a given problem. """

    # Careful : order matters + no_check for now
    is_regression = type_of_problem == ProblemType.REGRESSION

    has_text = any((VariableType.TEXT in v["TypeOfVariable"] for v in df_information.values()))
    has_category = any((VariableType.CAT in v["TypeOfVariable"] for v in df_information.values()))

    has_missing_value = any((v["HasMissing"] for v in df_information.values()))

    needed_steps = []

    if is_regression:
        needed_steps.append({
            "step": StepCategory.TargetTransformer,
            "optional": True
        })

    if has_text:
        needed_steps.append({
            "step": StepCategory.TextPreprocessing,
            "optional": True
        })
        needed_steps.append({
            "step": StepCategory.TextEncoder,
            "optional": False
        })
        needed_steps.append({
            "step": StepCategory.TextDimensionReduction,
            "optional": True
        })

    if has_category:
        needed_steps.append({
            "step": StepCategory.CategoryEncoder,
            "optional": False
        })

    if has_missing_value:
        needed_steps.append({
            "step": StepCategory.MissingValueImputer,
            "optional": False
        })

    needed_steps.append({
        "step": StepCategory.Scaling,
        "optional": True
    })
    needed_steps.append({
        "step": StepCategory.DimensionReduction,
        "optional": True
    })
    needed_steps.append({
        "step": StepCategory.FeatureExtraction,
        "optional": True
    })
    needed_steps.append({
        "step": StepCategory.FeatureSelection,
        "optional": True
    })

    needed_steps.append({
        "step": StepCategory.Model,
        "optional": False
    })

    return needed_steps


def filter_model_to_keep(problem_type, block_search_only=False):
    """ Filter the list of model that we want to keep
    based on the problem type. """
    # Here: we could potentially also remove models that are not possible
    # on a given machine (not installed, no enough resources, ...)
    models_to_keep = []
    for c, v in MODEL_REGISTRY.informations.items():

        if block_search_only:
            use_for_block_search = v.get("use_for_block_search", False)
            if not use_for_block_search:
                # skip that models
                continue

        model_type_of_problem = v.get("type_of_model", None)

        if model_type_of_problem is None:
            # The model 'v' is registered with type of model = None ==> I'll include it
            models_to_keep.append(c)

        elif isinstance(model_type_of_problem, (tuple, list, set)):
            # The model has a list of type of problem ==> I'll include it if the current type is among those
            if problem_type in model_type_of_problem:
                models_to_keep.append(c)

        else:
            # The model has only one type of problem ==> I'll include it if the current type is that
            if problem_type == model_type_of_problem:
                models_to_keep.append(c)

    return models_to_keep
