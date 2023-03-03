import copy

from ._class_registry import CLASS_REGISTRY
from ..enums import SpecialModels


def _is_model(param):
    """ does the param represent a simple model """
    if not isinstance(param, (tuple, list)):
        return False, None

    if len(param) == 0:
        return False, None

    try:
        model_klass = CLASS_REGISTRY.get(param[0], None)
    except TypeError:
        # Can arise if I try to hash something that isn't hashable
        model_klass = None

    if model_klass is not None:
        return True, model_klass

    return False, None


def sklearn_model_from_param(param, _copy=True):
    """ Converts a parameter into a sklearn model.
    The syntax is that a model is represented by a 2-uple with its name and its arguments.

    Example
    -------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> sklearn_model_from_param(("RandomForestClassifier", {"n_estimators": 100}))
    >>> RandomForestClassifier(n_estimators=100)
    """

    if _copy:
        param = copy.deepcopy(param)  # Internal copy

    model_node, model_klass = _is_model(param)

    if model_node and param[0] == SpecialModels.GraphPipeline:
        # GraphPipeline node
        rest_param = param[1:]
        list_args = []

        for i, arg in enumerate(rest_param[:-1]):
            if i == 0:
                list_args.append(sklearn_model_from_param(arg, _copy=False))
            else:
                # Second argument is edges => no need for translation
                list_args.append(arg)

        # If last attribute is a dict, it is named arguments
        if isinstance(rest_param[-1], dict):
            dict_args = rest_param[-1]
            for k, v in dict_args.items():
                if k != "edges":
                    dict_args[k] = sklearn_model_from_param(v, _copy=False)
        else:
            # Otherwise: just a regular param
            dict_args = {}
            if len(rest_param) == 1:
                list_args.append(sklearn_model_from_param(rest_param[-1], _copy=False))
            else:
                list_args.append(rest_param[-1])

        return model_klass(*list_args, **dict_args)

    elif model_node and param[0] != SpecialModels.GraphPipeline:
        # Classical model node
        rest_param = param[1:]

        # If last attribute is a dict, it is named arguments
        if isinstance(rest_param[-1], dict):
            list_args = list(rest_param[:-1])
            dict_args = rest_param[-1]
        else:
            list_args = list(rest_param)
            dict_args = {}

        return model_klass(
            *sklearn_model_from_param(list_args, _copy=False),
            **sklearn_model_from_param(dict_args, _copy=False))

    elif isinstance(param, dict):
        # Dictionary
        res = param.__class__()
        for k, v in param.items():
            res[k] = sklearn_model_from_param(v, _copy=False)
        return res

    elif isinstance(param, list):
        # List
        return [sklearn_model_from_param(v, _copy=False) for v in param]

    elif isinstance(param, tuple):
        # Tuple
        return tuple([sklearn_model_from_param(v, _copy=False) for v in param])

    else:
        return param
