from collections import OrderedDict


def filter_dict_on_key_value_pairs(d, func):
    """ filter a dictionary according to a function on its keys and values """
    if isinstance(d, OrderedDict):
        return OrderedDict([(k, v) for k, v in d.items() if func(k, v)])
    elif isinstance(d, dict):
        return {k: v for k, v in d.items() if func(k, v)}
    else:
        res = dict.__class__()
        for k, v in res.items():
            if func(k, v):
                res[k] = v
        return res


def filter_dict_on_keys(d, func):
    """ filter a dictionary according to a function on its keys
    keep original type of dict
    """
    if isinstance(d, OrderedDict):
        return OrderedDict([(k, v) for k, v in d.items() if func(k)])
    elif isinstance(d, dict):
        return {k: v for k, v in d.items() if func(k)}
    else:
        res = d.__class__()
        for k, v in res.items():
            if func(k):
                res[k] = v
        return res


def filter_dict_on_values(d, func):
    """ filter a dictionary according to a function on its values """
    if isinstance(d, OrderedDict):
        return OrderedDict([(k, v) for k, v in d.items() if func(v)])
    elif isinstance(d, dict):
        return {k: v for k, v in d.items() if func(v)}
    else:
        res = d.__class__()
        for k, v in res.items():
            if func(v):
                res[k] = v
        return res


def map_dict_keys(d, func):
    """ apply a function on the key of a dictionary """
    if isinstance(d, OrderedDict):
        return OrderedDict([(func(k), v) for k, v in d.items()])
    elif isinstance(d, dict):
        return {func(k): v for k, v in d.items()}
    else:
        res = d.__class__()
        for k, v in res.items():
            res[func(k)] = v
        return res


def map_dict_values(d, func):
    """ apply a function on the values of a dictionary """
    if isinstance(d, OrderedDict):
        return OrderedDict([(k, func(v)) for k, v in d.items()])
    elif isinstance(d, dict):
        return {k: func(v) for k, v in d.items()}
    else:
        res = d.__class__()
        for k, v in res.items():
            res[k] = func(v)
        return res
