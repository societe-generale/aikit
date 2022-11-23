from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target

from aikit.future.enums import VariableType, ProblemType


def guess_problem_type(X, y=None):  # noqa
    """ try to guess the type of problem """

    if y is None:
        return ProblemType.CLUSTERING

    if len(np.unique(y)) == 1:
        raise ValueError("Unique values in y")

    if type_of_target(y) in ["binary", "multiclass"]:
        if "float" in str(y.dtype) or "int" in str(y.dtype):
            nb_u = len(np.unique(y))
            nb = len(y)
            if nb_u >= 0.25 * nb:
                return ProblemType.REGRESSION
            else:
                return ProblemType.CLASSIFICATION
        return ProblemType.CLASSIFICATION
    else:
        return ProblemType.REGRESSION


def guess_type_of_variable(s):
    """ Guess the variable type.

    Parameters
    ----------
    s : pd.Series
        The variable to analyze

    Returns
    -------
    enumeration of TypeOfVariables
    """

    if not isinstance(s, pd.Series):
        raise TypeError(f"s should be a Serie, not a '{type(s)}'")

    st = str(s.dtype)

    if "int" in st:
        # Warning: sometime it could be categorical
        return VariableType.NUM

    elif "float" in st:
        return VariableType.NUM

    elif "object" in st:
        nb_u = s.nunique()  # number of different values
        nb = len(s)  # number of items

        if hasattr(s, "str"):  # For boolean
            avg_l = s.str.len().mean()
        else:
            avg_l = 0

        if avg_l >= 50 or nb_u >= 0.5 * nb:
            return VariableType.TEXT

        return VariableType.CAT

    elif "bool" in st:
        return VariableType.CAT

    elif "category" in st:
        return VariableType.CAT

    elif "category" in st:
        return VariableType.CAT

    else:
        raise ValueError(f"Invalid serie type: {st}")


def has_missing_values(s):
    """ Returns True if the serie has at least a missing value, False otherwise. """
    if not isinstance(s, pd.Series):
        raise TypeError(f"s should be a Serie, not a '{type(s)}'")
    return bool(np.asarray(s.isnull()).sum().sum() > 0)  # to prevent np.bool_


def get_columns_informations(df):
    """ returns the static information for each column """
    df_columns_info = OrderedDict()
    for c in df.columns:
        df_columns_info[c] = {
            "TypeOfVariable": guess_type_of_variable(df[c]),
            "HasMissing": has_missing_values(df[c]),
            "ToKeep": True,
        }
    return df_columns_info


def check_column_information(key, column_info):
    """ Verifies that the specified column_info dictionary has a valid structure. """

    if not isinstance(column_info, dict):
        raise TypeError(f"Column value must be a dictionary")

    # Should have 'HasMissing'
    if "HasMissing" not in column_info:
        raise ValueError(f"No 'HasMissing' information for column {key}, got keys: {column_info.keys()}")

    if not isinstance(column_info["HasMissing"], bool):
        raise TypeError(f"'HasMissing' should be a boolean,"
                        f"instead it is '{type(column_info['HasMissing'])}' for key {key}")

    # Should have 'TypeOfVariable'
    if "TypeOfVariable" not in column_info:
        raise ValueError(f"No 'TypeOfVariable' information for column {key}, got keys: {column_info.keys()}")

    if column_info["TypeOfVariable"] not in VariableType.alls:
        raise ValueError(f"Unknown 'TypeOfVariable' for key {key}: {column_info['TypeOfVariable']},"
                         f"it should be among ({str(VariableType.alls)})")

    # Should have 'ToKeep'
    if "ToKeep" not in column_info:
        raise ValueError(f"No 'ToKeep' information for column {key}, got keys: {column_info.keys()}")

    if not isinstance(column_info["ToKeep"], bool):
        raise TypeError(f"'ToKeep' should be a boolean for key {key}, instead it is '{type(column_info['ToKeep'])}'")


def get_columns_by_variable_type(columns_informations):
    """ get a dictionary with the list of columns for each type """
    var_type_columns_dico = OrderedDict()
    for col, info in columns_informations.items():
        if info["ToKeep"]:
            vt = info["TypeOfVariable"]
            if vt not in var_type_columns_dico:
                var_type_columns_dico[vt] = []
            var_type_columns_dico[vt].append(col)
    return var_type_columns_dico


def get_all_var_type(db_informations):
    all_var_type = tuple(sorted(set((v["TypeOfVariable"] for v in db_informations.values() if v["ToKeep"]))))
    return all_var_type


def get_var_type_columns_dict(columns_informations):
    """ get a dictionary with the list of columns for each type """
    var_type_columns_dict = OrderedDict()
    for col, info in columns_informations.items():
        if info["ToKeep"]:
            vt = info["TypeOfVariable"]
            if vt not in var_type_columns_dict:
                var_type_columns_dict[vt] = []
            var_type_columns_dict[vt].append(col)
    return var_type_columns_dict
