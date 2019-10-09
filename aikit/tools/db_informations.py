# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 14:56:22 2018

@author: Lionel Massoulard
"""

from collections import OrderedDict

import pandas as pd
import numpy as np


from sklearn.utils.multiclass import type_of_target

import aikit.enums as en
from aikit.enums import TypeOfVariables


def guess_type_of_problem(dfX, y=None):
    """ try to guess the type of problem """

    if y is None:
        return en.TypeOfProblem.CLUSTERING

    ### Attention : ca ne marche pas pas bien si y est entier...mais continue

    if len(np.unique(y)) == 1:
        raise ValueError("y seems to be unique")

    if type_of_target(y) in ["binary", "multiclass"]:

        if "float" in str(y.dtype) or "int" in str(y.dtype):
            nb_u = len(np.unique(y))
            nb = len(y)
            if nb_u >= 0.25 * nb:
                return en.TypeOfProblem.REGRESSION
            else:
                return en.TypeOfProblem.CLASSIFICATION

        return en.TypeOfProblem.CLASSIFICATION
    else:
        return en.TypeOfProblem.REGRESSION
    # TODO : maybe we'll add more type of problem later...


# Rmk : on a en fait d'autres types
# => colonne mixte avec des NOMBRES et des STRINGS parfois.
# x = pd.Series(np.random.randn(100))
# x[[10,20,30]] = "string"
#
# des trucs avec des IDs..
# des colonne avec des entier qui sont des categorie


def guess_type_of_variable(s):
    """ guess the type of a variable

    Parameters
    ----------
    s : pd.Series
        The variavble to analyze

    Result:
    -------
    enumeration of TypeOfVariables


    """

    if not isinstance(s, pd.Series):
        raise TypeError("s should be a Serie, not a '%s'" % type(s))

    st = str(s.dtype)

    if "int" in st:
        return TypeOfVariables.NUM
        # Carefull : sometime it could be categorical

    elif "float" in st:
        return TypeOfVariables.NUM

    elif "object" in st:
        nb_u = s.nunique()  # number of different values
        nb = len(s)  # number of items

        if hasattr(s, "str"):  # For boolean
            avg_l = s.str.len().mean()
        else:
            avg_l = 0

        if avg_l >= 50 or nb_u >= 0.5 * nb:
            return TypeOfVariables.TEXT

        return TypeOfVariables.CAT

    elif "bool" in st:
        return TypeOfVariables.CAT

    elif "category" in st:
        return TypeOfVariables.CAT

    elif "category" in st:
        return TypeOfVariables.CAT

    else:
        raise NotImplementedError("I don't know that type of Series : %s, please check" % st)


def has_missing_values(s):
    """ does the serie has missing value """

    if not isinstance(s, pd.Series):
        raise TypeError("s should be a Serie, not a '%s'" % type(s))

    return bool(s.isnull().sum() > 0)  # to prevent np.bool_


def get_columns_informations(dfX):
    """ return the static information for each columns """

    # result = OrderedDict([ (c,{"TypeOfVariable":guess_type_of_variable(dfX[c]),
    #                           "HasMissing":has_missing_values(dfX[c])}) for c in dfX.columns])
    # ... A little to difficult to read

    df_columns_info = OrderedDict()
    for c in dfX.columns:
        df_columns_info[c] = {
            "TypeOfVariable": guess_type_of_variable(dfX[c]),
            "HasMissing": has_missing_values(dfX[c]),
            "ToKeep": True,
        }

    # df_result = pd.DataFrame(result).T # readable DataFrame if needed
    return df_columns_info


def get_var_type_columns_dico(columns_informations):
    """ get a dictionnary with the list of columns for each type """
    var_type_columns_dico = OrderedDict()
    for col, info in columns_informations.items():

        if info["ToKeep"]:
            vt = info["TypeOfVariable"]
            if vt not in var_type_columns_dico:
                var_type_columns_dico[vt] = []

            var_type_columns_dico[vt].append(col)

    return var_type_columns_dico


def _update_columns_information(columns_info, column_to_update, **updates):
    """ modification of df_columns_info, made by user for example

    Parameters
    ----------
    dico_columns_info : dictionnary of information per column
        For example, the result of get_DataFrame_information

    columns_to_update : str
        the column that we want to change

    **updates : dictionnary of field and their value

    Returns
    -------
    modified dictionnary of information per column

    """
    if column_to_update not in columns_info:
        raise ValueError("column_to_update should be one of the column, instead I got '%s'" % column_to_update)

    for field_to_update, new_value in updates.items():
        if field_to_update not in columns_info[column_to_update]:
            raise ValueError("The field '%s' isn't present in the column information" % field_to_update)

        if field_to_update == "TypeOfVarable":
            if new_value not in TypeOfVariables.all_types:
                raise ValueError("the value '%s' isn't a valid update for %s" % (new_value, field_to_update))

        elif field_to_update == "HasMissing":
            if not isinstance(new_value, bool):
                raise ValueError("the value '%s' isn't a valid update for %s" % (new_value, field_to_update))

        elif field_to_update == "ToKeep":
            if not isinstance(new_value, bool):
                raise ValueError("the value '%s' isn't a valid update for %s" % (new_value, field_to_update))
        else:
            raise NotImplementedError("I need to code the check for that type of field : %s" % field_to_update)

        columns_info[column_to_update][field_to_update] = new_value

    return columns_info


def get_all_var_type(db_informations):
    all_var_type = tuple(sorted(set((v["TypeOfVariable"] for v in db_informations.values() if v["ToKeep"]))))  ###
    return all_var_type


def get_n_outputs(y):
    """ returns the number of ouputs of a given target """
    if getattr(y, "ndim", 1) > 1:
        return y.shape[1]
    else:
        return 1
