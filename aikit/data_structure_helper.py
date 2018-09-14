import pandas as pd
import numpy as np
from scipy import sparse

from aikit.enums import DataTypes


def get_type(data):
    """
    Retrieve the type of a data

    Parameters
    ----------
    data: pandas.DataFrame, numpy.array, ...

    Returns
    -------
    data_type or None
    """
    type_of_data = type(data)

    if type_of_data == pd.DataFrame:
        return DataTypes.DataFrame
    elif type_of_data == pd.Series:
        return DataTypes.Serie
    elif type_of_data == np.ndarray:
        return DataTypes.NumpyArray
    elif type_of_data == pd.SparseDataFrame:
        return DataTypes.SparseDataFrame
    elif sparse.issparse(data):
        return DataTypes.SparseArray
    else:
        return None
