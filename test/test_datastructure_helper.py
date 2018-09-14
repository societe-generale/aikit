import pandas as pd
import numpy as np
from scipy import sparse

from aikit.data_structure_helper import get_type
from aikit.enums import DataTypes


def test_get_type():
    df = pd.DataFrame({"a": np.arange(10)})
    dfs = pd.SparseDataFrame({"a": [0, 0, 0, 1, 1]})
    assert get_type(df) == DataTypes.DataFrame
    assert get_type(df["a"]) == DataTypes.Serie
    assert get_type(df.values) == DataTypes.NumpyArray
    assert get_type(sparse.coo_matrix(df.values)) == DataTypes.SparseArray
    assert get_type(dfs) == DataTypes.SparseDataFrame
