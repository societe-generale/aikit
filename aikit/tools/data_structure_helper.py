# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:34:01 2018

@author: Lionel Massoulard
"""

import pandas as pd
import numpy as np
from scipy import sparse

from aikit.enums import DataTypes


def get_type(data):
    """Retrieve the type of a data 
    
    Parameters
    ----------
    data : pd.DataFrame,np.array, ...
         the thing we want the type
         
    Returns
    -------
    one data_type  or None
    
    Example
    -------
    >>> df = pd.DataFrame({"a":np.arange(10)})
    >>> dfs = pd.SparseDataFrame({"a":[0,0,0,1,1]})
    >>> assert get_type(df) == DataTypes.DataFrame
    >>> assert get_type(df["a"]) == DataTypes.Serie
    >>> assert get_type(df.values ) == DataTypes.NumpyArray
    >>> assert get_type(sparse.coo_matrix(df.values)) == DataTypes.SparseArray
    >>> assert get_type(dfs) == DataTypes.SparseDataFrame
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


def get_rid_of_categories(df):
    did_copy = False
    for col in df.columns:
        if str(df[col].dtype) == "category":
            if not did_copy:
                df = df.copy()
                did_copy = True

            df[col] = df[col].get_values()

    return df


def convert_to_dataframe(xx, mapped_type=None):
    """ convert something to a DataFrame """
    if mapped_type is None:
        mapped_type = get_type(xx)

    if mapped_type is None:
        return pd.DataFrame(xx)  # try to create a DataFrame no matter what

    if mapped_type == DataTypes.DataFrame:
        return xx

    elif mapped_type == DataTypes.Serie:
        return pd.DataFrame(xx, index=xx.index)

    elif mapped_type == DataTypes.NumpyArray:
        return pd.DataFrame(xx)

    elif mapped_type == DataTypes.SparseDataFrame:
        return xx.to_dense()

    elif mapped_type == DataTypes.SparseArray:
        return pd.DataFrame(xx.todense())

    else:
        raise TypeError("I don't know that type : %s" % type(xx))


def convert_to_array(xx, mapped_type=None):
    """ convert something to a Numpy Array """

    if mapped_type is None:
        mapped_type = get_type(xx)

    if mapped_type is None:
        return convert_to_array(convert_to_dataframe(xx))

    if mapped_type == DataTypes.DataFrame:
        return xx.values

    elif mapped_type == DataTypes.Serie:
        return xx.values.reshape((xx.shape[0], 1))

    elif mapped_type == DataTypes.NumpyArray:
        if xx.ndim == 1:
            return xx.reshape((xx.shape[0], 1))
        else:
            return xx

    elif mapped_type == DataTypes.SparseArray:
        if xx.dtype == np.object:
            return np.array(xx.astype(np.float64).todense())
        else:
            return np.array(xx.todense())  # np.array to prevent type 'matrix'

    elif mapped_type == DataTypes.SparseDataFrame:
        return xx.to_dense().values

    else:
        raise TypeError("I don't know how to convert that %s" % type(xx))


def convert_to_sparsearray(xx, mapped_type=None):
    """ convert something to a Sparse Array """

    if mapped_type is None:
        mapped_type = get_type(xx)

    if mapped_type is None:
        return convert_to_sparsearray(convert_to_dataframe(xx))

    if mapped_type == DataTypes.DataFrame:
        return sparse.csr_matrix(xx.values)

    elif mapped_type == DataTypes.Serie:
        sparse.csr_matrix(xx.values[:, np.newaxis])  # np.newaxis to make sure I have 2 dimensio

    elif mapped_type == DataTypes.NumpyArray:
        if xx.ndim == 1:
            return sparse.csr_matrix(xx.reshape((xx.shape[0], 1)))
        else:
            return sparse.csr_matrix(xx)

    elif mapped_type == DataTypes.SparseArray:
        return xx

    elif mapped_type == DataTypes.SparseDataFrame:
        return xx.to_coo()  # maybe convert to csr ?

    else:
        raise TypeError("I don't know how to convert that %s" % type(xx))


def convert_to_sparsedataframe(xx, mapped_type=None):
    """ convert something to a Sparse DataFrame """

    if mapped_type is None:
        mapped_type = get_type(xx)

    if mapped_type is None:
        return convert_to_sparsedataframe(convert_to_dataframe(xx))

    if mapped_type == DataTypes.DataFrame:
        return pd.SparseDataFrame(xx, default_fill_value=0)

    elif mapped_type == DataTypes.Serie:
        return pd.SparseDataFrame(pd.DataFrame(xx), default_fill_value=0)

    elif mapped_type == DataTypes.NumpyArray:
        if xx.ndim == 1:
            return pd.SparseDataFrame(xx.reshape((xx.shape[0], 1)), default_fill_value=0)
        else:
            return pd.SparseDataFrame(xx, default_fill_value=0)

    elif mapped_type == DataTypes.SparseArray:
        return pd.SparseDataFrame(xx, default_fill_value=0)

    elif mapped_type == DataTypes.SparseDataFrame:
        return xx

    else:
        raise TypeError("I don't know how to convert that %s" % type(xx))


def convert_tononsparse(xx, mapped_type=None):
    """ convert something to the dense version of itself : 
        * SparseArray => Array
        * SparseDataFrame => DataFrame 
    """
    if mapped_type is None:
        mapped_type = get_type(xx)

    if mapped_type == DataTypes.SparseArray:
        return convert_to_array(xx)

    elif mapped_type == DataTypes.SparseDataFrame:
        return convert_to_dataframe(xx)

    else:
        return xx


def convert_tonondataframe(xx, mapped_type=None):
    """ convert something to an array :
        * DataFrame => Array
        * SparseDataFrame => Sparse Array
    """

    if mapped_type is None:
        mapped_type = get_type(xx)

    if mapped_type == DataTypes.DataFrame:
        return convert_to_array(xx)

    elif mapped_type == DataTypes.SparseDataFrame:
        return convert_to_sparsearray(xx)

    else:
        return xx


def convert_generic(xx, mapped_type=None, output_type=None):
    """ generic conversion function 
    
    Parameters
    ----------
    xx : array, DataFrame, ...
        the object to convert
        
    mapped_type : enumeration from enums.DataTypes or None
        if not None, the type enumeration of xx
        
    output_type : enumeration from enums.DataTypes or None
        if not None the desired output tpye
        
    """

    if output_type is None:
        return xx  # do nothing

    if mapped_type is None:
        mapped_type = get_type(xx)

    if output_type == DataTypes.DataFrame:
        return convert_to_dataframe(xx, mapped_type=mapped_type)

    elif output_type == DataTypes.NumpyArray:
        return convert_to_array(xx, mapped_type=mapped_type)

    elif output_type == DataTypes.SparseArray:
        return convert_to_sparsearray(xx, mapped_type=mapped_type)

    elif output_type == DataTypes.SparseDataFrame:
        return convert_to_sparsedataframe(xx, mapped_type=mapped_type)

    else:
        raise ValueError("I don't know this output_type : %s" % output_type)


def _nbcols(data):
    """ retrieve the number of columns of an object
    
    Example
    -------
    >>> df = pd.DataFrame({"a":np.arange(10),"b":["aa","bb","cc"]*3+["dd"]})
    >>> assert _nbcols(df) == 2
    >>> assert _nbcols(df.values) == 2
    >>> assert _nbcols(df["a"]) == 1
    >>> assert _nbcols(df["a"].values) == 1
    """
    s = getattr(data, "shape", None)
    if s is None:
        return 1
    else:
        if len(s) == 1:
            return 1
        else:
            return s[1]


def _nbrows(data):
    """ retrieve the number of rows of an object 
    
    Example
    -------

    >>> df = pd.DataFrame({"a":np.arange(10),"b":["aa","bb","cc"]*3+["dd"]})
    >>> assert _nbrows(df) == 10
    >>> assert _nbrows(df.values) == 10
    >>> assert _nbrows(df["a"]) == 10
    >>> assert _nbrows(df["a"].values) == 10
    
    """
    s = getattr(data, "shape", None)
    if s is None:
        return len(data)
    else:
        return s[0]


def guess_output_type(all_datas):
    """ try to guess which output type should be better based on size of the data """

    MAX_NUMBER_OF_CELLS = 10000000  #  1000 * 10000 # 1000 columns and 10 000 rows
    all_types = [get_type(data) for data in all_datas]
    all_types = list(np.unique([t for t in all_types if t is not None]))

    if len(all_types) == 0:
        # Only unknown  thing
        return DataTypes.DataFrame

    elif len(all_types) == 1:
        # Only one type of things
        return all_types[0]

    if all([t in (DataTypes.SparseArray, DataTypes.SparseDataFrame) for t in all_types]):
        # only Sparse things
        return DataTypes.SparseArray
        # return DataTypes.SparseDataFrame

    elif all([t in (DataTypes.NumpyArray, DataTypes.DataFrame) for t in all_types]):
        # only non sparse things
        return DataTypes.DataFrame

    elif all([t in (DataTypes.NumpyArray, DataTypes.SparseArray) for t in all_types]):
        # only arrays
        expected_number_of_columns = int(np.sum([_nbcols(data) for data in all_datas]))
        # carefull np.sum result should be cast at int : otherwise it can be np.int32 and generate overflow which can make the product

        # Lots of data point
        if expected_number_of_columns * _nbrows(all_datas[0]) >= MAX_NUMBER_OF_CELLS:
            return DataTypes.SparseArray
        else:
            return DataTypes.NumpyArray

    else:
        expected_number_of_columns = int(np.sum([_nbcols(data) for data in all_datas]))
        # careful np.sum result should be cast at int : otherwise it can be np.int32 and generate overflow
        if expected_number_of_columns * _nbrows(all_datas[0]) >= MAX_NUMBER_OF_CELLS:
            return DataTypes.SparseArray
            # return DataTypes.SparseDataFrame
        else:
            return DataTypes.DataFrame


def guess_hstack_columns(all_datas, raise_on_duplicate=True, pattern_if_not_columns="%d_%d", all_columns_names=None):
    """ from a list of data, create the aggregated columns name 
    
    Parameters
    ----------
    all_datas : list of data
        the datas that should be aggregated
        
    raise_on_duplicate : boolean, default = True
        if True will raise an error if duplicate columns name are found,
        otherwise will add '_1' ,'_2' to differentiate columns name

    pattern_if_not_columns : str, default '%d_%d"
        pattern to use for column name when data isn't a DataFrame
        column name will be : " pattern_if_not_columns % (block_number, column_number)
        
    all_columns_names  : list or None
        if not None it is the name of the columns of each block
    
    Returns
    -------
    list of columns
    """

    columns = []
    scolumns = set()

    for block_nb, data in enumerate(all_datas):

        if get_type(data) in (DataTypes.DataFrame, DataTypes.SparseDataFrame):
            cols = list(data.columns)

        elif all_columns_names is not None:
            cols = []
            if all_columns_names[block_nb] is not None:
                cols = all_columns_names[block_nb]
                if len(cols) != _nbcols(data):
                    raise ValueError(
                        "all_columns_names[%d] doesn't have the right length (%d, expected %d)"
                        % (block_nb, len(cols), _nbcols(data))
                    )
            else:
                cols = [pattern_if_not_columns % (block_nb, i) for i in range(_nbcols(data))]
        else:
            cols = [pattern_if_not_columns % (block_nb, i) for i in range(_nbcols(data))]

        if raise_on_duplicate:
            for c in cols:
                if c in scolumns:
                    raise ValueError("I have a duplicate column '%s'" % c)

            columns += cols
        else:
            for col in cols:
                d = 1
                while True:
                    if d == 1:
                        c = col
                    else:
                        c = col + ("_%d" % d)
                    if c not in scolumns:
                        break

                    d += 1

                columns.append(col)
                scolumns.add(col)

    return columns


def guess_hstack_index(all_datas, raise_if_different=False):
    """ generic function to guess the index of a concatenation """

    # All the indexes in all the block of data
    all_indexes = [d.index for d in all_datas if hasattr(d, "index")]

    # No indexes
    if len(all_indexes) == 0:
        return np.arange(_nbrows(all_datas[0]))  # default index

    # Only one indexes => I'll use it
    elif len(all_indexes) == 1:
        return all_indexes[0]

    # Check if all indexes are the same
    all_the_same = True
    for ind in all_indexes[1:]:
        if not np.array_equal(ind.values, all_indexes[0].values):
            all_the_same = False
            break

    if all_the_same:
        # ... If all the same => just return the first one
        return all_indexes[0]

    if raise_if_different:
        raise ValueError("I have different indexes")

    # I'll check if I have an index that isn't [0,1,2, .... , nbrow - 1]
    nb_row = _nbrows(all_datas[0])
    for ind in all_indexes:
        # If that is the case... I'll return this one
        if not np.array_equal(ind.values, np.arange(nb_row)):
            return ind

        # I do that because it probably means that the correct indexes is that one (and the other are just defaulting indexes)

    # 3) Otherwise, just return the first one
    return all_indexes[0]


def generic_hstack(all_datas, output_type=None, all_columns_names=None):
    """ generic function to concatenate horizontaly some datas objects
    
    All datas should have the same number of rows
    
    Parameters
    ----------
    all_datas : list of data object
        the things that we want to concatenate
        
    output_type : None or type of data
        if None will guess the type (See 'guess_output_type')
        otherwise will concatenate using that format
        
    all_columns_names = None or list of names
        if not None it corresponds to the list of columns of each sub datas
        
    Returns
    -------
    aggregated object
    """

    if output_type is None:
        output_type = guess_output_type(all_datas)

    all_datas = [data for data in all_datas if _nbcols(data) > 0]
    nb_of_datas = len(all_datas)

    if output_type == DataTypes.DataFrame:

        if nb_of_datas == 0:
            raise NotImplementedError(
                "no object to concat"
            )  # il faudrait faire un truc qui marche quand meme... et qu'on peut concatener apres peut etre...

        elif nb_of_datas == 1:
            return convert_to_dataframe(all_datas[0])  # no concat => no copy, no change of index
        else:
            nbrow0 = _nbrows(all_datas[0])
            for data in all_datas[1:]:
                if _nbrows(data) != nbrow0:
                    raise ValueError("I can't concatenate things of differente size")

            result = pd.concat(
                [convert_to_dataframe(data).reset_index(drop=True) for data in all_datas], axis=1, ignore_index=True
            )

            result.columns = guess_hstack_columns(
                all_datas, raise_on_duplicate=True, all_columns_names=all_columns_names
            )
            result.index = guess_hstack_index(all_datas, raise_if_different=False)

            return result

    elif output_type == DataTypes.NumpyArray:
        if nb_of_datas == 0:
            raise NotImplementedError("no object to concat")

        elif nb_of_datas == 1:
            return convert_to_array(all_datas[0])

        else:
            return np.hstack([convert_to_array(data) for data in all_datas])

    elif output_type == DataTypes.SparseArray:

        if nb_of_datas == 0:
            raise NotImplementedError("no object to concat")

        elif nb_of_datas == 1:
            return convert_to_sparsearray(all_datas[0])

        else:
            all_sparse_data = [convert_to_sparsearray(data) for data in all_datas]
            for i, data in enumerate(all_sparse_data):
                if data.dtype == np.object:
                    all_sparse_data[i] = data.astype(np.float64)

            return sparse.hstack(all_sparse_data)

    elif output_type == DataTypes.SparseDataFrame:

        if nb_of_datas == 0:
            raise NotImplementedError("no object to concat")

        elif nb_of_datas == 1:
            return convert_to_sparsedataframe(all_datas[0])

        else:

            nbrow0 = _nbrows(all_datas[0])
            for data in all_datas[1:]:
                if _nbrows(data) != nbrow0:
                    raise ValueError("I can't concatenate things of differente size")

            result = pd.concat(
                [convert_to_sparsedataframe(data).reset_index(drop=True) for data in all_datas],
                axis=1,
                ignore_index=True,
            )

            result.columns = guess_hstack_columns(
                all_datas, raise_on_duplicate=True, all_columns_names=all_columns_names
            )
            result.index = guess_hstack_index(all_datas, raise_if_different=False)

            return result

    else:
        raise TypeError("I don't know that type of conversion %s " % output_type)


# In[] :


def make2dimensions(X):
    """ generic function to make a data object at least bi-dimensional
    
    Example
    -------
    >>> df = pd.DataFrame({"a":np.arange(10),"b":["aa","bb","cc"]*3 + ["dd"]})
    >>> assert make2dimensions(df).shape == (10,2)
    >>> assert make2dimensions(df["a"]).shape == (10,1)
    >>> assert make2dimensions(df.values).shape == (10,2)
    >>> assert make2dimensions(df["a"].values).shape == (10,1)
    """

    Xtype = get_type(X)

    if Xtype in (DataTypes.DataFrame, DataTypes.SparseDataFrame):
        return X

    elif Xtype == DataTypes.Serie:
        return pd.DataFrame(X)

    elif Xtype in (DataTypes.NumpyArray, DataTypes.SparseArray):
        ndim = getattr(X, "ndim", None)
        if ndim is None:
            raise ValueError("I don't know how to deal with that type of object '%s'" % type(X))
            # should never go there

        if ndim == 2:
            return X

        elif ndim == 1:
            return X.reshape((X.shape[0], 1))

        else:
            raise ValueError("I don't now how to handle object of dimension %d" % ndim)

    else:
        raise ValueError("I don't know how to deal with that type of object '%s'" % type(X))


def make1dimension(X):
    """ generic function to make an object uni dimensional """

    Xtype = get_type(X)

    if Xtype == DataTypes.DataFrame:
        if X.shape[1] > 1:
            raise ValueError("This object is 2 dimensional")
        return X.iloc[:, 0]

    elif Xtype == DataTypes.Serie:
        return X

    elif len(X.shape) == 2:
        if X.shape[1] > 1:
            raise ValueError("This object is 2 dimensional")
        return X[:, 0]

    elif len(X.shape) == 1:
        return X

    raise ValueError("I don't know how to deal with that type of object '%s'" % type(X))


def _get_index(x):
    """ retrieve the index of something if it exists """
    if hasattr(x, "index"):
        return x.index
    else:
        return None


def _get_columns(x):
    """ retrieve the columns of something if it exists """
    if hasattr(x, "columns"):
        return list(x.columns)
    else:
        return None


def _set_columns(x, columns):
    """ set the columns attribute of something it is possible
    
    if columns is None, or x doesn't have an columns attribute it won't do anything
    
    """
    if columns is None:
        return x

    if hasattr(x, "columns"):
        x.columns = columns

    return x


def _set_index(x, index):
    """ set the index attribute of something it is possible
    
    if index is None, or x doesn't have an index attribute it won't do anything
    
    """
    if index is None:
        return x

    if hasattr(x, "index"):
        x.index = index

    return x


# In[]
