# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 14:24:58 2018

@author: Lionel Massoulard
"""


from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.utils import indexable

from aikit.tools.data_structure_helper import _nbrows, _nbcols


class BlockSelector(BaseEstimator, TransformerMixin):
    """ Transformer that works on blocks of data to select it """

    def __init__(self, block_to_select):
        self.block_to_select = block_to_select

        self._already_fitted = False

        self._input_type = None
        self._Xres_columns = None
        self._Xres_shape = None

    def _fit_transform(self, X, y, is_fit, is_transform):

        if is_fit:
            self._input_type = type(X)
        else:
            if type(X) != self._input_type:
                raise TypeError("The model except X to be of type %s, instead I got %s" % (self._input_type, type(X)))

        try:
            Xres = X[self.block_to_select]
        except (KeyError, TypeError):
            raise ValueError("%s doesn't exist in the data" % self.block_to_select)

        if is_fit:
            self._Xres_columns = getattr(Xres, "columns", None)
            self._Xres_shape = getattr(Xres, "shape", None)

        if is_transform:
            return Xres
        else:
            return self

    def _check_is_fitted(self):
        if not self._already_fitted:
            raise NotFittedError(
                "This %s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method." % type(self).__name__
            )

    def fit(self, X, y=None):

        self._fit_transform(X, y=None, is_fit=True, is_transform=False)
        self._already_fitted = True

        return self

    def transform(self, X):
        self._check_is_fitted()

        Xres = self._fit_transform(X, y=None, is_fit=False, is_transform=True)

        return Xres

    def fit_transform(self, X, y=None):

        Xres = self._fit_transform(X, y=None, is_fit=True, is_transform=True)
        self._already_fitted = True

        return Xres

    def get_feature_names(self, input_features=None):
        self._check_is_fitted()

        if input_features is not None:
            try:
                feature = input_features[self.block_to_select]
            except (KeyError, TypeError):
                feature = None

            if feature is not None:
                return list(feature)

        if self._Xres_columns is not None:
            return list(self._Xres_columns)

        if self._Xres_shape is not None:
            return list(range(self._Xres_shape[1]))

        else:
            raise ValueError("I can't find feature_names")

    def approx_cross_validation(
        self,
        X,
        y=None,
        groups=None,
        cv=None,
        verbose=1,
        fit_params=None,
        return_predict=True,
        method="transform",
        no_scoring=True,
        stopping_round=None,
        stopping_threshold=None,
        approximate_cv=False,
        **kwargs
    ):

        if not return_predict:
            raise ValueError("This is a transformer it should only be called with 'return_predict = True'")

        if not no_scoring:
            raise ValueError("This is a transformer it should only be called with 'no_scoring = True'")

        if method != "transform":
            raise ValueError("This is a transformer it should only be called with 'method = 'transform'")

        self_clone = clone(self)
        return None, self_clone.fit_transform(X, y)

    def can_cv_transform(self):
        return True


# In[]


def _safe_indexing(X, indices):
    """Return items or rows from X using indices.

    copy of sklearn utils safe_indexing
    with handling of slice as well
    
    Allows simple indexing of lists or arrays.

    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series.
        Data from which to sample rows or items.
    indices : array-like of int
        Indices according to which X will be subsampled.

    Returns
    -------
    subset
        Subset of X on first axis

    Notes
    -----
    CSR, CSC, and LIL sparse matrices are supported. COO sparse matrices are
    not supported.
    """
    if hasattr(X, "iloc"):
        # Work-around for indexing with read-only indices in pandas
        if hasattr(indices, "flags"):  # 'aikit' patch
            indices = indices if indices.flags.writeable else indices.copy()
        # Pandas Dataframes and Series
        try:
            return X.iloc[indices]
        except ValueError:
            # Cython typed memoryviews internally used in pandas do not support
            # readonly buffers.
            return X.copy().iloc[indices]
    elif hasattr(X, "shape"):
        if hasattr(X, "take") and (hasattr(indices, "dtype") and indices.dtype.kind == "i"):
            # This is often substantially faster than X[indices]
            return X.take(indices, axis=0)
        else:
            return X[indices]
    else:
        return [X[idx] for idx in indices]


class _IlocBlockManager(object):
    """ subsetting object for the DataManager """

    def __init__(self, dtm):
        self.dtm = dtm

    def __getitem__(self, index):
        return self.dtm.iloc_fun(index)


class BlockManager(object):
    """ Light wrapper around a dictionnary of data object, to allow subsetting using iloc.
    This object is meants to store different type of data that you don't want to merge together but that you can still keep aligned.
    
    It is basically a dictionnary of data with a few more things :
        * shape attributes
        * possiblity to subset it using iloc
        
    Those two things allow the BlockManager to be used as any other type of data (DataFrame, numpy array).
    It works well with the BlockSelector transformer.
    
    Parameters
    ----------
    all_datas : list or dictionnary
        the data to store in the BlockManager
        
    Example
    -------

    df = pd.DataFrame({"a":np.arange(10),"b":["aaa","bbb","ccc"] * 3 + ["ddd"]})
    arr = np.random.randn(df.shape[0],5)
    
    X = BlockManager({"df":df, "arr":arr})
    X["df"]    # retrieve df
    
    X["arr"]   # retrive array
    
    Xsubset = X.iloc[0:5,:]    # smaller BlockManager
    
    X.shape    # (10,7)
    """

    def __init__(self, all_datas):

        self.all_datas = all_datas
        self._iloc = None

        self._verif()
        self._make_indexable()

    def _verif(self):

        if not isinstance(self.all_datas, (list, dict)):
            raise TypeError("I don't know how to handle that type of Data : %s" % type(self.all_datas))

        if hasattr(self.all_datas, "items"):
            nbrows = [_nbrows(data) for key, data in self.all_datas.items()]
            nbcols = [_nbcols(data) for key, data in self.all_datas.items()]
            self._is_dict = True
        else:
            nbrows = [_nbrows(data) for data in self.all_datas]
            nbcols = [_nbcols(data) for data in self.all_datas]
            self._is_dict = False

        if len(set(nbrows)) > 1:
            raise ValueError("All objects don't have the same length")

        self._nbrows = nbrows[0]
        self._nbcols = sum(nbcols)

    def _make_indexable(self):
        """ make sure everything in the BlockManager is indexable """
        if self._is_dict:
            # keys = list(self.all_datas.keys())
            for k, v in self.all_datas.items():
                self.all_datas[k] = indexable(v)[0]

        else:
            self.all_datas = indexable(*self.all_datas)

    @property
    def iloc(self):
        """ access to subsetting object """
        if self._iloc is None:
            self._iloc = _IlocBlockManager(self)
            # As in pandas DataFrame, dynamically created to limit memory issue (because it is created a loop of references)

        return self._iloc

    def __repr__(self):
        string = super(BlockManager, self).__repr__()
        n, c = self.shape
        if self._is_dict:
            return "\n".join([string, "nrows %d" % n, "ncols %d" % c, "keys :"] + list(self.all_datas.keys()))
        else:
            return "\n".join(
                [string, "nrows %d" % n, "ncols %d" % c, "keys :"] + ["%d" % d for d in range(len(self.all_datas))]
            )

    def __str__(self):
        return self.__repr__()

    @property
    def columns(self):
        if isinstance(self.all_datas, dict):
            result = self.all_datas.__class__()
            for k, v in self.all_datas.items():
                result[k] = getattr(v, "columns", None)

        else:
            result = [getattr(v, "columns", None) for v in self.all_datas]

        return result

    @property
    def shape(self):
        return self._nbrows, self._nbcols

    def keys(self):
        """ iterators on keys """
        if self._is_dict:
            return self.all_datas.keys()
        else:
            return range(len(self.all_datas))  # return an iterator

    def items(self):
        """ iterators on items """
        if self._is_dict:
            return self.all_datas.items()  # doesn't work on list datas
        else:
            return enumerate(self.all_datas)

    def values(self):
        """ iterator on datas """
        if self._is_dict:
            return self.all_datas.values()
        else:
            return (datas for datas in self.all_datas)  # return generator

    def __getitem__(self, block):
        try:
            res = self.all_datas[block]
        except (ValueError, IndexError, TypeError, KeyError):
            raise KeyError("Can't subset block %s" % block)

        return res

    def iloc_fun(self, index):
        """ subsetting function, will create a BlockManager subsetted using index """
        if isinstance(self.all_datas, dict):
            all_data_subset = self.all_datas.__class__()

            for key, value in self.all_datas.items():
                all_data_subset[key] = _safe_indexing(value, index)

        elif isinstance(self.all_datas, (list, tuple)):
            all_data_subset = [_safe_indexing(value, index) for value in self.all_datas]

        else:
            raise TypeError("I don't know how to handle that type of Data : %s" % type(self.all_datas))

        return BlockManager(all_data_subset)


class TransformToBlockManager(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return BlockManager(X)


# In[]
