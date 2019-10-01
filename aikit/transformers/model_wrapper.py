# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:41:27 2018

@author: Lionel Massoulard
"""

import re

# from scipy import sparse
import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.metaestimators import if_delegate_has_method

from aikit.enums import DataTypes
from aikit.tools.helper_functions import intersect, diff, exception_improved_logging

import aikit.tools.data_structure_helper as dsh
from aikit.tools.helper_functions import function_has_named_argument

from sklearn.exceptions import NotFittedError


class ColumnsSelector(TransformerMixin, BaseEstimator):
    """ sklearn Transformer to select columns
    
    Parameters
    ----------
    columns_to_use : list of str (or int), or None
        the columns to keep, if None (everything)
            
    columns_to_drop : list of str (or int), or None
        the columns to drop, if None (nothing)

    regex_match : boolean, default = False
        if True will use regex to find the columns to use, otherwise will use exact match
    
    """

    def __init__(self, columns_to_use=None, columns_to_drop=None, regex_match=False):

        self.columns_to_use = columns_to_use
        self.columns_to_drop = columns_to_drop

        self.regex_match = regex_match

        self._already_fitted = False

    @staticmethod
    def convert_to_list(cols_list):
        """ helper function to make sure something is a list """
        if isinstance(cols_list, str):
            return [cols_list]

        elif isinstance(cols_list, int):
            return [cols_list]

        elif isinstance(cols_list, np.ndarray):

            if cols_list.ndim != 1:
                raise ValueError("columns_to_use should be a list of columns")

            return list(cols_list)

        elif isinstance(cols_list, (list, tuple)):
            return list(cols_list)

        else:
            raise ValueError("I don't know how to treat columns_to_use")

    def fit(self, X, y=None):
        self._expected_type = dsh.get_type(X)
        self._expected_nbcols = dsh._nbcols(X)

        ### Columns to use ###
        if self.columns_to_use is None:
            list_columns_to_use = None  # [i for i in range(self._expected_nbcols)]
        else:
            list_columns_to_use = self.convert_to_list(cols_list=self.columns_to_use)

        ### Columns to drop ###
        if self.columns_to_drop is None:
            list_columns_to_drop = None
        else:
            list_columns_to_drop = self.convert_to_list(cols_list=self.columns_to_drop)

        if list_columns_to_use is not None and len(list_columns_to_use) == 0:
            raise ValueError("columns_to_use is empty")

        ### What is the type of columns_to_use and columns_to_drop :
        if list_columns_to_use is not None:
            is_int = "int" in str(type(list_columns_to_use[0]))
        else:
            is_int = None

        if list_columns_to_drop is not None and len(list_columns_to_drop) > 0:
            is_int_to_drop = "int" in str(type(list_columns_to_drop[0]))
        else:
            is_int_to_drop = is_int

        ### Verify type:
        if is_int is not None and is_int_to_drop is not None:
            if is_int != is_int_to_drop:
                raise ValueError(
                    "Please be consistent between columns_to_use and columns_to_drop, both can be integer or str, but they should have the same type"
                )

        if is_int is None and is_int_to_drop is None:
            is_int = True
            is_int_to_drop = True

        if is_int is None and is_int_to_drop is not None:
            is_int = is_int_to_drop
        if is_int_to_drop is None and is_int is not None:
            is_int_to_drop = is_int

        if self._expected_type in (DataTypes.DataFrame, DataTypes.SparseDataFrame):
            if is_int:

                ##############################################
                ### Case 1 : DataFrame + Integer selection ###
                ##############################################

                if self.regex_match:
                    #######################
                    ## Case 1a : + Regex ##
                    #######################
                    raise ValueError("regex_match can only work with strings 'columns_to_use' not int")

                cols_set = set(range(self._expected_nbcols))
                if list_columns_to_use is not None:

                    # Check all column are available

                    for l in list_columns_to_use:
                        if l not in cols_set:
                            raise ValueError("Column %d isn't in the columns of the DataFrame" % l)

                    final_columns_to_use = list_columns_to_use
                    # final_columns_to_use = intersect( list_columns_to_use  , list(range(self._expected_nbcols)) )
                else:
                    final_columns_to_use = list(range(self._expected_nbcols))

                if list_columns_to_drop is not None:

                    for l in list_columns_to_drop:
                        if l not in cols_set:
                            raise ValueError("Column %d isn't in the columns of the DataFrame" % l)

                    final_columns_to_use = diff(final_columns_to_use, list_columns_to_drop)

            else:

                #############################################
                ### Case 2 : DataFrame + String selection ###
                #############################################
                if self.regex_match:
                    #######################
                    ## Case 2a : + Regex ##
                    #######################
                    if list_columns_to_use is not None:
                        cols_that_match = []
                        for col in list(X.columns):
                            for r in list_columns_to_use:
                                if re.search(r, col) is not None:
                                    cols_that_match.append(col)
                                    break

                    if list_columns_to_drop is not None:
                        cols_that_match_drop = []
                        for col in list(X.columns):
                            for r in list_columns_to_drop:
                                if re.search(r, col) is not None:
                                    cols_that_match_drop.append(col)
                                    break

                    if list_columns_to_use is not None:
                        final_columns_to_use = cols_that_match
                        # final_columns_to_use = intersect(cols_that_match ,  list(X.columns)) # technically the intersect is useless
                    else:
                        final_columns_to_use = list(X.columns)

                    if list_columns_to_drop is not None:
                        final_columns_to_use = diff(final_columns_to_use, cols_that_match_drop)

                else:
                    ########################
                    ## Case 2b : no Regex ##
                    ########################
                    cols_set = set(X.columns)
                    if list_columns_to_use is not None:

                        for l in list_columns_to_use:
                            if l not in cols_set:
                                raise ValueError("Column %s isn't in the columns of the DataFrame" % l)
                        final_columns_to_use = list_columns_to_use  # intersect(list_columns_to_use, list(X.columns))

                    else:
                        final_columns_to_use = list(X.columns)

                    if list_columns_to_drop is not None:

                        for l in list_columns_to_drop:
                            if l not in cols_set:
                                raise ValueError("Column %s isn't in the columns of the DataFrame" % l)

                        final_columns_to_use = diff(final_columns_to_use, list_columns_to_drop)

        else:

            if is_int:
                ##########################################
                ### Case 3 : Array + Integer selection ###
                ##########################################
                if self.regex_match:

                    ########################
                    ## Case 3a  : + Regex ##
                    ########################
                    raise ValueError("regex_match can only work with strings 'columns_to_use' not int")

                ########################
                ## Case 3b : no Regex ##
                ########################
                cols_set = set(range(self._expected_nbcols))
                if list_columns_to_use is not None:

                    for l in list_columns_to_use:
                        if l not in cols_set:
                            raise ValueError("Column %d isn't in the columns of the DataFrame" % l)

                    final_columns_to_use = intersect(list_columns_to_use, list(range(self._expected_nbcols)))
                else:
                    final_columns_to_use = list(range(self._expected_nbcols))

                if list_columns_to_drop is not None:

                    for l in list_columns_to_drop:
                        if l not in cols_set:
                            raise ValueError("Column %d isn't in the columns of the DataFrame" % l)

                    final_columns_to_use = diff(final_columns_to_use, list_columns_to_drop)

            else:
                #########################################
                ### Case 4 : Array + String selection ###
                #########################################
                raise ValueError("columns_to_use must be integers when type is array or sparseArray")

        self._columns_to_use_is_integer = is_int
        self._final_columns_to_use = final_columns_to_use

        if self._expected_type in (DataTypes.DataFrame, DataTypes.SparseDataFrame):
            self._Xcolumns = list(X.columns)
        else:
            self._Xcolumns = list(range(self._expected_nbcols))

        ## TODO : here make a simplification into a slice when it is possible

        self._already_fitted = True

        return self

    def transform(self, X):

        self._check_is_fitted()

        Xtype = dsh.get_type(X)
        Xnbcols = dsh._nbcols(X)

        if self._expected_type != Xtype:
            raise ValueError(
                "I don't have the correct type as input, expected : %s, got : %s" % (self._expected_type, Xtype)
            )

        if self._expected_nbcols != Xnbcols:
            raise ValueError(
                "I don't have the correct number of columns, expected : %d, got : %d" % (self._expected_nbcols, Xnbcols)
            )

        if self._expected_type in (DataTypes.DataFrame, DataTypes.SparseDataFrame):
            if self._columns_to_use_is_integer:

                set_col = set(range(X.shape[1]))
                for l in self._final_columns_to_use:
                    if l not in set_col:
                        raise ValueError("Column %d isn't in the column of the DataFrame" % l)

                return X.iloc[:, self._final_columns_to_use]
            else:

                set_col = set(X.columns)
                for l in self._final_columns_to_use:
                    if l not in set_col:
                        raise ValueError("Column %s isn't in the column of the DataFrame" % l)

                return X.loc[:, self._final_columns_to_use]

        else:
            if self._columns_to_use_is_integer:

                set_col = set(range(X.shape[1]))
                for l in self._final_columns_to_use:
                    if l not in set_col:
                        raise ValueError("Column %d isn't in the column of the DataFrame" % l)

                return X[:, self._final_columns_to_use]

            else:
                raise ValueError("columns_to_use must be integers when type if array or sparseArray")

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names(self, input_features=None):
        self._check_is_fitted()

        if self._columns_to_use_is_integer:

            if input_features is None:
                input_features = self._Xcolumns

            return [input_features[c] for c in self._final_columns_to_use]
        else:
            return self._final_columns_to_use

    def _check_is_fitted(self):
        """ raise an error if model isn't fitted yet """
        if not self._already_fitted:
            raise NotFittedError(
                "This %s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method." % type(self).__name__
            )


# In[]


def _concat(*args, sep="__"):
    return sep.join([str(a) for a in args if a != "" and a is not None])


# In[]

def try_to_find_features_names(model, input_features=None):
    # TODO : il faudrait que ca prenne en entree un champs 'input_features_names' a passer a get_features_names
    # TODO : il faut tester si le model accept 'input_features_names'
    # TODO : il faudrait que pour les pipelines ca iter avec 'input_features_names' = get_features_names(last step)

    if hasattr(model, "get_feature_names"):
        # It already has a 'get_feature_names' method
        f = None

        if input_features is not None and function_has_named_argument(model.get_feature_names, "input_features"):
            # I have an input_features argument AND the method accepts it
            # => I'll use it
            try:
                f = model.get_feature_names(input_features)
            except (ValueError, AttributeError):
                pass

        else:

            try:
                f = model.get_feature_names()
            except (ValueError, AttributeError):
                pass

        if f is not None:
            return f

    if hasattr(model, "steps"):
        # It is a pipeline
        last_step = model.steps[-1][1]

        return try_to_find_features_names(last_step, input_features=input_features)

    if hasattr(model, "transformer_list"):

        features = []
        for name, transformer in model.transformer_list:
            fs = try_to_find_features_names(transformer, input_features=input_features)
            if fs is None:
                return None
            features += [name + "__" + f for f in fs]

        return features
        # Rmk : FeatureUnion, already implemented
    else:
        # I don't know

        return None  # don't know


def _tolist(x):
    """ transform to a list or tuple if that is not the case """
    if isinstance(x, (list, tuple)):
        return x
    else:
        return [x]


class DebugPassThrough(TransformerMixin, BaseEstimator):
    """ Dummy transformer that does nothing, used to debug, test or if a step in a pipeline is needed """

    # Useful for test
    def __init__(self, verbose=False, name=None, column_prefix=None, debug=False):
        self.verbose = verbose
        self.name = name
        self.column_prefix = column_prefix
        self.debug = debug

    def fit(self, X, y=None, **fit_params):
        if self.verbose:
            print("within 'DebugPassThrough' fit named %s" % self.name)
            if fit_params:
                print("fit_params given")
                print(fit_params)

        if self.debug:
            self._expected_type = dsh.get_type(X)
            self._expected_nbcols = dsh._nbcols(X)
            if self._expected_type in (dsh.DataTypes.DataFrame, dsh.DataTypes.SparseDataFrame):
                self._expected_columns = list(X.columns)

            self.fit_params = fit_params  # stored, just to help test

        if self.column_prefix is None:
            self._features = getattr(X, "columns", None)
            if self._features is not None:
                self._features = list(self._features)
        else:
            if hasattr(X, "columns"):
                self._features = [self.column_prefix + "_" + c for c in X.columns]
            else:
                self._features = None

        return self

    def transform(self, X):
        if self.verbose:
            print("within 'DebugPassThrought' transform named %s" % self.name)

        Xres = X
        if self.column_prefix is not None:
            Xres = X.copy()
            Xres.columns = [self.column_prefix + "_" + c for c in Xres.columns]

        return Xres

    def fit_transform(self, X, y=None, **fit_params):
        if self.verbose:
            print("withing 'DebugPassThrought' fit_transform named %s" % self.name)
            if fit_params:
                print("fit_params given")
                print(fit_params)

        if self.debug:
            self._expected_type = dsh.get_type(X)
            self._expected_nbcols = dsh._nbcols(X)
            if self._expected_type in (dsh.DataTypes.DataFrame, dsh.DataTypes.SparseDataFrame):
                self._expected_columns = list(X.columns)

            self.fit_params = fit_params  # stored, just to help test

        Xres = X
        if self.column_prefix is not None:
            Xres = X.copy()
            Xres.columns = [self.column_prefix + "_" + c for c in Xres.columns]

        self._features = getattr(Xres, "columns", None)
        if self._features is not None:
            self._features = list(self._features)

        return Xres

    def get_feature_names(self):
        return self._features


class ModelWrapper(TransformerMixin, BaseEstimator):
    """ This is a generic class to help wrapping existing transformer and make them more robust 
    
    Parameters
    ----------
    
    columns_to_use : None or list of string
        this parameters will allow the wrapped transformer to select its columns
        
    work_on_one_column_only : boolean
        if True tells that the underlying transformer works with 1 dimensinal data (pd.Serie for example)
        
    all_columns_at_once : boolean
        if False it tells that the underlying transformer only know how to work one a singular column
        This is the case for sklearn CountVectorizer for example.
        If that is the case the wrapped model will work one several column has well (a clone of the underlying model will be create for each column)
        
    accepted_input_types : list of DataType
        tells what is accepted by the underlying transformer, a conversion will be made if the input type is not among that list
        if None nothing is done
        
    column_prefix : str or None
        if we want the features_names to be prefixed by something like 'SVD_' or 'BAG_' (for TruncatedSVD or CountVectorizer)
        
    desired_output_type : None or DataType
        specify the desired output type of transformer, a conversion will be made if necesary
        
    must_transform_to_get_features_name : boolean
        specify if the transformer should transform its data in order to get its features names.
        Ideally the underlying transformer should implement a  'get_features_names' method but sometimes the features names are only retrieve from the column of the transformed DataFrame
        
    dont_change_columns : boolean
        indicate that the transformer doesn't change the column (for example a StandardScaler)
        if that is the case you know that the resulting feature are the input feature

    keep_other_columns : string, default = 'drop'
        choices : 'keep','drop','delta'.
        If 'keep'  : the original columns are kept     => result = columns + transformed columns
        If 'drop'  : the original columns are dropped  => result = transformed columns
        If 'delta' : only the original columns not used in transformed are kept => result = un-touched original columns + transformed columns
        
    regex_match : boolean, default = False
        if True will use a regex to match columns otherwise exact match

    """

    def __init__(
        self,
        columns_to_use,
        work_on_one_column_only,
        all_columns_at_once,
        accepted_input_types,
        column_prefix,
        desired_output_type,
        must_transform_to_get_features_name,
        dont_change_columns,
        keep_other_columns="drop",
        regex_match=False,
    ):

        ## Underlying transformer/model ##

        ## Where to apply it ##
        self.columns_to_use = columns_to_use  # list of columns to apply the model on
        self.regex_match = regex_match

        ## How to apply it ##
        self.work_on_one_column_only = work_on_one_column_only  # does the model want a 1 dimensional input or not
        self.all_columns_at_once = all_columns_at_once  # do we apply the model on all column or column by column

        ## Input type ##
        self.accepted_input_types = accepted_input_types  # What can be accepted as input

        ## Output ##
        self.column_prefix = column_prefix  # what suffix to put on columns
        self.desired_output_type = desired_output_type  # None, numpy, sparse, DataFrame, SparseDataFrame

        self.must_transform_to_get_features_name = (
            must_transform_to_get_features_name
        )  # if True I'll transform in the fit to get the feature names
        self.dont_change_columns = dont_change_columns

        self.keep_other_columns = keep_other_columns

        self._model = None
        self._models = None
        self.selector = None
        self._already_fitted = False

        self._do_assert = False

    def can_cv_transform(self):
        """ this method tells if a given transformer can be used to return out-sample prediction
        
        If this returns True, a call to cross_validation(self, X , y , return_predict = True, no_scoring = True, method = "transform") will works
        Otherwise it will generate an error
        
        If the model is part of a GraphPipeline it will tell the GraphPipeline object how to cross-validate this node
        
        Method should be overrided if needed
        
        Return
        ------
        boolean, True or False depending on the model

        """
        return False

    @property
    def model(self):
        if self.all_columns_at_once:
            if self._model is None:
                raise ValueError("Model doesn't exist yet, please fit first")
            return self._model

        else:
            if self._models is None:
                raise ValueError("Model doesn't exist yet, please fit first")

            return self._models

    # self.keep_other_columns = keep_other_columns
    # What to do with the other columns
    def _get_model(self, X, y):
        """ method used to delay the creation of the model until after X and y are known
        
        If the model works directly on all columns (all_columns_at_once = True)  , called only once with X = complete data
        If the model works column by columns       (all_columns_at_once = False) , called one type per column with X = one column
        
        """
        raise NotImplementedError("Must be coded in inherited classes")

    def _verif_params(self):
        # Model
        # for m in ("fit","fit_transform","transform"):
        #    if not hasattr(self._model,m):
        #        raise ValueError("model should have a '%s' method" % m)
        for attr in ("work_on_one_column_only", "all_columns_at_once", "dont_change_columns"):
            if not isinstance(getattr(self, attr), bool):
                raise TypeError("%s should be boolean" % attr)

        if self.accepted_input_types is not None:
            if not isinstance(self.accepted_input_types, (tuple, list)):
                raise TypeError("accepted_input_types should be a list or tuple")

            for t in self.accepted_input_types:
                if t not in DataTypes.alls:
                    raise ValueError("accepted_input_types should be within DataTypes, instead I got '%s'" % t)

    def _get_rest_columns(self, input_features=None):
        """ retrieve the name of the feature for the untransformed part on the data """

        if self.keep_other_columns == "keep":
            if input_features is None:
                return self._Xcolumns
            else:
                return input_features

        elif self.keep_other_columns == "drop":
            return []

        else:
            if self.columns_to_use is None:
                return []
            else:
                return self.anti_selector.get_feature_names(input_features)

    def _fit_transform_rest(self, X, transformed_part, is_fit, is_transform):
        """ method to take care of the rest of data, that wasn't transformed,
        it can either be 
        * dropped (default) :  'keep_other_columns' == 'drop'
        * kept as is        :  'keep_other_columns' == 'keep'
        * keep only not used columns 'keep_other_columns' == 'delta'

        """
        if self.keep_other_columns == "keep":
            # In that case I'll keep the original columns as well
            if is_fit:
                if hasattr(X, "columns"):
                    self._Xcolumns = list(getattr(X, "columns"))
                elif hasattr(X, "shape"):
                    self._Xcolumns = [i for i in range(X.shape[1])]
                else:
                    self._Xcolumns = None

            if is_transform:
                kept_features_names = self._get_rest_columns()
                Xcomplete_result = dsh.generic_hstack(
                    [X, transformed_part],
                    output_type=self.desired_output_type,
                    all_columns_names=[kept_features_names, self._feature_names_for_transform],
                )

                return Xcomplete_result
            else:
                return self

        elif self.keep_other_columns == "drop":

            return None

        # "delta' mode, I'll keep only the columns that were not used
        if self.columns_to_use is None:
            return transformed_part

        if is_fit and is_transform:
            self.anti_selector = ColumnsSelector(columns_to_drop=self.columns_to_use, regex_match=self.regex_match)
            Xother = self.anti_selector.fit_transform(X)

        elif is_transform:
            Xother = self.anti_selector.transform(X)

        elif is_fit:
            self.anti_selector = ColumnsSelector(columns_to_drop=self.columns_to_use, regex_match=self.regex_match)
            self.anti_selector.fit(X)

        if is_transform:
            kept_features_names = self._get_rest_columns()
            return dsh.generic_hstack(
                [Xother, transformed_part],
                output_type=self.desired_output_type,
                all_columns_names=[kept_features_names, self._feature_names_for_transform],
            )

            # Rmk : generic_hstack will handle the case where Xother has no columns
        else:
            return self

    def _check_is_fitted(self):
        """ raise an error if model isn't fitted yet """
        if not self._already_fitted:
            raise NotFittedError(
                "This %s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method." % type(self).__name__
            )

    @exception_improved_logging
    def fit(self, X, y=None, **fit_params):
        self._fit_transform(X, y, is_fit=True, is_transform=False, fit_params=fit_params)

        if self.keep_other_columns in ("keep", "delta"):
            self._fit_transform_rest(X, transformed_part=None, is_fit=True, is_transform=False)

        self._already_fitted = True
        return self

    @exception_improved_logging
    def transform(self, X):

        self._check_is_fitted()

        transformed_part = self._fit_transform(X, y=None, is_fit=False, is_transform=True)

        if self.keep_other_columns in ("keep", "delta"):
            return self._fit_transform_rest(X, transformed_part=transformed_part, is_fit=False, is_transform=True)
        else:
            return transformed_part

    @exception_improved_logging
    def fit_transform(self, X, y=None, **fit_params):
        transformed_part = self._fit_transform(X, y=y, is_fit=True, is_transform=True, fit_params=fit_params)

        if self.keep_other_columns in ("keep", "delta"):
            result = self._fit_transform_rest(X, transformed_part=transformed_part, is_fit=True, is_transform=True)
        else:
            result = transformed_part

        self._already_fitted = True
        return result

    # TODO : inverse transform
    def _fit_transform(self, X, y, is_fit, is_transform, fit_params=None):
        """ internal method that handle the fit and the transform """
        if fit_params is None:
            fit_params = {}

        if is_fit:
            if self.columns_to_use is not None:
                self.selector = ColumnsSelector(columns_to_use=self.columns_to_use, regex_match=self.regex_match)

        if hasattr(X, "shape"):
            if X.shape[0] == 0:
                raise ValueError("the X object has 0 rows")

        Xindex = dsh._get_index(X)  # if X has an index retrieve it
        if self.columns_to_use is not None:
            if is_fit:
                Xsubset = self.selector.fit_transform(X)
            else:
                Xsubset = self.selector.transform(X)
        else:
            Xsubset = X

        # TODO : here allow a preprocessing pipeline
        #        if self.has_preprocessing:
        #            if is_fit:
        #                self.preprocessing = self._get_preprocessing()
        #                Xsubset = self.preprocessing.fit_transform(Xsubset)
        #            else:
        #                Xsubset = self.preprocessing.transform(Xsubset)

        # Store columns and shape BEFORE any modification
        if self.selector is not None:
            Xsubset_columns = self.selector.get_feature_names()
        else:
            Xsubset_columns = getattr(Xsubset, "columns", None)

        Xsubset_shape = getattr(Xsubset, "shape", None)
        # TODO : ici utiliser d'une facon ou d'une autre un '
        # https://github.com/scikit-learn/scikit-learn/issues/6425

        if is_fit:
            self._expected_type = dsh.get_type(Xsubset)
            self._expected_nbcols = dsh._nbcols(Xsubset)
            self._expected_columns = dsh._get_columns(Xsubset)

        else:
            Xtype = dsh.get_type(Xsubset)
            if Xtype != self._expected_type:
                raise ValueError(
                    "I don't have the correct type as input, expected : %s, got : %s" % (self._expected_type, Xtype)
                )

            nbcols = dsh._nbcols(Xsubset)
            if nbcols != self._expected_nbcols:
                raise ValueError(
                    "I don't have the correct nb of colmns as input, expected : %d, got : %d"
                    % (self._expected_nbcols, nbcols)
                )

            columns = dsh._get_columns(Xsubset)
            expected_columns = getattr(self, "_expected_columns",None) # to allow pickle compatibility
            
            if expected_columns is not None and columns is not None and columns != self._expected_columns:
                raise ValueError(
                    "I don't have the correct names of columns"
                )
                

        if self.accepted_input_types is not None and self._expected_type not in self.accepted_input_types:
            Xsubset = dsh.convert_generic(
                Xsubset, mapped_type=self._expected_type, output_type=self.accepted_input_types[0]
            )

        if is_fit:
            self._verif_params()

        if self.all_columns_at_once:

            if is_fit:
                self._model = self._get_model(Xsubset, y)

            ##############################################
            ### Apply the model on ALL columns at ONCE ###
            ##############################################

            if self.work_on_one_column_only:
                Xsubset = dsh.make1dimension(Xsubset)  # will generate an error if 2 dimensions
            else:
                Xsubset = dsh.make2dimensions(Xsubset)

            Xres = None
            if is_fit and is_transform:
                # fit_transform method
                Xres = self._model.fit_transform(Xsubset, y, **fit_params)

            elif is_fit and not is_transform:
                # fit method
                if self.must_transform_to_get_features_name:
                    Xres = self._model.fit_transform(Xsubset, y, **fit_params)
                    # Peut etre que je vais forcer a faire le fit_transform a chaque fois...
                    # pour savoir la taille
                else:
                    self._model.fit(Xsubset, y, **fit_params)
            else:
                # tansform
                Xres = self._model.transform(Xsubset)

            if is_fit:
                self._columns_informations = {
                    "output_columns": getattr(Xres, "columns", None),  # names of transformed columns if exist
                    "output_shape": getattr(Xres, "shape", None),  # shape of transformed result if exist
                    "input_columns": Xsubset_columns,  # name of input columns
                    "input_shape": Xsubset_shape,  # shape of input data
                }

                self._feature_names_for_transform = self.try_to_find_feature_names_all_at_once(
                    output_columns=self._columns_informations["output_columns"],
                    output_shape=self._columns_informations["output_shape"],
                    input_columns=self._columns_informations["input_columns"],
                    input_shape=self._columns_informations["input_shape"],
                )

                self.kept_features_names = None  # for now

            if is_transform:
                Xres = dsh.convert_generic(Xres, output_type=self.desired_output_type)
                Xres = dsh._set_index(Xres, Xindex)

        else:
            ########################################
            ### Apply the model COLUMN BY COLUMN ###
            ########################################
            if is_fit:
                self._models = []

            if is_transform or self.must_transform_to_get_features_name:
                all_Xres = []
            else:
                all_Xres = None

            Xsubset = dsh.make2dimensions(Xsubset)

            for j in range(self._expected_nbcols):

                if self._expected_type in (DataTypes.DataFrame, DataTypes.SparseDataFrame, DataTypes.Serie):
                    Xsubset_j = Xsubset.iloc[:, j]
                else:
                    Xsubset_j = Xsubset[:, j]

                if is_fit:
                    sub_model = self._get_model(Xsubset, y)
                    self._models.append(sub_model)
                else:
                    sub_model = self._models[j]

                if not self.work_on_one_column_only:
                    Xsubset_j = dsh.make2dimensions(Xsubset_j)

                if is_fit and is_transform:
                    # fit_transform method
                    Xres_j = sub_model.fit_transform(Xsubset_j, y, **fit_params)

                    all_Xres.append(Xres_j)

                elif is_fit and not is_transform:
                    # fit method
                    if self.must_transform_to_get_features_name:
                        Xres_j = sub_model.fit_transform(Xsubset_j, y, **fit_params)
                        all_Xres.append(Xres_j)

                    else:
                        sub_model.fit(Xsubset_j, y, **fit_params)

                elif is_transform:
                    # transform method

                    Xres_j = sub_model.transform(Xsubset_j)
                    all_Xres.append(Xres_j)

            if is_fit:

                self._columns_informations = {
                    "all_output_columns": None
                    if all_Xres is None
                    else [getattr(Xres, "columns", None) for Xres in all_Xres],
                    "all_output_shape": None
                    if all_Xres is None
                    else [getattr(Xres, "shape", None) for Xres in all_Xres],
                    "input_columns": Xsubset_columns,  # name of input columns
                    "input_shape": Xsubset_shape,  # shape of input data
                }

                self._feature_names_for_transform = list(
                    self.try_to_find_feature_names_separate(
                        all_output_columns=self._columns_informations["all_output_columns"],
                        all_output_shape=self._columns_informations["all_output_shape"],
                        input_columns=self._columns_informations["input_columns"],
                        input_shape=self._columns_informations["input_shape"],
                    )
                )

                self.kept_features_names = None  # for now

            if is_transform:
                Xres = dsh.generic_hstack(all_Xres, output_type=self.desired_output_type)
                Xres = dsh._set_index(Xres, Xindex)

        if is_transform:
            if self._feature_names_for_transform is not None:
                ### LA ca marche pas en transform !!!
                Xres = dsh._set_columns(Xres, self._feature_names_for_transform)

        if is_transform:
            return Xres
        else:
            return self

    @exception_improved_logging
    def get_feature_names(self, input_features=None):

        self._check_is_fitted()

        if self.all_columns_at_once:

            if input_features is None:
                input_columns = self._columns_informations["input_columns"]

            elif self.columns_to_use is not None:
                input_columns = self.selector.get_feature_names(input_features)

            else:
                input_columns = input_features

            feature_names = self.try_to_find_feature_names_all_at_once(
                output_columns=self._columns_informations["output_columns"],
                output_shape=self._columns_informations["output_shape"],
                input_columns=input_columns,
                input_shape=self._columns_informations["input_shape"],
            )

        else:

            if input_features is None:
                input_columns = self._columns_informations["input_columns"]

            elif self.columns_to_use is not None:
                input_columns = self.selector.get_feature_names(input_features)

            else:
                input_columns = input_features

            feature_names = self.try_to_find_feature_names_separate(
                all_output_columns=self._columns_informations["all_output_columns"],
                all_output_shape=self._columns_informations["all_output_shape"],
                input_columns=input_columns,
                input_shape=self._columns_informations["input_shape"],
            )

        if feature_names is None:
            raise ValueError("I can't find features names")

        kept_features_names = self._get_rest_columns(input_features=input_features)

        return list(kept_features_names) + list(feature_names)

    def try_to_find_feature_names_all_at_once(
        self, output_columns=None, output_shape=None, input_columns=None, input_shape=None
    ):

        ##############################################
        ### Apply the model on ALL columns at once ###
        ##############################################
        def nbcols_from_shape(s):
            if len(s) == 1:
                return 1
            else:
                return s[1]

        # 1) use get_features_names
        features = try_to_find_features_names(self._model, input_features=input_columns)

        # 2) if not found... I'll try to read it from Xres
        if features is None:

            # columns_res   = getattr(Xres , "columns", None)     # Rmk : if no columns or Xres None => return None

            # 1) I can use the column for the result
            if output_columns is not None:
                features = output_columns

            # 2) I know the transformer doesn't change its columns : I can use the column of the input
            if features is None:
                if self.dont_change_columns and input_columns is not None:
                    features = input_columns

            if features is None:
                # Otherwise, I'll look at the shape

                def temp(s):
                    if len(s) == 1:
                        return 1
                    else:
                        return s[1]

                # 3) look at shape of result
                if output_shape is not None:
                    features = list(range(temp(output_shape)))

                # 4) look at shape of input
                if features is None:
                    if input_shape is not None and self.dont_change_columns:
                        features = list(range(temp(input_shape)))

                # => default names here

        if features is None:
            # Cant do anything
            return None

        if self.column_prefix is not None:
            # Add the prefix
            features = [_concat(self.column_prefix, str(c), sep="__") for c in features]

        return list(features)

    def try_to_find_feature_names_separate(
        self, all_output_columns=None, all_output_shape=None, input_columns=None, input_shape=None
    ):

        #                                           all_Xres = None,
        #                                           Xsubset_columns = None,
        #                                           Xsubset_shape = None):
        ########################################
        ### Apply the model COLUMN BY COLUMN ###
        ########################################
        def any_none(ll):
            if ll is None:
                return True
            return any([l is None for l in ll])

        def nbcols_from_shape(s):
            if len(s) == 1:
                return 1
            else:
                return s[1]

        # 1) use get_feature_names
        all_features = [try_to_find_features_names(submodel, input_features=input_columns) for submodel in self._models]

        # 2) if not found.. I'll try to read it elsewhere..
        if any_none(all_features):

            if not any_none(all_output_columns):
                all_features = all_output_columns

            if any_none(all_features) and self.dont_change_columns and input_columns is not None:
                all_features = [[c] for c in input_columns]

            if any_none(all_features) and not any_none(all_output_shape):
                all_features = [list(range(nbcols_from_shape(s))) for s in input_shape]

            if any_none(all_features) and self.dont_change_columns and input_shape is not None:
                all_features = [[i] for i in range(nbcols_from_shape(input_shape))]

        if any_none(all_features):
            return None

        if input_columns is None:

            if input_shape is None:
                input_columns = list(range(nbcols_from_shape(input_shape)))

            if input_columns is None:
                if all_output_columns is not None:
                    input_columns = list(range(len(all_output_columns)))

        if input_columns is None:
            return None

        final_features = []
        if self.column_prefix is not None:
            for col, features in zip(input_columns, all_features):
                final_features += [_concat(col, self.column_prefix, f, sep="__") for f in features]

        return final_features


# In[]
