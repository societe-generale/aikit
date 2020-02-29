# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 13:50:25 2018

@author: Lionel Massoulard
"""


import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_random_state

from aikit.tools.db_informations import guess_type_of_variable
from aikit.enums import TypeOfVariables, DataTypes
from aikit.tools.data_structure_helper import get_type, generic_hstack, get_rid_of_categories
from aikit.tools.helper_functions import diff

from aikit.cross_validation import create_cv

from aikit.transformers.model_wrapper import ModelWrapper


# In[]


class _TargetEncoderBase(TransformerMixin, BaseEstimator):
    """ Class to encode categorical value using the target 
    
    Parameters
    ----------    
    max_na_percentage : float, default = 0.05
        if more than 'max_na_percentage' None within a column, None will be treated as a special modality, otherwise it will default to the global aggregat
        
    smoothing_min : float, default = 1
        handle the prior weight, see formula bellow
        
    smoothing_value : float, default = 10
        handle the *speed* with which the prior is forgotten (see formula bellow)
    
    noise_level : float or None, default = None
        degree of noise to add within the fit_transform
        
    cv : int, None, or CV object, default = 10
        the cv to use within fit_transform


    Those parameters handles the prior weight, WEIGHT = 1/[1 + EXP( - (nb - smoothing_min) / smoothing_value ) ] where 'nb' is the number of observations of the corresponding modality
    
    
    """

    is_regression = None  # should be used in inherited classes

    def __init__(
        self, max_na_percentage=0.05, smoothing_min=1, smoothing_value=10, noise_level=None, cv=10, random_state=None
    ):
        self.max_na_percentage = max_na_percentage

        self.noise_level = noise_level

        self.smoothing_min = smoothing_min
        self.smoothing_value = smoothing_value

        self.cv = cv

        self.random_state = random_state

    @classmethod
    def _get_output_column_name(cls, col, target_classes):
        """ return the name of the created column 
        
        Parameters
        ----------
        col : str
            name of the input column
            
        target_classes : array like 
            modalities of y
            
        Returns:
        --------
        list of str, name of the new features for a given column
        
            
        """
        raise NotImplementedError("Should be implemented in inherited classes")

        # Example : return "%s__target_%s" % (col,str(target))

    #

    def aggregating_function(self, subY, target_classes, noise_level=None):
        """ aggregating function
        
        Parameters
        ----------
        subY : array like
            target on which we will compute an aggregat
            
        target_classes : like of y modalities
            result will be aligned on those modalities
            
        noise_level : None or float
            if not None, will add a uniform number between 0 and noise_level to each observation
            
        col : None or str
            the name of the column we are aggregating (used to set the name)
            
        Returns
        -------
            np.array of shape (len(self.target_classes) , )
            
        """
        raise NotImplementedError("Should be implemented in inherited classes")

    def smoothing(self, nb):
        """ weight function, the more observations the higher the weight
        
        Parameters
        ----------
        nb : float/int
            number of observation
        """
        if self.smoothing_value == 0:
            return 1.0
        else:
            return 1 / (1 + np.exp(-(nb - self.smoothing_min) / self.smoothing_value))

    @staticmethod
    def na_remplacing(serie):
        """ remplace None with '_missing_' to make it a modality """

        ii_null = serie.isnull()

        if not ii_null.any():
            return serie

        result_serie = serie.copy()
        result_serie[ii_null] = "_missing_"

        return result_serie

    def fit(self, X, y):

        if y is None:
            raise ValueError("I need a value for 'y'")

        self._random_gen = check_random_state(self.random_state)

        Xtype = get_type(X)
        if Xtype != DataTypes.DataFrame:
            raise TypeError("X should be a DataFrame")
        Xcolumns = list(X.columns)

        if not isinstance(y, pd.Series):
            sy = pd.Series(y)
        else:
            sy = y

        # Columns to encode and to keep

        self._columns_to_encode = list(X.columns)

        X = get_rid_of_categories(X)

        # Verif:
        if not isinstance(self._columns_to_encode, list):
            raise TypeError("_columns_to_encode should be a list")

        for c in self._columns_to_encode:
            if c not in Xcolumns:
                raise ValueError("column %s isn't in the DataFrame" % c)

        self._columns_to_keep = []

        # Verif:
        if not isinstance(self._columns_to_keep, list):
            raise TypeError("_columns_to_keep should be a list")

        for c in self._columns_to_keep:
            if c not in Xcolumns:
                raise ValueError("column %s isn't in the DataFrame" % c)

        # Target information
        if self.is_regression:

            self.target_classes = None  # No target classes for Regressor
            self.global_std = np.std(sy)

        else:
            # For classification I need to store it
            self.global_std = None
            self.target_classes = list(np.unique(sy))

            if len(self.target_classes) == 2:
                self.target_classes = self.target_classes[1:]

        # Columns on which we want None to be a special modality
        self._na_to_null = dict()
        for col in self._columns_to_encode:
            ii_null = X[col].isnull()
            self._na_to_null[col] = ii_null.sum() >= self.max_na_percentage * len(X)

        self._target_aggregat, self._target_aggregat_global = self._fit_aggregat(X, sy, noise_level=None)

        # Features names
        self._feature_names = [c for c in self._columns_to_keep]  # copy
        for col in self._columns_to_encode:
            self._feature_names += self._get_output_column_name(col=col, target_classes=self.target_classes)
            # self._feature_names += ["%s__target_%s" % (col,str(t)) for t in self.target_classes]

        return self

    def fit_transform(self, X, y):

        if y is None:
            raise ValueError("I need a value for 'y'")

        if not isinstance(y, pd.Series):
            sy = pd.Series(y)
        else:
            sy = y

        self.fit(X, sy)

        X = get_rid_of_categories(X)

        if self.cv is None:  # No Cross Validation ...
            target_aggregat, target_aggregat_global = self._fit_aggregat(X, y, noise_level=self.noise_level)
            all_results = self._transform_aggregat(X, target_aggregat, target_aggregat_global)

        else:
            cv = create_cv(self.cv, y=sy, classifier=not self.is_regression, random_state=123)

            all_results = []
            for train, test in cv.split(X, y):
                target_aggregat, target_aggregat_global = self._fit_aggregat(
                    X.iloc[train, :], sy.iloc[train], noise_level=self.noise_level
                )

                sub_result = self._transform_aggregat(X.iloc[test, :], target_aggregat, target_aggregat_global)

                all_results.append(sub_result)

            all_results = pd.concat(all_results, axis=0)
            all_results = all_results.loc[X.index, :]

            assert len(all_results) == len(X)
            assert (all_results.index == X.index).all()
            assert all_results.shape[1] == len(self.get_feature_names())

        return all_results

    def transform(self, X):

        if get_type(X) != DataTypes.DataFrame:
            raise TypeError("X should be a DataFrame")
        X = get_rid_of_categories(X)

        result = self._transform_aggregat(X, self._target_aggregat, self._target_aggregat_global)
        assert result.shape[1] == len(self.get_feature_names())

        return result

    def _fit_aggregat(self, X, y, noise_level):
        """ compute the aggregat by variable and modality """

        aggregat_global = self.aggregating_function(y, target_classes=self.target_classes, noise_level=noise_level)

        target_aggregat = dict()
        target_aggregat_global = dict()

        for col in self._columns_to_encode:
            index = self._get_output_column_name(col=col, target_classes=self.target_classes)

            target_aggregat[col] = dict()
            target_aggregat_global[col] = pd.Series(aggregat_global, index=index)
            # I put it column by column to have the same index to prevent mistake when joining Serie
            # It will also be usefull when I have 'level2' interaction whene I might have different default value for different modality

            if self._na_to_null[col]:
                Xcol = self.na_remplacing(X[col])
            else:
                Xcol = X[col]

            gps = pd.Series(y).groupby(Xcol)
            for group_name, subY in gps:

                aggregat = self.aggregating_function(subY, target_classes=self.target_classes, noise_level=noise_level)

                weight = self.smoothing(nb=len(subY))
                prior = target_aggregat_global[col]

                target_aggregat[col][group_name] = pd.Series(weight * aggregat + (1 - weight) * prior, index=index)

        return target_aggregat, target_aggregat_global

    @staticmethod
    def get_value(x, target, default):
        if pd.isnull(x):
            return default

        try:
            res = target[x]
        except KeyError:
            res = default
        return res

    def _transform_aggregat(self, X, target_aggregat, target_aggregat_global):

        all_results = []
        for col in self._columns_to_encode:

            if self._na_to_null[col]:
                Xcol = self.na_remplacing(X[col])
            else:
                Xcol = X[col]

            result = Xcol.apply(lambda x: self.get_value(x, target_aggregat[col], target_aggregat_global[col]))
            # result.columns = ["%s__%s" % (col,c) for c in result.columns]
            all_results.append(result)

            assert len(result) == len(X)
            assert len(result.shape) == 2

        if len(all_results) == 0:
            if len(self._columns_to_keep) > 0:
                result_other = X.loc[:, self._columns_to_keep]
                return result_other
            else:
                return pd.DataFrame(index=range(X.shape[0]), columns=[])  # empty DataFrame

        all_results = pd.concat(all_results, axis=1)

        assert (all_results.index == X.index).all()

        if len(self._columns_to_keep) > 0:
            result_other = X.loc[:, self._columns_to_keep]
            return generic_hstack([result_other, all_results])
        else:
            return all_results

    def get_feature_names(self):
        return self._feature_names


class _TargetEncoderClassifier(_TargetEncoderBase):
    __doc__ = _TargetEncoderBase.__doc__

    is_regression = False

    @classmethod
    def _get_output_column_name(cls, col, target_classes):
        return ["%s__target_%s" % (col, str(target)) for target in target_classes]

    def aggregating_function(self, subY, target_classes, noise_level=None, col=None):
        """ aggregating function
        
        Parameters
        ----------
        subY : array like
            target on which we will compute an aggregat
            
        target_classes : like of y modalities
            result will be aligned on those modalities
            
        noise_level : None or float
            if not None, will add a uniform number between 0 and noise_level to each observation
            
        col : None or str
            the name of the column we are aggregating (used to set the name)
            
        Returns
        -------
            pd.Serie, index = target_classes
            value = probability of each classes
            
        """
        #            return len(sub_y
        result = pd.Series(subY).value_counts()

        if noise_level is not None:
            result += self._random_gen.rand(len(result)) * noise_level * 100

        result = result / result.sum()

        complete_result = pd.Series(index=target_classes, data=0)
        index_intersect = np.intersect1d(complete_result.index, result.index)

        complete_result.loc[index_intersect] = result.loc[index_intersect]

        return complete_result.values


class _TargetEncoderRegressor(_TargetEncoderBase):
    is_regression = True
    __doc__ = _TargetEncoderBase.__doc__

    @classmethod
    def _get_output_column_name(cls, col, target_classes):
        return ["%s__target_mean" % col]

    def aggregating_function(self, subY, target_classes, noise_level=None, col=None):
        result = np.mean(subY)
        if noise_level is not None:
            result += self._random_gen.randn(1)[0] * self.global_std * self.noise_level

        return np.array([result])


class _TargetEncoderEntropyClassifier(_TargetEncoderBase):

    __doc__ = _TargetEncoderBase.__doc__

    is_regression = False

    @classmethod
    def _get_output_column_name(cls, col, target_classes):
        return ["%s__target_entropy" % col]

    def aggregating_function(self, subY, target_classes, noise_level=None):
        result = pd.Series(subY).value_counts()

        if noise_level is not None:
            result += self._random_gen.rand(len(result)) * noise_level * 100

        pi = result / result.sum()

        pi_logpi = pi * np.log(pi)
        pi_logpi[pi == 0] = 0

        return np.array([-pi_logpi.sum()])


# In[]


class TargetEncoderClassifier(ModelWrapper):

    __doc__ = _TargetEncoderClassifier.__doc__
    # TODO : concat doc with wrapper

    def __init__(
        self,
        columns_to_use=TypeOfVariables.CAT,
        max_na_percentage=0.05,
        smoothing_min=1,
        smoothing_value=10,
        noise_level=None,
        cv=10,
        random_state=None,
        regex_match=False,
        desired_output_type=DataTypes.DataFrame,
        drop_used_columns=True,
        drop_unused_columns=False,
    ):

        self.max_na_percentage = max_na_percentage

        self.smoothing_min = smoothing_min
        self.smoothing_value = smoothing_value

        self.noise_level = noise_level
        self.cv = cv

        self.random_state = random_state

        #        self.columns_to_use = columns_to_use
        #        self.regex_match = regex_match

        super(TargetEncoderClassifier, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=regex_match,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=(DataTypes.DataFrame,),
            column_prefix=None,
            desired_output_type=desired_output_type,
            must_transform_to_get_features_name=False,
            dont_change_columns=False,
            drop_used_columns=drop_used_columns,
            drop_unused_columns=drop_unused_columns,
        )

    def _get_model(self, X, y=None):
        return _TargetEncoderClassifier(
            max_na_percentage=self.max_na_percentage,
            noise_level=self.noise_level,
            smoothing_min=self.smoothing_min,
            smoothing_value=self.smoothing_value,
            cv=self.cv,
            random_state=self.random_state,
        )

    def can_cv_transform(self):
        """ this method tells if a given transformer can be used to return out-sample prediction
        
        If this returns True, a call to approx_cross_validation(self, X , y , return_predict = True, no_scoring = True, method = "transform") will works
        Otherwise it will generate an error
        
        If the model is part of a GraphPipeline it will tell the GraphPipeline object how to cross-validate this node
        
        Method should be overrided if needed
        
        Return
        ------
        boolean, True or False depending on the model

        """
        return True


class TargetEncoderEntropyClassifier(ModelWrapper):

    __doc__ = _TargetEncoderEntropyClassifier.__doc__

    def __init__(
        self,
        columns_to_use=TypeOfVariables.CAT,
        max_na_percentage=0.05,
        smoothing_min=1,
        smoothing_value=10,
        noise_level=None,
        cv=10,
        random_state=None,
        regex_match=False,
        desired_output_type=DataTypes.DataFrame,
        drop_used_columns=True,
        drop_unused_columns=False,
    ):

        self.columns_to_use = columns_to_use

        self.max_na_percentage = max_na_percentage

        self.smoothing_min = smoothing_min
        self.smoothing_value = smoothing_value

        self.noise_level = noise_level
        self.cv = cv

        self.random_state = random_state

        super(TargetEncoderEntropyClassifier, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=regex_match,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=(DataTypes.DataFrame,),
            column_prefix=None,
            desired_output_type=desired_output_type,
            must_transform_to_get_features_name=False,
            dont_change_columns=False,
            drop_used_columns=drop_used_columns,
            drop_unused_columns=drop_unused_columns,
        )

    def _get_model(self, X, y=None):
        return _TargetEncoderEntropyClassifier(
            max_na_percentage=self.max_na_percentage,
            noise_level=self.noise_level,
            smoothing_min=self.smoothing_min,
            smoothing_value=self.smoothing_value,
            cv=self.cv,
            random_state=self.random_state,
        )

    def can_cv_transform(self):
        """ this method tells if a given transformer can be used to return out-sample prediction
        
        If this returns True, a call to approx_cross_validation(self, X , y , return_predict = True, no_scoring = True, method = "transform") will works
        Otherwise it will generate an error
        
        If the model is part of a GraphPipeline it will tell the GraphPipeline object how to cross-validate this node
        
        Method should be overrided if needed
        
        Return
        ------
        boolean, True or False depending on the model

        """
        return True


class TargetEncoderRegressor(ModelWrapper):

    __doc__ = _TargetEncoderRegressor.__doc__

    def __init__(
        self,
        columns_to_use=TypeOfVariables.CAT,
        max_na_percentage=0.05,
        smoothing_min=1,
        smoothing_value=10,
        noise_level=None,
        cv=10,
        random_state=None,
        regex_match=False,
        desired_output_type=DataTypes.DataFrame,
        drop_used_columns=True,
        drop_unused_columns=False,
    ):

        self.columns_to_use = columns_to_use
        self.max_na_percentage = max_na_percentage

        self.smoothing_min = smoothing_min
        self.smoothing_value = smoothing_value

        self.noise_level = noise_level
        self.cv = cv

        self.random_state = random_state

        super(TargetEncoderRegressor, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=regex_match,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=(DataTypes.DataFrame,),
            column_prefix=None,
            desired_output_type=desired_output_type,
            must_transform_to_get_features_name=False,
            dont_change_columns=False,
            drop_used_columns=drop_used_columns,
            drop_unused_columns=drop_unused_columns,
        )

    def _get_model(self, X, y=None):
        return _TargetEncoderRegressor(
            max_na_percentage=self.max_na_percentage,
            noise_level=self.noise_level,
            smoothing_min=self.smoothing_min,
            smoothing_value=self.smoothing_value,
            cv=self.cv,
            random_state=self.random_state,
        )

    def can_cv_transform(self):
        """ this method tells if a given transformer can be used to return out-sample prediction
        
        If this returns True, a call to approx_cross_validation(self, X , y , return_predict = True, no_scoring = True, method = "transform") will works
        Otherwise it will generate an error
        
        If the model is part of a GraphPipeline it will tell the GraphPipeline object how to cross-validate this node
        
        Method should be overrided if needed
        
        Return
        ------
        boolean, True or False depending on the model

        """
        return True


# In[]
