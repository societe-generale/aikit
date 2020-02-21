# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:47:48 2018

@author: Lionel Massoulard
"""
import numpy as np
import pandas as pd

import scipy.sparse as sps

import scipy.stats
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from scipy.interpolate import interp1d

from collections import OrderedDict

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics.scorer import _BaseScorer, _PredictScorer


from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sklearn.feature_selection


from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.cluster import KMeans
import sklearn.metrics.scorer

# from aikit.helper_functions import is_user
from aikit.enums import DataTypes
from aikit.transformers.model_wrapper import ModelWrapper, ColumnsSelector
from aikit.tools.data_structure_helper import _nbcols, _nbrows, get_type, generic_hstack, make2dimensions, convert_to_array

from aikit.cross_validation import cross_validation


assert ColumnsSelector  # Trick to shutup python warning 'imported but unused'
# I import ColumnsSelector to have it in this namespace as well


# In[]


def int_n_components(nbcolumns, n_components):
    """ convert n_component into a number of component

    Parameters
    ----------
    nbcolumns : int
        number of columns in X

    n_component : int or float
        setted number of columns to convert to int
    """
    if n_components < 1 or (isinstance(n_components, float) and n_components == 1.0):
        n_components = min(max(int(nbcolumns * n_components), 1), nbcolumns - 1)
    else:
        n_components = min(max(int(n_components), 1), nbcolumns - 1)

    return n_components


def f_forest_regression(X, y, rf_params=None):
    """ return features importances for classification problems based on RandomForest """
    if rf_params is None:
        rf_params = {"n_estimators": 100, "random_state":123}

    forest = RandomForestRegressor(**rf_params)
    forest.fit(X, y)
    return forest.feature_importances_


def f_forest_classification(X, y, rf_params=None):
    """ return features importances for regression problems based on RandomForest """

    if rf_params is None:
        rf_params = {"n_estimators": 100, "random_state":123}

    forest = RandomForestClassifier(**rf_params)
    forest.fit(X, y)
    return forest.feature_importances_


def f_linear_regression(X, y, ridge_params=None):
    """ return features importances for regression problems based on RidgeRegression """
    if ridge_params is None:
        ridge_params = {}

    scaler = StandardScaler(
        with_mean=False
    )  # with_mean = False : so that it works with sparse matrix + I don't need it anyway
    ridge = Ridge(**ridge_params)

    ridge.fit(scaler.fit_transform(X), y)

    if ridge.coef_.ndim == 1:
        return np.abs(ridge.coef_)
    else:
        return np.sum(np.abs(ridge.coef_), axis=0)


def f_linear_classification(X, y, logit_params=None):
    """ return features importances for classification problems based on LogisticRegression """

    if logit_params is None:
        logit_params = {}

    scaler = StandardScaler(
        with_mean=False
    )  # with_mean = False : so that it works with sparse matrix + I don't need it anyway
    logit = LogisticRegression(**logit_params)
    logit.fit(scaler.fit_transform(X), y)

    if logit.coef_.ndim == 1:
        return np.abs(logit.coef_)
    else:
        return np.sum(np.abs(logit.coef_), axis=0)


class _BaseFeaturesSelector(BaseEstimator, TransformerMixin):
    """Features Selection based on RandomForest, LinearModel or Correlation.

    Parameters
    ----------
    n_components : int or float, default = 0.5
        number of component to keep, if float interpreted as a percentage of X size

    component_selection : str, default = "number"
        if "number" : will select the first 'n_components' features
        if "elbow"  : will use a tweaked 'elbow' rule to select the number of features

    selector_type : string, default = 'forest'
        'default' : using sklearn f_regression/f_classification
        'forest'  : using RandomForest features importances
        'linear'  : using Ridge/LogisticRegression coefficient

    random_state : int, default = None

    model_params :
        Model hyper parameters
    """

    is_regression = None

    def __init__(
        self,
        n_components=0.5,
        component_selection="number",
        selector_type="forest",
        random_state=None,
        model_params=None,
    ):

        self.n_components = n_components
        self.component_selection = component_selection
        self.selector_type = selector_type
        self.random_state = random_state

        self.model_params = model_params

        self._already_fitted = False

    def _check_is_fitted(self):
        """ raise an error if model isn't fitted yet """
        if not self._already_fitted:
            raise NotFittedError(
                "This %s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method." % type(self).__name__
            )

    def fit(self, X, y):

        # shape of X
        self._Xnbcolumns = X.shape[1]
        n_components = int_n_components(self._Xnbcolumns, self.n_components)

        # is regression
        if self.is_regression is None:
            is_regression = type_of_target(y) not in ["binary", "multiclass"]
        else:
            is_regression = self.is_regression

        if self.selector_type not in ["forest", "linear", "default"]:
            raise ValueError("selector_type should be 'forest','linear' or 'default'")

        # function to retrieve the importances
        if self.selector_type == "forest" and is_regression:

            rf_params = self.model_params
            if rf_params is None:
                rf_params = {"n_estimators": 100}

            if self.random_state is not None:
                rf_params["random_state"] = self.random_state

            features_importances = f_forest_regression(X, y, rf_params=rf_params)

        elif self.selector_type == "forest" and not is_regression:

            rf_params = self.model_params
            if rf_params is None:
                rf_params = {"n_estimators": 100}

            if self.random_state is not None:
                rf_params["random_state"] = self.random_state

            features_importances = f_forest_classification(X, y, rf_params=self.model_params)

        elif self.selector_type == "linear" and is_regression:
            features_importances = f_linear_regression(X, y, ridge_params=self.model_params)

        elif self.selector_type == "linear" and not is_regression:
            features_importances = f_linear_classification(X, y, logit_params=self.model_params)

        elif self.selector_type == "default" and is_regression:
            features_importances = sklearn.feature_selection.f_regression(X, y)

        elif self.selector_type == "default" and not is_regression:
            features_importances = sklearn.feature_selection.f_classif(X, y)

        else:
            raise ValueError("Unknown selector_type %s" % self.selector_type)  # we should never go there

        if isinstance(features_importances, (list, tuple)):
            features_importances = features_importances[
                0
            ]  # f_regression and f_classification returnes 2uple with value AND pvalue

        columns_index = np.argsort(-features_importances)
        if self.component_selection == "number":
            self.columns_index_to_keep = columns_index[0:n_components]

        elif self.component_selection == "elbow":
            nn = len(features_importances)
            if n_components < nn:
                to_keep = features_importances[columns_index] >= np.max(features_importances) * np.arange(
                    nn
                ) / nn * n_components * 1 / (nn - n_components)
                self.columns_index_to_keep = columns_index[to_keep]
            else:
                self.columns_index_to_keep = columns_index

        else:
            raise ValueError("I don't know that type of 'component_selection' : %s" % self.component_selection)

        if get_type(X) in (DataTypes.DataFrame, DataTypes.SparseDataFrame):
            self._Xcolumns = list(X.columns)
        else:
            self._Xcolumns = list(range(self._Xnbcolumns))

        self._already_fitted = True
        return self

    def get_feature_names(self, input_features=None):
        self._check_is_fitted()

        if input_features is None:
            input_features = self._Xcolumns

        return [input_features[c] for c in self.columns_index_to_keep]

    def transform(self, X):
        self._check_is_fitted()

        if X.shape[1] != self._Xnbcolumns:
            raise ValueError("X doest have the correct size :%d, expected :%d" % (X.shape[1], self._Xnbcolumns))

        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.columns_index_to_keep]
        else:
            if get_type(X) == DataTypes.SparseArray:
                if isinstance(X, sps.coo_matrix):
                    return sps.csc_matrix(X)[:, self.columns_index_to_keep]  # coo_matrix are not subsetable
                else:
                    return X[:, self.columns_index_to_keep]
            else:
                return X[:, self.columns_index_to_keep]


class _FeaturesSelectorClassifier(_BaseFeaturesSelector):
    __doc__ = _BaseFeaturesSelector.__doc__
    is_regression = False


class _FeaturesSelectorRegressor(_BaseFeaturesSelector):
    __doc__ = _BaseFeaturesSelector.__doc__
    is_regression = True


class FeaturesSelectorClassifier(ModelWrapper):
    __doc__ = _BaseFeaturesSelector.__doc__

    def __init__(
        self,
        n_components=0.5,
        selector_type="forest",
        component_selection="number",
        random_state=None,
        model_params=None,
        columns_to_use="all",
        regex_match=False,
        drop_used_columns=True,
        drop_unused_columns=True,
    ):
        self.n_components = n_components
        self.selector_type = selector_type
        self.component_selection = component_selection
        self.model_params = model_params
        self.columns_to_use = columns_to_use
        self.regex_match = regex_match

        super(FeaturesSelectorClassifier, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=regex_match,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=None,
            column_prefix=None,
            desired_output_type=None,
            must_transform_to_get_features_name=False,
            dont_change_columns=False,
            drop_used_columns=drop_used_columns,
            drop_unused_columns=drop_unused_columns,
        )

    def _get_model(self, X, y=None):
        return _FeaturesSelectorClassifier(
            n_components=self.n_components,
            component_selection=self.component_selection,
            selector_type=self.selector_type,
            model_params=self.model_params,
        )


class FeaturesSelectorRegressor(ModelWrapper):
    __doc__ = _BaseFeaturesSelector.__doc__

    def __init__(
        self,
        n_components=0.5,
        selector_type="forest",
        component_selection="number",
        model_params=None,
        columns_to_use="all",
        regex_match=False,
        drop_used_columns=True,
        drop_unused_columns=True,
    ):
        self.n_components = n_components
        self.selector_type = selector_type
        self.component_selection = component_selection
        self.model_params = model_params
        self.columns_to_use = columns_to_use
        self.regex_match = regex_match

        super(FeaturesSelectorRegressor, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=regex_match,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=None,
            column_prefix=None,
            desired_output_type=None,
            must_transform_to_get_features_name=False,
            dont_change_columns=False,
            drop_used_columns=drop_used_columns,
            drop_unused_columns=drop_unused_columns,
        )

    def _get_model(self, X, y=None):
        return _FeaturesSelectorRegressor(
            n_components=self.n_components,
            component_selection=self.component_selection,
            selector_type=self.selector_type,
            model_params=self.model_params,
        )


class _PassThrough(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X


class PassThrough(ModelWrapper):
    """ Dummy transformer that does nothing, used to debug, test or if a step in a pipeline is needed """

    def __init__(self):

        super(PassThrough, self).__init__(
            columns_to_use="all",
            regex_match=False,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=None,
            column_prefix=None,
            desired_output_type=None,
            must_transform_to_get_features_name=False,
            dont_change_columns=True,
            drop_used_columns=True,
            drop_unused_columns=True,
        )

    def _get_model(self, X, y=None):
        return _PassThrough()

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


class _LambdaTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, fun):
        self.fun = fun

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.fun(X)


class LambdaTransformer(ModelWrapper):
    def __init__(
        self,
        fun,
        columns_to_use="all",
        regex_match=False,
        desired_output_type=None,
        drop_used_columns=True,
        drop_unused_columns=True,
    ):

        self.fun = fun
        self.columns_to_use = columns_to_use
        self.regex_match = regex_match
        self.desired_output_type = desired_output_type

        super(LambdaTransformer, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=regex_match,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=None,
            column_prefix=None,
            desired_output_type=desired_output_type,
            must_transform_to_get_features_name=False,
            dont_change_columns=False,
            drop_used_columns=drop_used_columns,
            drop_unused_columns=drop_unused_columns,
        )

    def _get_model(self, X, y=None):
        return _LambdaTransformer(self.fun)


class TruncatedSVDWrapper(ModelWrapper):
    """Wrapper around sklearn :class:`TruncatedSVD` with additional capabilities:

    * can select its columns to keep/drop
    * work on more than one columns
    * can return a DataFrame
    * can add a prefix to the name of columns
    ``n_components`` can be a float, if that is the case it is considered to be a percentage of the total number of columns.
    """

    # TODO : add percentage of explained variance ?
    def __init__(
        self,
        n_components=2,
        columns_to_use="all",
        regex_match=False,
        random_state=None,
        drop_used_columns=True,
        drop_unused_columns=True,
    ):
        self.n_components = n_components
        self.columns_to_use = columns_to_use
        self.regex_match = regex_match
        self.random_state = random_state

        super(TruncatedSVDWrapper, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=regex_match,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=None,
            column_prefix="SVD",
            desired_output_type=DataTypes.DataFrame,
            must_transform_to_get_features_name=True,
            dont_change_columns=False,
            drop_used_columns=drop_used_columns,
            drop_unused_columns=drop_unused_columns,
        )

    def _get_model(self, X, y=None):

        nbcolumns = _nbcols(X)
        n_components = int_n_components(nbcolumns, self.n_components)

        return TruncatedSVD(n_components=n_components, random_state=self.random_state)


class PCAWrapper(ModelWrapper):
    """
        Wrapper around sklearn :class:`PCA` with additional capability:
        n_components``can not be greater than the total number of columns.
    """

    def __init__(
        self, n_components=2, columns_to_use="all", regex_match=False, drop_used_columns=True, drop_unused_columns=True
    ):
        self.n_components = n_components
        self.columns_to_use = columns_to_use
        self.regex_match = regex_match

        super(PCAWrapper, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=regex_match,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=(DataTypes.NumpyArray, DataTypes.DataFrame),
            column_prefix="PCA",
            desired_output_type=DataTypes.DataFrame,
            must_transform_to_get_features_name=True,
            dont_change_columns=False,
            drop_used_columns=drop_used_columns,
            drop_unused_columns=drop_unused_columns,
        )

    def _get_model(self, X, y=None):
        nbcolumns = _nbcols(X)
        nbrows    = _nbrows(X)
        
        n_components = min(int_n_components(nbcolumns, self.n_components),nbrows)

        return PCA(n_components=n_components)


class _KMeansTransformer(BaseEstimator, TransformerMixin):

    _allowed_result_type = ("distance", "inv_distance", "log_distance", "probability", "cluster")

    def __init__(self, n_clusters=8, result_type="probability", temperature=1, scale_input=True, random_state=None):
        self.n_clusters = n_clusters

        self.result_type = result_type
        self.temperature = temperature
        self.scale_input = scale_input

        self.random_state = random_state

        if result_type not in self._allowed_result_type:
            raise ValueError(
                "I don't know that result_type '%s', please choose among '%s"
                % (result_type, ",".join(self._allowed_result_type))
            )

        self.model = None
        self._already_fitted = False

    def fit(self, X, y=None):

        self._fit_transform(X=X, y=y, is_fit=True, is_transform=False)
        self._already_fitted = True

        return self

    def transform(self, X):
        self._check_is_fitted()

        Xres = self._fit_transform(X=X, y=None, is_fit=False, is_transform=True)
        return Xres

    def fit_transform(self, X, y=None):
        Xres = self._fit_transform(X=X, y=y, is_fit=True, is_transform=True)
        self._already_fitted = True

        return Xres

    def _fit_transform(self, X, y, is_fit, is_transform):

        if self.scale_input:
            if is_fit:
                self._scaler = StandardScaler(with_mean=False)
                Xscaled = self._scaler.fit_transform(X)
            else:
                Xscaled = self._scaler.transform(X)
        else:
            Xscaled = X

        if is_fit:
            self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

        if is_fit:
            cluster_distance = self.model.fit_transform(Xscaled)
        else:
            cluster_distance = self.model.transform(Xscaled)

        N, P = X.shape
        K = self.n_clusters

        if self.result_type == "distance":
            if not is_transform:
                return self

            return cluster_distance

        elif self.result_type == "inv_distance":
            if not is_transform:
                return self

            return 1 / (1 + cluster_distance)

        elif self.result_type == "log_distance":
            if not is_transform:
                return self

            return np.log1p(cluster_distance)

        elif self.result_type == "cluster":
            if not is_transform:
                return self

            result = np.zeros((N, K), dtype=np.int32)
            result[np.arange(N), cluster_distance.argmin(axis=1)] = 1

            return result

        elif self.result_type == "probability":

            cluster_one_hot = np.zeros((N, K), dtype=np.bool)
            cluster_one_hot[np.arange(N), cluster_distance.argmin(axis=1)] = 1
            # One-Hot with cluster number

            if is_fit:
                nb_by_cluster = cluster_one_hot.sum(axis=0)
                # Nb of observation by cluster
                self.mean_squared_distance = (
                    (cluster_distance ** 2 * cluster_one_hot).sum(axis=0) / nb_by_cluster
                ).mean()
                self._fitted_N = N

            # Mean squared distance

            if not is_transform:
                return self

            # Un-normalized probability (real proba if temperature = 1)
            exp_median = np.median(P * cluster_distance ** 2 / self.mean_squared_distance * self.temperature)
            unormalized_proba = np.exp(
                -0.5 * (P * cluster_distance ** 2 / self.mean_squared_distance * self.temperature) + 0.5 * exp_median
            )
            # Rmk : exp_median disparear but help make the exp not 0

            row_sum = unormalized_proba.sum(axis=1).reshape((N, 1))

            # Normalized proba
            result = unormalized_proba / row_sum

            # make it more robust:
            result[row_sum[:, 0] == 0, :] = 0.0
            result[result < 10 ** (-10)] = 0.0
            result[result > 1 - 10 ** (-10)] = 1.0
            
            row_sum_is_inf = np.isinf(row_sum)[:,0]
            result[row_sum_is_inf,:] = 1*(cluster_one_hot[row_sum_is_inf,:]) # go back to one-hot
            
            assert not np.isinf(result).any()
            assert not pd.isnull(result).any()

            return result

            ################################################
            ##### Remark : Explanation of calculation ######
            ################################################

            # Notation :
            # * N  : nb of observations
            # * P  : dimension of observation
            # * K  : nb of cluster

            # mu(j) : center of cluster j

            # cluster_distance : (N,K) matrix, with euclidian distance between each observation and each cluster
            # cluster_disance[i,j] = || Xi - mu(j) || = np.sqrt(  np.sum(   (X[i,p] - mu[j,p]) ** 2, p = 0 .. P-1)

            # We assume that when X is in cluster j, X follow a normal law, centered around the cluster center and with diagonal variance
            # with mean       :  mu(j)
            # variance matrix :  sigma**2 * Identity(P) # We assume constante variance across Cluster as KMeans doesn't make that hypothesis

            # sigma**2 = Var( X[,p] | X in cluster j) for all p
            # sigma**2 = 1/P * sum(p = 0..P-1 of E[ (X[:,p] - mu[j,p])**2)
            # sigma**2 = 1/P * E[ distance**2 between X and mu(j)]
            # sigma**2 = 1/P * E[ cluster_distance ** 2 | X in cl], for all j

            # Now we can compute P(cl | X ) :
            # P(cl | X ) = P(X | cl) * P(cl) / sum of (P(X | cl) * P(cl))
            # We assume a priori of 1/K over each cluster

            # Which gives the following proba :
            # exp-1/2 * [  distance**2 / sigma(j)**2      ] / sum of j

            # Remark : the temperature make proba more or less close to 0,1 ...
            # => High temperature => almost like-one hot
            # => Law  temperature => uniform

        else:
            raise ValueError("I don't know that 'result_type' %s" % self.result_type)

    def _check_is_fitted(self):
        """ raise an error if model isn't fitted yet """
        if not self._already_fitted:
            raise NotFittedError(
                "This %s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method." % type(self).__name__
            )

    def get_feature_names(self):
        return ["CL%d" % d for d in range(self.n_clusters)]


class KMeansTransformer(ModelWrapper):
    """ Transformer that apply a KMeans and output distance from cluster center

    Parameters
    ----------

    n_clusters : int, default = 10
        the number of clusters

    result_type : str, default = 'probability'
        determines what to output. Possible choices are

        * 'probability'  : number between 0 and 1 with 'probability' to be in a given cluster
        * 'distance'     : distance to each center
        * 'inv_distance' : inverse of the distance to each cluster
        * 'log_disantce' : logarithm of distance to each cluster
        * 'cluster'      : 0 if in cluster, 1 otherwise

    temperature : float, default = 1
        used to shift probability :unormalized proba = proba ^ temperature

    scale_input : boolean, default = True
        if True the input will be scaled using StandardScaler before applying KMeans

    random_state : int or None, default = None
        the initial random_state of KMeans

    columns_to_use : list of str
        the columns to use

    regex_match : boolean, default = False
        if True use regex to match columns

    drop_used_columns : boolean, default=True
        what to do with the ORIGINAL columns that were transformed.
        If False, will keep them in the result (un-transformed)
        If True, only the transformed columns are in the result

    drop_unused_columns: boolean, default=True
        what to do with the column that were not used.
        if False, will drop them
        if True, will keep them in the result

    desired_output_type : DataType
        the type of result


    """

    def __init__(
        self,
        n_clusters=10,
        result_type="probability",
        temperature=1,
        scale_input=True,
        random_state=None,
        columns_to_use="all",
        regex_match=False,
        desired_output_type=DataTypes.DataFrame,
        drop_used_columns=True,
        drop_unused_columns=True,
    ):

        self.n_clusters = n_clusters
        self.result_type = result_type
        self.random_state = random_state
        self.temperature = temperature
        self.scale_input = scale_input

        self.columns_to_use = columns_to_use
        self.regex_match = regex_match
        self.desired_output_type = desired_output_type

        super(KMeansTransformer, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=regex_match,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=None,
            column_prefix="KM_",
            desired_output_type=desired_output_type,
            must_transform_to_get_features_name=False,
            dont_change_columns=False,
            drop_used_columns=drop_used_columns,
            drop_unused_columns=drop_unused_columns,
        )

    def _get_model(self, X, y=None):
        return _KMeansTransformer(
            n_clusters=self.n_clusters,
            result_type=self.result_type,
            temperature=self.temperature,
            scale_input=self.scale_input,
            random_state=self.random_state,
        )


# In[]


class _PassThroughModel(BaseEstimator, RegressorMixin):
    """ model that predict what is given to it """

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        if X.ndim > 1 and X.shape[1] > 1:
            raise ValueError("only work with 1 dimensional shape X")

        Xa = convert_to_array(X)

        return Xa[:, 0]


# In[]


class _PredictScorerWithTargetModif(_BaseScorer):
    def __init__(self, score_func, sign, inv_function, kwargs):
        self.inv_function = inv_function

        super(_PredictScorerWithTargetModif, self).__init__(score_func, sign, kwargs)

    def __call__(self, estimator, X, y_true, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        y_pred = estimator.predict(X)  # Call predictions

        y_pred_inv = self.inv_function(y_pred)  # Inverse transformation on predictions
        y_true_inv = self.inv_function(y_true)  # Inverse transformation on target

        # So : y_pred_inv, True predictions and y_true_inv : True target

        if sample_weight is not None:
            return self._sign * self._score_func(y_true_inv, y_pred_inv, sample_weight=sample_weight, **self._kwargs)
        else:
            return self._sign * self._score_func(y_true_inv, y_pred_inv, **self._kwargs)


def make_predict_target_modif_scorer(score_func, inv_function, greater_is_better=True, **kwargs):
    sign = 1 if greater_is_better else -1
    return _PredictScorerWithTargetModif(score_func, sign, inv_function, kwargs)


# In[]


class _TargetTransformer(BaseEstimator, RegressorMixin):
    """ TargetTransformer, it is used to fit the underlying model on a transformation of the target

    the model does the following :
        1. transform target using 'target_transform'
        2. fit the underlying model on transformation
        3. when prediction, apply 'inverse_transformation' to result

    Parameter
    ---------
    model : sklearn like model
        the model to use

    """

    def __init__(self, model):
        self.model = model

    def verif_target(self, y):
        pass  # no error by default

    def fit(self, X, y, **fit_transform):
        self.verif_target(y)

        my = self.target_transform(y)

        self.model.fit(X, my, **fit_transform)
        return self

    @if_delegate_has_method("model")
    def predict(self, X):
        my = self.model.predict(X)
        return self.target_inverse_transform(my)

    @if_delegate_has_method("model")
    def fit_transform(self, X, y=None, **fit_params):
        self.verif_target(y)
        my = self.target_transform(y)
        return self.model.fit_transform(X, my, **fit_params)

    @if_delegate_has_method("model")
    def fit_predict(self, X, y=None, **fit_params):
        self.verif_target(y)

        my = self.target_transform(y)
        my_pred = self.model.fit_predict(X, my, **fit_params)
        return self.target_inverse_transform(my_pred)

    @if_delegate_has_method("model")
    def transform(self, X):
        return self.model.transform(X)

    @if_delegate_has_method("model")
    def decision_function(self, X):
        return self.model.decision_function(X)

    def _make_scorer(self, score_name):

        if isinstance(score_name, str):

            score_fun_dico = {
                "explained_variance": sklearn.metrics.scorer.explained_variance_score,
                "r2": sklearn.metrics.scorer.r2_score,
                "neg_median_absolute_error": sklearn.metrics.scorer.median_absolute_error,
                "neg_mean_absolute_error": sklearn.metrics.scorer.mean_absolute_error,
                "neg_mean_squared_error": sklearn.metrics.scorer.mean_squared_error,
                "neg_mean_squared_log_error": sklearn.metrics.scorer.mean_squared_log_error,
                "median_absolute_error": sklearn.metrics.scorer.median_absolute_error,
                "mean_absolute_error": sklearn.metrics.scorer.mean_absolute_error,
                "mean_squared_error": sklearn.metrics.scorer.mean_squared_error,
            }

            greater_is_better = {
                "explained_variance": True,
                "r2": True,
                "neg_median_absolute_error": False,
                "neg_mean_absolute_error": False,
                "neg_mean_squared_error": False,
                "neg_mean_squared_log_error": False,
                "median_absolute_error": False,
                "mean_absolute_error": False,
                "mean_squared_error": False,
            }

            fun = score_fun_dico.get(score_name, None)
            if fun is None:
                return None

            return make_predict_target_modif_scorer(
                fun, inv_function=self.target_inverse_transform, greater_is_better=greater_is_better[score_name]
            )

        elif isinstance(score_name, _PredictScorer):

            scorer = _PredictScorerWithTargetModif(
                score_name._score_func, score_name._sign, self.target_inverse_transform, score_name._kwargs
            )

            return scorer

        else:
            return None

    def approx_cross_validation(
        self,
        X,
        y,
        groups=None,
        scoring=None,
        cv=10,
        verbose=0,
        fit_params=None,
        return_predict=False,
        method="predict",
        no_scoring=False,
        stopping_round=None,
        stopping_threshold=None,
        **kwargs
    ):
        """ Does a cross-validation on a model,

        Read more in the :ref:`User Guide <cross_validation>`. Differences from sklearn function
        * remove paralelle capabilities
        * allow more than one scoring
        * allow return scores and probas or predictions
        * return score on test and train set for each fold
        * bypass complete cv if score are too low
        * call the 'approx_cross_validation' method in the estimator if it exists (allow specific approximate cv for each estimator)

        Parameters
        ----------

        X : array-like
            The data to fit. Can be, for example a list, or an array at least 2d.

        y : array-like, optional, default: None
            The target variable to try to predict in the case of
            supervised learning.

        groups : array-like, optional, default: None
            The groups to use for the CVs

        scoring : string or list of string for each scores
            Can also be a dictionnary of scorers
            A string (see model evaluation documentation) or
            a scorer callable object / function with signature
            ``scorer(estimator, X, y)``.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

            - None, to use the default 3-fold cross-validation,
            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validation strategies that can be used here.

        fit_parameters : dict or None
            Parameters to pass to the fit method of the estimator.

        verbose : integer, optional
            The verbosity level.

        return_predict: boolean, default:False
            if True will also return the out-of-sample predictions

        method : None or string
            the name of the method to use to return predict ('transform','predict','predict_proba',...). if None will guess based on type of estimator

        no_scoring : boolean, default: False
            if True won't score predictions, cv_result will None in that case

        stopping_round : int or None
            if not None the number of the round on which to start looking if the cv must be stopped (ex: stopping_round = 0, stops after first round)

        stopping_threshold : number of None
            if not None the value bellow which we'll stop the CV

        approximate_cv : boolean, default:False
            if False won't call method on estimator and thus won't do any approximation

        **kwargs : keywords arguments to be passed to method call

        Returns
        -------
        cv_res : pd.DataFrame (None if 'no_scoring = True')

        outsample prediction (only if return_predict is True)

        """
        self.verif_target(y)
        my = self.target_transform(y)

        if isinstance(scoring, str):
            scoring = [scoring]

        if method != "predict":
            raise NotImplementedError("please check the code and make adjustement")

        ### Create a list of custom scorers ###
        # Those scorers will modify their target
        modified_scorers = OrderedDict()
        if isinstance(scoring, list):
            for s in scoring:
                scorer = self._make_scorer(s)
                modified_scorers[s] = scorer

        elif isinstance(scoring, str):
            modified_scorers[scoring] = self._make_scorer(s)

        elif isinstance(scoring, dict):
            for k, s in scoring.items():
                scorer = self._make_scorer(s)
                modified_scorers[k] = scorer

        else:
            modified_scorers = None

        use_default_cv = False
        if modified_scorers is None:
            for k, v in modified_scorers.items():
                if v is None:
                    use_default_cv = True
                    break

        if use_default_cv:
            ###############################################
            ### Default mode : I wont modify the scorer ###
            ###############################################

            _, yhat_m = cross_validation(
                self.model,
                X,
                my,
                groups=groups,
                cv=cv,
                verbose=verbose,
                fit_params=fit_params,
                return_predict=True,
                method="predict",
                no_scoring=True,
                stopping_round=None,
                stopping_threshold=None,
                **kwargs
            )

            yhat = self.target_inverse_transform(yhat_m)

            pt_model = _PassThroughModel()
            # .. Rmk : the stopping doesnt do anything here...
            # TODO : il faudrait plutot changer les scorers pour qu'il fasse la transformation inverse ...

            result = cross_validation(
                pt_model,
                yhat,
                yhat,
                scoring=scoring,
                cv=cv,
                verbose=False,
                fit_params=None,
                return_predict=return_predict,
                method=method,
                no_scoring=no_scoring,
                stopping_round=stopping_round,
                stopping_threshold=stopping_threshold,
            )

            return result

        else:

            ################################################
            ### Regular mode : with scoring modification ###
            ################################################

            res = cross_validation(
                self.model,
                X,
                my,
                groups=groups,
                scoring=modified_scorers,
                cv=cv,
                verbose=verbose,
                fit_params=fit_params,
                return_predict=return_predict,
                method=method,
                no_scoring=no_scoring,
                stopping_round=stopping_round,
                stopping_threshold=stopping_threshold,
                **kwargs
            )
            if return_predict:
                cv_res, yhat_m = res

                yhat = self.target_inverse_transform(yhat_m)

                return cv_res, yhat

            else:
                return res


class BoxCoxTargetTransformer(_TargetTransformer):
    """ BoxCoxTargetTransformer, it is used to fit the underlying model on a transformation of the target

    the model does the following :
        1. transform target using 'target_transform'
        2. fit the underlying model on transformation
        3. when prediction, apply 'inverse_transformation' to result

    Here the transformation is in the 'box-cox' family.

    * ll = 0 means this transformation : sign(x) * log(1 + abs(x))
    * ll > 0 sign(x) * exp( log( 1 + ll * abs(xx) ) / ll - 1 )

    Parameters
    ----------
    model : sklearn like model
        the model to use

    ll : float, default = 0
        the boxcox parameter
    """

    def __init__(self, model, ll=0):
        self.model = model
        self.ll = ll

        if self.ll < 0:
            raise ValueError("ll should be positive or null")

    def target_transform(self, y):
        if self.ll == 0:
            return np.sign(y) * np.log1p(np.abs(y))
        else:
            return np.sign(y) * (np.exp(self.ll * np.log1p(np.abs(y))) - 1) / self.ll

    def target_inverse_transform(self, my):
        if self.ll == 0:
            return np.sign(my) * (np.exp(np.abs(my)) - 1)
        else:
            return np.sign(my) * (np.exp(np.log1p(self.ll * np.abs(my)) / self.ll) - 1)


# In[]
def _gen_column_iterator(X, type_of_data=None):
    """ generic column interator, helper to iterator if the column of a data object """
    if type_of_data is None:
        type_of_data = get_type(X)

    if type_of_data in (DataTypes.DataFrame, DataTypes.SparseDataFrame):
        for col in X.columns:
            yield col, X[col]

    elif type_of_data in (DataTypes.NumpyArray, DataTypes.SparseArray):
        ndim = getattr(X, "ndim", None)
        if ndim is None or ndim != 2:
            raise ValueError("This function is used for 2 dimensional data")

            # Sparse Matrix COO are not subscriptable

        for j in range(X.shape[1]):
            yield j, X[:, j]

        ### Attention ca marche pas avec un type COO


def _get_columns(X, type_of_data=None):
    if type_of_data is None:
        type_of_data = get_type(X)

    if type_of_data in (DataTypes.DataFrame, DataTypes.SparseDataFrame):
        return list(X.columns)
    else:
        return list(range(X.shape[1]))


def _index_with_number(X):
    if X.dtype.kind in ("f", "i"):
        return np.array([True] * X.shape[0])

    index_res = np.zeros(X.shape[0], dtype=np.bool)
    for i, x in enumerate(X):
        if isinstance(x, str):
            is_number = False
        else:
            is_number = True
            try:
                np.float32(x)
            except ValueError:
                is_number = False
        index_res[i] = is_number

    return index_res


class _NumImputer(BaseEstimator, TransformerMixin):
    """Numerical Imputer base transformer."""

    def __init__(self, strategy="mean", add_is_null=True, fix_value=0, allow_unseen_null=True, copy_df=True):

        self.strategy = strategy
        self.add_is_null = add_is_null
        self.fix_value = fix_value

        self.copy_df = copy_df
        self.allow_unseen_null = allow_unseen_null
        # If True,  I'll allow the case where a column is never Null in training set, but can be null in testing set
        # If False, I'll generate an error in that case

        self.filling_values = None
        self.columns_with_null = None

        if strategy not in ("mean", "median", "fix"):
            raise ValueError("I don't know that type of strategy '%s' " % self.strategy)

    def fit(self, X, y=None):

        type_of_data = get_type(X)
        self._expected_type = type_of_data

        self.filling_values = {}
        self.columns_with_null = []

        self.columns_mapping = {}

        if type_of_data == DataTypes.SparseArray and not isinstance(X, sps.csc_matrix):
            X = sps.csc_matrix(X)

        for col, Xc in _gen_column_iterator(X, type_of_data=type_of_data):

            if type_of_data == DataTypes.SparseArray:
                Xca = Xc.todense().view(np.ndarray)[:, 0]

            elif type_of_data in (DataTypes.DataFrame, DataTypes.SparseDataFrame):
                Xca = Xc.values

            else:
                Xca = Xc

            # Here Xca is an array
            ii_not_null = ~pd.isnull(Xca)

            # ii_not_inf  = np.abs(Xca) != np.inf
            # ii_not_null = np.logical_and(ii_not_null, ii_not_inf)

            if Xca.dtype.kind not in ("f", "i"):
                ii_contain_number = _index_with_number(Xca)
                ii_not_null = np.logical_and(ii_not_null, ii_contain_number)

                ii_not_inf = np.array([True] * ii_not_null.shape[0])  # assume not 'inf'
                ii_not_inf[ii_contain_number] = np.logical_not(
                    np.isinf(Xca[ii_contain_number].astype(np.float32))
                )  # only compute 'isinf' where it is a number

                ii_not_null = np.logical_and(ii_not_null, ii_not_inf)
            else:
                ii_not_inf = np.logical_not(np.isinf(Xca))
                ii_not_null = np.logical_and(ii_not_null, ii_not_inf)

            any_not_null = ii_not_null.any()
            all_not_null = ii_not_null.all()

            if self.strategy == "fix":
                m = self.fix_value

            elif any_not_null:
                ### There are things that are NOT null

                if not self.allow_unseen_null and all_not_null:
                    m = None
                    # No need to compute mean/median because
                    # 1) I dont have any missing value in that column
                    # 2) I wont allow missing value in testing, if there weren't any missing value in train

                elif self.strategy == "mean":
                    m = Xca[ii_not_null].mean()

                elif self.strategy == "median":
                    m = np.median(Xca[ii_not_null])

                else:
                    raise ValueError("unknown strategy %s" % self.strategy)

            else:
                ### Column is null everywhere...
                m = self.fix_value

            if not all_not_null:
                self.columns_with_null.append(col)

            if m is not None:
                self.filling_values[col] = m

        # cols = _get_columns(X)
        self._Xcolumns = _get_columns(X)

        return self

    def transform(self, X):

        type_of_data = get_type(X)
        if type_of_data != self._expected_type:
            raise ValueError("I'm expecting a type %s" % self._expected_type)

        if self.filling_values is None:
            raise ValueError("model isn't fitted yet")

        if self.copy_df:
            Xcopy = None
            # I'll delayed the copy until I need... that way if no missing value and I don't
        else:
            Xcopy = X

        if self.add_is_null:
            new_columns = []

        if type_of_data == DataTypes.SparseArray and not isinstance(X, sps.csc_matrix):
            X = sps.csc_matrix(X)  #

        for col, Xc in _gen_column_iterator(X, type_of_data=type_of_data):

            # if type_of_data == DataTypes.SparseArray:
            #    Xca = Xc.todense().view(np.ndarray)[:,0]
            # TODO : directly optimized way to get NaN index without converting to sparse

            if type_of_data == DataTypes.DataFrame:
                Xca = Xc.values

            elif type_of_data == DataTypes.SparseDataFrame:
                raise NotImplementedError("I didn't code it yet")

            else:
                Xca = Xc

            if type_of_data == DataTypes.SparseArray:

                ### ATTENTION : ce ne marche que pour CSC matrix !!!
                ii_null = Xca.indices[pd.isnull(Xca.data)]
                has_null = ii_null.shape[0] > 0

                # elif isinstance(Xca,sps.csr_matrix):

                # INDEXES of NON EMPTY things

                # Directly look within non empty things
            else:
                ii_null = pd.isnull(Xca)

                if Xca.dtype.kind not in ("f", "i"):
                    ii_contain_number = _index_with_number(Xca)

                    ii_null = np.logical_or(ii_null, np.logical_not(ii_contain_number))

                    ii_inf = np.array([False] * ii_null.shape[0])  # assume not 'inf'
                    ii_inf[ii_contain_number] = np.isinf(
                        Xca[ii_contain_number].astype(np.float32)
                    )  # only compute 'isinf' where it is a number

                    ii_null = np.logical_or(ii_null, ii_inf)
                else:
                    ii_inf = np.isinf(Xca)
                    ii_null = np.logical_or(ii_null, ii_inf)

                has_null = ii_null.any()

            if has_null:

                if not self.allow_unseen_null and col not in self.columns_with_null:
                    raise ValueError(
                        "This column %s add a null value but it wasn't null anywhere in training set" % str(col)
                    )

                if Xcopy is None:
                    # Now I explicitely need a copy, since I'll modify the DataFrame
                    Xcopy = X.copy()
                    if type_of_data == DataTypes.SparseArray and not isinstance(X, sps.csc_matrix):
                        Xcopy = sps.csc_matrix(Xcopy)  # coo_matrix can't be be subsetted

                if type_of_data in (DataTypes.DataFrame, DataTypes.SparseDataFrame):

                    Xcopy.loc[ii_null, col] = self.filling_values[col]
                    if Xcopy.dtypes[col].kind not in ("f", "i"):
                        Xcopy[col] = Xcopy[col].astype(np.number)

                else:
                    Xcopy[ii_null, col] = self.filling_values[col]

            if self.add_is_null and col in self.columns_with_null:

                if type_of_data in (DataTypes.DataFrame, DataTypes.SparseDataFrame):

                    if col + "_isnull" in list(X.columns):
                        raise ValueError("column %s already exists" % (col + "_isnull"))

                    new_columns.append(pd.DataFrame(1 * ii_null, index=X.index, columns=[col + "_isnull"]))

                elif type_of_data == DataTypes.SparseArray:

                    # Direct creation of a sparse vector of 1
                    _nb = ii_null.shape[0]
                    _data = np.ones(_nb, dtype=np.int32)
                    _col = np.zeros(_nb, dtype=np.int32)
                    _row = ii_null
                    new_columns.append(sps.csc_matrix((_data, (_row, _col)), shape=(X.shape[0], 1)))
                    # TODO  : maybe use 'coo_matrix' ? (more efficient to concatenate after ?)

                    # sps.csr_matrix((np.array([1,1]),(np.array([1,4]),np.array([0,0])))).todense()

                else:
                    new_columns.append(1 * make2dimensions(ii_null))

        if self.add_is_null:

            if Xcopy is None:
                # I dont need a copy... (it will be done copied any way when I stack everything)
                Xcopy = X

            Xcopy = generic_hstack([Xcopy] + new_columns, output_type=type_of_data)
        else:
            if Xcopy is None:
                Xcopy = X  # If I'm here, it means that nothing was actually Null... and so I don't need a copy

        return Xcopy

    def get_feature_names(self, input_features=None):

        if input_features is None:
            input_features = self._Xcolumns

        features_names = [str(c) for c in input_features]
        if self.add_is_null:

            features_names += [
                str(c1) + "_isnull" for c1, c2 in zip(input_features, self._Xcolumns) if c2 in self.columns_with_null
            ]

        return features_names


class NumImputer(ModelWrapper):
    """Missing value imputer for numerical features.

    Parameters
    ----------
    strategy : str, default = 'mean'
        how to fill missing value, possibilities ('mean', 'fix' or 'median')

    add_is_null : boolean, default = True
        if this is True of 'is_null' columns will be added to the result

    fix_value : float, default = 0
        the fix value to use if needed

    allow_unseen_null : boolean, default = True
        if not True an error will be generated on testing data if a column has missing value in test but didn't have one in train

    columns_to_use : list of str or None
        the columns to use

    drop_used_columns : boolean, default=True
        what to do with the ORIGINAL columns that were transformed.
        If False, will keep them in the result (un-transformed)
        If True, only the transformed columns are in the result
        
    drop_unused_columns: boolean, default=True
        what to do with the column that were not used.
        if False, will drop them
        if True, will keep them in the result

    regex_match : boolean, default = False
        if True, use regex to match columns
    """

    def __init__(
        self,
        strategy="mean",
        add_is_null=True,
        fix_value=0,
        allow_unseen_null=True,
        columns_to_use="all",
        regex_match=False,
        drop_used_columns=True,
        drop_unused_columns=True,
    ):
        self.strategy = strategy
        self.add_is_null = add_is_null
        self.fix_value = fix_value
        self.allow_unseen_null = allow_unseen_null
        super(NumImputer, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=regex_match,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=None,
            column_prefix=None,
            desired_output_type=None,
            must_transform_to_get_features_name=False,
            dont_change_columns=False,
            drop_used_columns=drop_used_columns,
            drop_unused_columns=drop_unused_columns,
        )

    def _get_model(self, X, y=None):
        return _NumImputer(
            strategy=self.strategy,
            add_is_null=self.add_is_null,
            fix_value=self.fix_value,
            allow_unseen_null=self.allow_unseen_null,
            copy_df=True,
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
        return not self.add_is_null


# In[] : Scaler


class _CdfScaler(BaseEstimator, TransformerMixin):
    """ Scaler using the CDF of a law """

    def __init__(
        self,
        distribution="auto-kernel",
        output_distribution="uniform",
        copy=True,
        verbose=False,
        sampling_number=1000,
        random_state=None,
    ):

        self.distribution = distribution
        self.output_distribution = output_distribution
        self.copy = copy
        self.verbose = verbose
        self.sampling_number = sampling_number
        self.random_state = random_state

    def _prepare_attributes(self, X):
        """ method to create the distributions attributes """
        nbcols = _nbcols(X)
        if isinstance(self.distribution, str):
            self.distributions = [self.distribution] * nbcols

        elif isinstance(self.distribution, (list, tuple)):

            if len(self.distributions) != nbcols:
                raise ValueError("If distribution is a list it should have the same number of column has X")

            self.distributions = self.distribution

        # TODO : dico of distributions
        else:
            raise TypeError("I don't know how to handle that type of distribution %s" % type(self.distribution))

    def _guess_distribution(self, X, type_of_data):
        """ method to guess which distribution to use in the case of "auto-kernel" or "auto-param"

        The guessing uses the following rules :
            * if less than 5 differents values : use "none" <=> no transformation applied
            * otherwise if "auto-kernel" : uses "kernel" <=> fit a kernel density
            * otherwise if "auto-param" :
                * if negative and positive values : use "normal" <=> fit a normal law
                * if positive values only and values above 1 : use 'gamma' <=> fit a gamma law
                * if values between 0 and 1 : use  'beta' <=> fit a betta law

        """

        if len({"auto-param", "auto-kernel", "auto-rank", "auto-nonparam"}.intersection(self.distributions)) == 0:
            return

        modified_distributions = []
        for dist, (col, Xc) in zip(self.distributions, _gen_column_iterator(X, type_of_data=type_of_data)):

            if dist not in ("auto-param", "auto-kernel", "auto-rank", "auto-nonparam"):
                modified_distributions.append(dist)
                continue

            if type_of_data == DataTypes.SparseArray:
                Xca = Xc.data  # => only non zero elements
                # Xc.todense().view(np.ndarray)[:,0] # everything

            elif type_of_data in (DataTypes.DataFrame, DataTypes.SparseDataFrame):
                Xca = Xc.values

            else:
                Xca = Xc

            ### Less than 5 elements => no scaling ###
            if len(np.unique(Xca)) <= 5:
                modified_distributions.append("none")

            else:
                if dist == "auto-kernel":
                    guess = "kernel"

                elif dist == "auto-rank":
                    guess = "rank"

                elif dist == "auto-nonparam":
                    if len(Xca) <= 1000:
                        guess = "kernel"
                        # When too many observations, kernel estimation takes too much time
                    else:
                        guess = "rank"

                else:  # auto-param : we'll fit a parametric distribution
                    m = Xca.min()
                    M = Xca.max()
                    if m <= 0 and M > 0:
                        guess = "normal"

                    elif m > 0 and M >= 1:
                        guess = "gamma"

                    elif m > 0 and M < 1:
                        guess = "beta"

                    else:
                        guess = "kernel"  # never go there

                modified_distributions.append(guess)

        self.distributions = modified_distributions

    def fit(self, X, y=None):

        type_of_data = get_type(X)

        self._prepare_attributes(X)

        self._guess_distribution(X, type_of_data=type_of_data)

        self._random_gen = check_random_state(self.random_state)

        self._expected_type = type_of_data
        
        s = getattr(X,"shape",None)
        if s is not None:
            self._epsilon = 1/(2*s[0])
        else:
            self._epsilon = 0.0001

        if type_of_data == DataTypes.SparseArray and not isinstance(X, sps.csc_matrix):
            X = sps.csc_matrix(X)

        self.fitted_distributions = {}

        for dist, (col, Xc) in zip(self.distributions, _gen_column_iterator(X, type_of_data=type_of_data)):

            if self.verbose:
                print("start processing %s, using %s" % (str(col), str(dist)))

            if type_of_data == DataTypes.SparseArray:
                # Xca = Xc.todense().view(np.ndarray)[:,0]
                Xca = Xc.data  # only non zero elements

            elif type_of_data in (DataTypes.DataFrame, DataTypes.SparseDataFrame):
                Xca = Xc.values

            else:
                Xca = Xc

            if dist == "kernel" and self.sampling_number is not None and self.sampling_number < len(Xca):
                if self.verbose:
                    print("I'll sample %d values from the data" % self.sampling_number)

                index = self._random_gen.choice(len(Xca), size=self.sampling_number, replace=False)
                Xca_sample = Xca[index]
            else:
                Xca_sample = Xca

            ########################
            ###  Parametric law  ###
            ########################
            if dist == "gamma":
                if Xca_sample.std() == 0:
                    self.fitted_distributions[col] = None
                else:
                    params = scipy.stats.gamma.fit(Xca_sample)
                    self.fitted_distributions[col] = scipy.stats.gamma(*params)

            elif dist == "normal":
                params = scipy.stats.norm.fit(Xca_sample)
                if params[1] == 0.0: # std == 0
                    self.fitted_distributions[col] = None
                else:
                    self.fitted_distributions[col] = scipy.stats.norm(*params)

            elif dist == "beta":
                params = scipy.stats.beta.fit(Xca_sample)
                self.fitted_distributions[col] = scipy.stats.beta(*params)

            ####################
            ###  No Scaling  ###
            ####################
            elif dist == "none":
                self.fitted_distributions[col] = None

            ##########################################
            ###  Non Parametric kernel estimation  ###
            ##########################################
            elif dist == "kernel":
                kde = KDEMultivariate(data=Xca_sample, var_type="c", bw="normal_reference")
                if kde.bw == 0: # means data is almost constant => can't compute a cdf
                    self.fitted_distributions[col] = None
                else:                    
                    self.fitted_distributions[col] = kde

            ###############################################
            ###  Non Parametric simple rank estimation  ###
            ###############################################
            elif dist == "rank":
                Xca_sorted = np.unique(np.sort(Xca_sample))

                n = len(Xca_sorted)
                nn = np.arange(n) / n + 1 / (2 * n)
                self.fitted_distributions[col] = interp1d(
                    Xca_sorted, nn, kind="linear", bounds_error=False, fill_value=(10 ** (-10), 1 - 10 ** (-10))
                )

            else:
                raise ValueError("I don't know this distribution %s" % dist)

        return self

    def transform(self, X):
        type_of_data = get_type(X)

        if type_of_data != self._expected_type:
            raise TypeError("I should have a type %s instead I got %s" % (self._expected_type, type_of_data))

        Xcopy = None
        if type_of_data == DataTypes.SparseArray:
            if not isinstance(X, sps.csc_matrix):
                Xcopy = sps.csc_matrix(X)  # copy to csc matrix

        if Xcopy is None:
            if self.copy:
                Xcopy = X.copy()
            else:
                Xcopy = X

        for dist, (col, Xc) in zip(self.distributions, _gen_column_iterator(Xcopy, type_of_data=type_of_data)):

            if self.verbose:
                print("transforming %s using %s" % (col, dist))

            if type_of_data == DataTypes.SparseArray:
                # Xca = Xc.todense().view(np.ndarray)[:,0]
                Xca = Xc.data  # only non zero data

            elif type_of_data in (DataTypes.DataFrame, DataTypes.SparseDataFrame):
                Xca = Xc.values

            else:
                Xca = Xc

            ### Apply CDF ###
            if dist in {"gamma", "normal", "beta", "kernel"}:
                if self.fitted_distributions[col] is not None:
                    Xca_modified = (1 - 2 * self._epsilon) * self.fitted_distributions[col].cdf(Xca) + self._epsilon
                else:
                    Xca_modified = Xca[:] * 0 + 0.5 # constant 0.5

            ### Don't do anything ###
            elif dist == "none":
                Xca_modified = Xca  # nothing

            ### Apply rank function ###
            elif dist == "rank":

                Xca_modified = self.fitted_distributions[col](Xca)

            else:
                raise ValueError("I don't know that type of distribution %s" % dist)

            ### Modify cdf ###
            if self.output_distribution == "normal" and dist != "none":
                Xca_modified = scipy.stats.norm.ppf(
                    Xca_modified
                )  # inverse normal law to have a guassian distribution at the end

            if type_of_data in (DataTypes.DataFrame, DataTypes.SparseDataFrame):
                Xcopy[col] = Xca_modified

            elif type_of_data in (DataTypes.SparseArray,):
                #                for c in range(Xcopy.shape[1]):
                #                    assert (Xcopy[:,col].data == Xcopy.data[ Xcopy.indptr[col]:Xcopy.indptr[col+1]]).all()

                Xcopy.data[Xcopy.indptr[col] : Xcopy.indptr[col + 1]] = Xca_modified

            else:
                Xcopy[:, col] = Xca_modified

        return Xcopy


# In[]


class CdfScaler(ModelWrapper):
    """ Scaler based on the distribution

    Each variable is scaled according to its law. The law can be approximated using :
        * parametric law : distribution = "normal", "gamma", "beta"
        * kernel approximation : distribution = "kernel"
        * rank approximation   : "rank"
        * if distribution = "none" : no distribution is learned and no transformation is applied (useful to not transform some of the variables)
        * if distribution = "auto-kernel" : automatic guessing on which column to use a kernel (columns whith less than 5 differents values are un-touched)
        * if distribution = "auto-param"  : automatic guessing on which column to use a parametric distribution (columns with less than 5 differents valuee are un-touched)
        for other columns choice among "normal", "gamma" and "beta" law based on values taken

    After the law is learn, the result is transformed into :
        * a uniform distribution (output_distribution = 'uniform')
        * a gaussian distribution (output_distribution = 'normal')

    Parameters
    ----------

    distribution : str or list of str, default = "auto-kernel"
        the distribution to use for each variable, if only one string the same transformation is applied everything where

    output_distribution : str, default = "uniform"
        type of output, either "uniform" or "normal"

    copy : boolean, default = True
        if True wil copy the data then modify it

    verbose : boolean, default = True
        set the verbosity level

    sampling_number : int or None, default = 1000
        if set subsample of size 'sampling_number' will be drawn to estimate kernel densities

    random_state : int or None
        state of the random generator

    columns_to_use : list of str
        the columns to use

    regex_match : boolean, default = False
        if True use regex to match columns

    drop_used_columns : boolean, default=True
        what to do with the ORIGINAL columns that were transformed.
        If False, will keep them in the result (un-transformed)
        If True, only the transformed columns are in the result
        
    drop_unused_columns: boolean, default=True
        what to do with the column that were not used.
        if False, will drop them
        if True, will keep them in the result

    desired_output_type : DataType
        the type of result

    """

    def __init__(
        self,
        distribution="auto-kernel",
        output_distribution="uniform",
        copy=True,
        verbose=False,
        sampling_number=1000,
        random_state=None,
        columns_to_use="all",
        regex_match=False,
        drop_used_columns=True,
        drop_unused_columns=True,
        desired_output_type=None,
    ):

        self.distribution = distribution
        self.output_distribution = output_distribution
        self.copy = copy
        self.verbose = verbose
        self.sampling_number = sampling_number
        self.random_state = random_state

        super(CdfScaler, self).__init__(
            columns_to_use=columns_to_use,
            regex_match=regex_match,
            work_on_one_column_only=False,
            all_columns_at_once=True,
            accepted_input_types=None,
            column_prefix=None,
            desired_output_type=desired_output_type,
            must_transform_to_get_features_name=False,
            dont_change_columns=True,
            drop_used_columns=drop_used_columns,
            drop_unused_columns=drop_unused_columns,
        )

    def _get_model(self, X, y=None):

        return _CdfScaler(
            distribution=self.distribution,
            output_distribution=self.output_distribution,
            copy=self.copy,
            verbose=self.verbose,
            random_state=self.random_state,
            sampling_number=self.sampling_number,
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
