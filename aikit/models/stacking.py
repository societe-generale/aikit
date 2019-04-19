# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 09:07:47 2018

@author: Lionel Massoulard
"""

import numpy as np

from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, TransformerMixin
from sklearn.base import is_classifier, is_regressor
from sklearn.exceptions import NotFittedError

from sklearn.utils.metaestimators import if_delegate_has_method


from aikit.cross_validation import cross_validation, create_cv
from aikit.tools.data_structure_helper import convert_generic


def maketwodimensions(x):
    """ helper function to make sure a numpy array is at least 2 dimensions """
    if len(x.shape) == 1:
        return x.reshape((x.shape[0], 1))
    else:
        return x


# In[]


class _BaseStacker(BaseEstimator):
    """ generic class to handle stacking 
    
    This class takes a list of models and does the following during its fitting phase
    
    1. does a cross-validation on each model to output out-sample predictions
    2. use those out-sample prediction to fit a blending model
    3. re-fit the models on all the datas
    
    During test:
    1. call each models to retrieve predictions
    2. call the blender to retrieve final aggregated prediction
    
    
    
    Parameters
    ---------
    models : list of model
        the models that we want to stacked
        
    cv : cv object or int
        the cross-validation to use to fit the blender
        
    blender : model
        the blending model
        
    
    """

    _is_classifier = None
    _method = None

    def __init__(self, models, cv, blender, random_state=None):
        self.models = models
        self.cv = cv
        self.blender = blender

        self.random_state = random_state

    def get_outsample(self, X, y, method, groups=None, cv=None):
        """ retrieve 'outsample' prediction using a cross-validation """

        if cv is None:
            cv = self._cv
        else:
            cv = create_cv(cv, y, random_state=self.random_state, classifier=self._is_classifier, shuffle=True)

        ### 1) CV fitting of all models ####
        all_yhat_pred = []
        for model in self.models:

            yhat_pred = maketwodimensions(cross_val_predict(model, X, y, groups=groups, cv=cv, method=method))
            all_yhat_pred.append(yhat_pred)

        ### 2) concatenate ####
        all_yhat_pred = np.concatenate(all_yhat_pred, axis=1)

        return all_yhat_pred

    def get_predict(self, X):
        all_yhat_pred = [maketwodimensions(model.predict(X)) for model in self.models]
        all_yhat_pred = np.concatenate(
            all_yhat_pred, axis=1
        )  # Rmk : probably more memory efficient to pre-allocate the array...
        return all_yhat_pred

    def get_predict_proba(self, X):
        all_yhat_pred = [maketwodimensions(model.predict_proba(X)) for model in self.models]
        all_yhat_pred = np.concatenate(
            all_yhat_pred, axis=1
        )  # Rmk : probably more memory efficient to pre-allocate the array...
        return all_yhat_pred

    def fit(self, X, y, groups=None):

        self._cv = create_cv(self.cv, y, classifier=self._is_classifier, random_state=self.random_state)

        all_yhat_pred = self.get_outsample(X, y, method=self._method, groups=groups)

        N = y.shape[0]
        assert all_yhat_pred.shape[0] == N

        ### 3) fit blender ####
        self.blender.fit(all_yhat_pred, y)

        ### 4) refit model ####
        for model in self.models:
            model.fit(X, y)

        return self

    def approx_cross_validation(
        self,
        X,
        y,
        groups=None,
        scoring=None,
        cv=None,
        verbose=1,
        fit_params=None,
        return_predict=False,
        method=None,
        no_scoring=False,
        stopping_round=None,
        stopping_threshold=None,
        _save_outsample_predict=False,
        _use_saved_outsample_predict=False,
    ):
        """ cross validation of the blender of the stacker
        The fold to use to cross-validate the blender are the SAME as the one used to generate 'outsample prediction'
        """

        cv = create_cv(cv, y, classifier=self._is_classifier, shuffle=True, random_state=self.random_state)

        if _use_saved_outsample_predict:
            all_yhat_pred = self.all_yhat_pred
        else:
            all_yhat_pred = self.get_outsample(X, y, method=self._method, groups=groups, cv=cv)

            if _save_outsample_predict:
                self.all_yhat_pred = all_yhat_pred

        return cross_validation(
            self.blender,
            all_yhat_pred,
            y,
            scoring=scoring,
            cv=cv,
            verbose=verbose,
            fit_params=fit_params,
            return_predict=return_predict,
            method=method,
            no_scoring=no_scoring,
            stopping_round=stopping_round,
            stopping_threshold=stopping_threshold,
        )

    @if_delegate_has_method("blender")
    def predict(self, X):
        ### 1) call models
        if self._method == "predict_proba":
            all_yhat_pred = self.get_predict_proba(X)
        else:
            all_yhat_pred = self.get_predict(X)

        ### 2) call blender
        yhat = self.blender.predict(all_yhat_pred)

        return yhat

    @if_delegate_has_method("blender")
    def predict_proba(self, X):
        ### 1) call models
        if self._method == "predict_proba":
            all_yhat_pred = self.get_predict_proba(X)
        else:
            all_yhat_pred = self.get_predict(X)

        ### 2) call blender
        yhat = self.blender.predict_proba(all_yhat_pred)

        return yhat

    @if_delegate_has_method("blender")
    def predict_log_proba(self, X):
        ### 1) call models
        if self._method == "predict_proba":
            all_yhat_pred = self.get_predict_proba(X)
        else:
            all_yhat_pred = self.get_predict(X)

        ### 2) call blender
        yhat = self.blender.predict_log_proba(all_yhat_pred)

        return yhat

    @if_delegate_has_method("blender")
    def decision_function(self, X):
        ### 1) call models
        if self._method == "predict_proba":
            all_yhat_pred = self.get_predict_proba(X)
        else:
            all_yhat_pred = self.get_predict(X)

        ### 2) call blender
        yhat = self.blender.decision_function(all_yhat_pred)

        return yhat


class StackerClassifier(_BaseStacker, ClassifierMixin):
    __doc__ = _BaseStacker.__doc__

    _method = "predict_proba"
    _is_classifier = True

    @property
    def classes_(self):
        return self.blender.classes_


class StackerRegressor(_BaseStacker, RegressorMixin):
    __doc__ = _BaseStacker.__doc__

    _method = "predict"
    _is_classifier = False


# In[] : model wrapper


class OutSamplerTransformer(BaseEstimator, TransformerMixin):
    """ This class is used to transform a model in a transformers that makes out of sample predictions
    
    This transformation can be used to easily use model in part of a GraphPipeline.
    
    fit method :
    1. simply fit the underlying model
        
    fit_transform method :
    1. do a cross-validation on the underlying model to output out-of-sample prediction
    2. re-fit underlying model on all the data
    
    transform method :
    1. just output prediction of underlying model

    
    Parameters
    ----------
    model : a model
        the model that we want to 'transform'
        
    cv : cv object or int
        which crossvalidation to use
        
    random_state : int or None
        specify the random state (to force the CV the be fixed)
        
    desired_output_type : None or type of output
        the output type of the result of the transformation
        
    columns_prefix : None or str
        each column will be prefixed by it 
    
    """

    def __init__(
        self,
        model,
        cv=10,
        random_state=123,  # I fix the seed by default so that the CVs are identical
        desired_output_type=None,
        columns_prefix=None,
    ):
        self.model = model
        self.cv = cv
        self.random_state = random_state

        self.desired_output_type = desired_output_type
        self.columns_prefix = columns_prefix

        self._already_fitted = False

    def fit(self, X, y):

        self._already_fitted = True

        if is_classifier(self.model):
            self._is_classifier = True

        elif is_regressor(self.model):
            self._is_classifier = False

        else:
            raise ValueError("model should either be a Classifier or a Regressor")

        if self._is_classifier:
            self._nby = len(np.unique(y))

        self.model.fit(X, y)

        if self._is_classifier:
            # Classification model
            if self._nby == 2:
                self._feature_names = ["%s__%s" % (self.model.__class__.__name__, self.model.classes_[1])]
            else:
                self._feature_names = ["%s__%s" % (self.model.__class__.__name__, c) for c in self.model.classes_]

        else:
            # Regression model
            self._feature_names = ["%s__target" % self.model.__class__.__name__]

        if self.columns_prefix is not None:
            self._feature_names = ["%s__%s" % (self.columns_prefix, c) for c in self._feature_names]

        return self

    def transform(self, X):

        if not self._already_fitted:
            raise NotFittedError(
                "This %s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method." % type(self).__name__
            )

        if self._is_classifier:
            if self._nby == 2:
                res = maketwodimensions(self.model.predict_proba(X)[:, 1])
            else:
                res = maketwodimensions(self.model.predict_proba(X))

        else:
            res = maketwodimensions(self.model.predict(X))

        res = convert_generic(res, output_type=self.desired_output_type)

        if hasattr(res, "columns"):
            res.columns = self.get_feature_names()

        return res

    def fit_transform(self, X, y, groups=None):

        self._already_fitted = True

        if is_classifier(self.model):
            self._is_classifier = True

        elif is_regressor(self.model):
            self._is_classifier = False

        else:
            raise ValueError("model should either be a Classifier or a Regressor")

        if self.cv is None:
            self._cv = None

            return self.fit(X, y).transform(X)
            # No CV in that case

        else:
            self._cv = create_cv(
                self.cv, y, random_state=self.random_state, classifier=self._is_classifier, shuffle=True
            )

        if self._is_classifier:
            self._nby = len(np.unique(y))

            if self._nby == 2:
                all_yhat_pred = maketwodimensions(
                    cross_val_predict(self.model, X, y, groups=groups, cv=self._cv, method="predict_proba")[:, 1]
                )
            else:
                all_yhat_pred = maketwodimensions(
                    cross_val_predict(self.model, X, y, groups=groups, cv=self._cv, method="predict_proba")
                )
        else:
            all_yhat_pred = maketwodimensions(
                cross_val_predict(self.model, X, y, groups=groups, cv=self._cv, method="predict")
            )

        self.model.fit(X, y)

        if self._is_classifier:
            # Classification model
            if self._nby == 2:
                self._feature_names = ["%s__%s" % (self.model.__class__.__name__, self.model.classes_[1])]
            else:
                self._feature_names = ["%s__%s" % (self.model.__class__.__name__, c) for c in self.model.classes_]

        else:
            # Regression model
            self._feature_names = ["%s__target" % self.model.__class__.__name__]

        if self.columns_prefix is not None:
            self._feature_names = ["%s__%s" % (self.columns_prefix, c) for c in self._feature_names]

        if hasattr(all_yhat_pred, "columns"):
            all_yhat_pred.columns = self.get_feature_names()

        return all_yhat_pred

    def get_feature_names(self):
        return self._feature_names

    def approx_cross_validation(
        self,
        X,
        y,
        groups=None,
        scoring=None,
        cv=None,
        verbose=1,
        fit_params=None,
        return_predict=False,
        method="transform",
        no_scoring=True,
        stopping_round=None,
        stopping_threshold=None,
    ):

        if is_classifier(self.model):
            _is_classifier = True

        elif is_regressor(self.model):
            _is_classifier = False

        else:
            raise ValueError("model should either be a Classifier or a Regressor")

        if cv is None:
            # I'll use cv of stacker
            raise ValueError("I need a cv do cross-validate")
            # cv = create_cv(self.cv, y, random_state = self.random_state, classifier = self._is_classifier, shuffle = True)

        cv = create_cv(cv, y, random_state=123, classifier=_is_classifier, shuffle=True)

        if not no_scoring:
            raise ValueError("no scoring should be True for a transformer")

        if method != "transform":
            raise ValueError("method should be 'transform' for a transformer")

        if _is_classifier:
            _nby = len(np.unique(y))

            if _nby == 2:
                all_yhat_pred = maketwodimensions(
                    cross_val_predict(self.model, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
                )
            else:
                all_yhat_pred = maketwodimensions(
                    cross_val_predict(self.model, X, y, groups=groups, cv=cv, method="predict_proba")
                )
        else:
            all_yhat_pred = maketwodimensions(
                cross_val_predict(self.model, X, y, groups=groups, cv=cv, method="predict")
            )

        # None : no scoring, this is a transformer
        return None, all_yhat_pred
