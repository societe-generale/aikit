# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:19:11 2020

@author: LionelMassoulard
"""

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, clone

from aikit.tools.data_structure_helper import get_type, DataTypes

def _gen_sub_index(X, index):
    
    if isinstance(X, list):
        return [_gen_sub_index(xsub, index) for xsub in X]
    
    if get_type(X) == DataTypes.DataFrame:
        return X.loc[index]
    else:
        return X[index]

def _gen_sub_column(X, col):
    
    if isinstance(X, list):
        return [_gen_sub_column(xsub, col) for xsub in X]
        
    if get_type(X) == DataTypes.DataFrame:
        if isinstance(col, int):
            return X.iloc[:, col]
        else:
            return X.loc[:, col]
    else:
        return X[:, col]
    
def _gen_assign(X, index, value):
    if isinstance(X, list):
        if not isinstance(value, list):
            raise TypeError("value should be a list if X is a list")
        if len(X) != len(value):
            raise ValueError("value and X should have the same size, instead I got %d and %d" % (len(X), len(value)))
    
        return [_gen_assign(xs, index, vs) for xs, vs in zip(X, value)]
    
    if get_type(X) == DataTypes.DataFrame:
        X.loc[index, :] = value.values

    else:
        X[index, ...] = value
    
    return X
    

def _align_predict(predictions, predictions_classes, classes):
    """ method to align prediction made by 'predict_proba', 'predict_log_proba', and 'decision_function
    To deal with the fact that predictions might not be aligned on the same classes
    
    Ex : if predictions has 2 columns, predictions_classes = ["a", "c"]
    classes = ["a",b","c"] (so class 'b' wasn't present)
    
    The final result should have 3 classes
    
    """
    
    if isinstance(predictions, list):
        
        if not isinstance(predictions_classes, list):
            raise TypeError("predictions_class should be a list if predictions is a list")
        
        if not isinstance(classes, list):
            raise TypeError("classes should be a list if predictions is a list")
        
        if len(predictions) != len(predictions_classes):
            raise ValueError("predictions (len = %d) and predictions_classes (len = %d) should have the same length" % (len(predictions), len(predictions_classes)))
            
        if len(predictions) != len(classes):
            raise ValueError("predictions (len = %d) and classes (len = %d) should have the same length" % (len(predictions), len(classes)))
    
        return [_align_predict(p, pc, c) for p, pc, c in zip(predictions, predictions_classes, classes ) ]
    
    
    float_min = np.finfo(predictions.dtype).min
    default_values = {"decision_function": float_min, "predict_log_proba": float_min, "predict_proba": 0}

    predictions_for_all_classes = np.zeros( (predictions.shape[0],len(classes)), dtype=predictions.dtype)
    predictions_for_all_classes[:] = default_values[method]

    class_to_col = {c:j for j,c in enumerate(classes)}
    for j, c in enumerate(predictions_classes):
        
        predictions_for_all_classes[:, class_to_col[c]] = predictions[:, j]

    return predictions_for_all_classes


class SplitByVariable(BaseEstimator):
    
    def __init__(self, estimator, variable):
        self.estimator = estimator
        self.variable=variable
        
        
    def fit(self, X, y=None):
        
        # check type of 'X' here
        
        unique_X = np.sort(np.unique(_gen_sub_column(X, self.variable)))
        
        global_model = None
        self._estimators = {}
        for ux in unique_X:
            index = _gen_sub_column(X, self.variable) == ux
            
            Xsub = _gen_sub_index(X, index)
            if y is not None:
                ysub = y[index]
            else:
                ysub = None
            
            # TODO : test that ysub contains more than one value : if not the case, use global model
            use_global = len(np.unique(ysub)) == 1
            
            if use_global:
                
                if global_model is None:
                    global_model = clone(self.estimator)
                    global_model.fit(X, y)

                self._estimators[ux] = global_model                    
            else:
         
                cloned_estimator = clone(self.estimator)
                cloned_estimator.fit(Xsub, ysub)
                self._estimators[ux] = cloned_estimator

        return self
    
    def predict(self, X):
        return self._predict(X, "predict")
    
    def predict_proba(self, X):
        # pour le predict proba il faut merger les classes...
        return self._predict(X, "predict_proba")
    
    def _predict(self, X, method):
        
        unique_X = np.sort(np.unique(X[self.variable])) # if DataFrame
        for i, ux in enumerate(unique_X):
            
            index = _gen_sub_column(X, self.variable) == ux            
            Xsub = _gen_sub_index(X, index)
            
            yhat_sub = getattr(self._estimators[ux], method)(Xsub)
            if method == "predict_proba":
                yhat_sub = _realign_class(yhat_sub, current_classes=self._estimators[ux].classes_, self.classes_)
            
            if i == 0:
                if isinstance(yhat_sub, list):
                    yhat = [np.zeros( (X.shape[0],) + ys.shape[1:], dtype=ys.dtype) for ys in yhat_sub]
                else:
                    yhat = np.zeros( (X.shape[0],) + yhat_sub.shape[1:], dtype=yhat_sub.dtype)
                
            yhat = _gen_assign(yhat, index, yhat_sub)
            
        return yhat
            
                                
# In[]

X = pd.DataFrame({"a":[0,0,0,1,1,1],"b":np.random.randn(6)})
y = 1*(np.random.randn(6) > 0)
y = 1*(np.random.randn(6,2) > 0)




from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

self = SplitByVariable(DecisionTreeClassifier(), "a")
self.fit(X, y)
yhat = self.predict(X)
yhat

yhat = self.predict_proba(X)
yhat

self._predict(X, "predict_proba")

