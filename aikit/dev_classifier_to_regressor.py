# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:01:03 2020

@author: LionelMassoulard
"""


from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, is_classifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer

from aikit.tools.data_structure_helper import make2dimensions, make1dimension

import numpy as np
import pandas as pd

X = np.random.randn(100,10)
y = np.random.randn(100)
y_ord = np.random.randint(0,3, size=100)


class RegressorFromClassifier(BaseEstimator, RegressorMixin):
    """ Transform a Classifier into a regressor
    
    does it by clustering the target and fit a classification
    """
    
    def __init__(self,
                 classifier_model,
                 strategy="kmeans",
                 n_bins=10,
                 y_clusterer=None
                 ):
        self.classifier_model=classifier_model
        self.strategy=strategy
        self.n_bins=n_bins
        self.y_clusterer=y_clusterer
        
        
    def get_default_y_cluster(self, y=None):
        return KBinsDiscretizer(n_bins=self.n_bins, strategy=self.strategy, encode="ordinal")
  
    
    def fit(self, X, y):
        
        if self.y_clusterer is None:
            y_clusterer = self.get_default_y_cluster(y)
        else:
            y_clusterer = self.y_clusterer
            # TODO : check that it is a cluserer
            
        if not is_classifier(self.classifier_model):
            raise TypeError("classifier_model should be a classifer")
            
        yd2 = make2dimensions(y)
        
        if hasattr(y_clusterer, "fit_predict"):
            y_cl = y_clusterer.fit_predict(yd2)
        else:
            y_cl = y_clusterer.fit_transform(yd2).astype('int32')
        
        if y_cl.ndim > 1 and y_cl.shape[1] > 1:
            raise ValueError("The cluster should return only 1 dimensional clusters")
        elif y_cl.ndim > 1:
            y_cl = y_cl[:,0]
            
        self.classifier_model.fit(X, y_cl) # fit classifier on result of cluster
        
        y_mean_mapping = self.compute_y_mean(yd2, y_cl)
        
        
        self._y_mean_mapping_matrix = np.concatenate([y_mean_mapping[cl] for cl in self.classifier_model.classes_], axis=0)
        
        return self
    
    @staticmethod
    def compute_y_mean(yd2, y_cl):
        index_dico = {cl : g.index for cl, g in pd.DataFrame({"y":y_cl}).groupby("y")}
        mean_mapping = {cl:yd2[index.values,:].mean(axis=0, keepdims=True) for cl, index in index_dico.items()}
        
        return mean_mapping
    
    
    def predict(self, X):
        y_hat_proba = self.classifier_model.predict_proba(X)
        
        y_hat = np.dot(y_hat_proba, self._y_mean_mapping_matrix)
        
        return make1dimension(y_hat)


from sklearn.preprocessing import OrdinalEncoder

OrdinalEncoder(dtype=np.int32).fit_transform(y_ord[:,np.newaxis])
OrdinalEncoder(dtype=np.int32).fit_transform(y_ord[:,np.newaxis].astype(str))

y = np.array(["z","a","b"]*3)
enc = OrdinalEncoder(dtype=np.int32)
enc.fit_transform(y[:, np.newaxis])[:, 0]
enc.categories_

y = np.array(["z","a","b"]*3)
enc = OrdinalEncoder(dtype=np.int32, categories=[["a","b","c","z"]])
enc.fit_transform(y[:, np.newaxis])[:, 0]
enc.categories_

# In[]

class ClassifierFromRegressor(BaseEstimator, ClassifierMixin):
    """ this class transform a regressor into a classifier
    it can be used for ordinal classification
    """

    def __init__(self,
                 regressor_model,
                 classes="auto",
                 kernel_windows=0.2
                 ):
        self.regressor_model=regressor_model
        self.classes=classes
        self.kernel_windows=kernel_windows
        
        
    def fit(self, X, y):

        ## Conversion of target into integer      
        self._target_encoder = OrdinalEncoder(dtype=np.int32) # TODO add classes here
        yd2 = make2dimensions(y)
        
        yd2_int = self._target_encoder.fit_transform(yd2)
        
        if y.ndim == 1:
            self._mono_target = True
            y_int = yd2_int[:,0]
            
            assert len(self._target_encoder.categories_) == 1
        else:
            self._mono_target = False
            y_int = yd2_int # I keep the multi-dimension
            
            assert len(self._target_encoder.categories_) == y.shape[1]
        
        self.regressor_model.fit(X, y_int)
        

        
        return self
    
    @property
    def classes_(self):
        if self._mono_target:        
            return self._target_encoder.categories_[0]
        else:
            return self._target_encoder.categories_
    
    def predict(self, X):
        
        y_hat = self.regressor_model.predict(X)     # call regressor
        y_int_hat = (y_hat + 0.5).astype(np.int32)  # conversion to closest int
        
        y_hat = self._target_encoder.inverse_transform(make2dimensions(y_int_hat))        
        
        if self._mono_target:
            y_hat = y_hat[:, 0]
            
        return y_hat
    
    def predict_proba(self, X):
        y_hat = self.regressor_model.predict(X)     # call regressor
        if self._mono_target:
            y_hat_2d = y_hat[:, np.newaxis]
            assert y_hat.ndim == 1
        else:
            y_hat_2d = y_hat
            assert y_hat.ndim == 2
            

        pivot_integers = [np.arange(len(category))[np.newaxis, :] for category in self._target_encoder.categories_]
        
        distances_to_pivot = [np.abs(y_hat_2d[:, j:(j+1)] - pivot_integers[j]) for j in range(y_hat_2d.shape[1])] # Rmk [:, j:(j+1)]  so that I keep the dimension
        
        probas = [self.distance_to_proba(distance_to_pivot) for distance_to_pivot in distances_to_pivot]

        if self._mono_target:
            return probas[0]
    
        else:
            return probas

    def distance_to_proba(self, d):
        e = np.exp(-d/self.kernel_windows)
        
        return e / e.sum(axis=1, keepdims=True)


# In[]    

from sklearn.linear_model import Ridge

X = np.random.randn(9, 2)
y = np.array(["c","a","b"]*3)

self = ClassifierFromRegressor(regressor_model=Ridge())
self.fit(X, y)

proba = self.predict_proba(X)

yhat = self.predict(X)

assert list(self.classes_) == ["a","b","c"]
assert proba.shape == (y.shape[0] , 3)
assert yhat.shape == y.shape
assert (self.classes_[proba.argmax(axis=1)] == yhat).all()
assert proba.min() >= 0
assert proba.max() <= 1
assert not pd.isnull(proba).any()
assert np.abs(proba.sum(axis=1) - 1).max() <= 0.0001

# In[]
        
self = RegressorFromClassifier(LogisticRegression())
self.fit(X, y)

yhat = self.predict(X)
yhat
y

import matplotlib.pylab as plt

plt.cla()
plt.plot(y, yhat, ".")

