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
    

# In[]
        
self = RegressorFromClassifier(LogisticRegression())
self.fit(X, y)

yhat = self.predict(X)
yhat
y

import matplotlib.pylab as plt

plt.cla()
plt.plot(y, yhat, ".")

