# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:32:08 2020

@author: LionelMassoulard
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, is_classifier, is_regressor
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer


from aikit.tools.data_structure_helper import make2dimensions, convert_generic
from aikit.enums import DataTypes
from aikit.transformers.categories import _OrdinalOneHotEncoder



class _BaseOrdinalClassifier(BaseEstimator, ClassifierMixin):
    """ this class is the base class for Ordinal classifier,
    it contains methods to convert the target into another format that will be used to fit the underlying classifier
    
    """
    def _prepare_target(self, y, klass, conversion_type):
        """ prepare the target so that it can be given to the underlying model to use
        
        Parameters
        ----------
        
        y : array
            the original target 
            
        klass : type
            the encoder to use for the target
            
        conversion_type : DataType
            the output type desired by the target
            
        Set
        ---
        self._mono_target : bool 
            does the original problem as one target or not
        self._target_encoded : the encoder used on the target
        
        Returns
        --------
        y_encoded : array
            the modified target
        """
        self._mono_target = y.ndim == 1
        self._target_dtype = y.dtype
        
        if isinstance(self.classes, str) and self.classes == "auto":
            categories = "auto"
        else:
            if self._mono_target:
                categories = [self.classes] # because OrdinalEncoder expect a list
            else:
                if not isinstance(self.classes, list):
                    raise TypeError("For multi-target classes should be a list, instead I got %s" % str(type(self.classes)))
                
                categories = self.classes

        self._target_encoder = klass(categories=categories, dtype=np.int32)
        
        yd2 = convert_generic(make2dimensions(y), output_type=conversion_type)
  
        if conversion_type == DataTypes.NumpyArray and yd2.dtype.kind == 'U':
            yd2 = yd2.astype(np.object, copy=False)
        
        y_encoded = self._target_encoder.fit_transform(yd2)

        return y_encoded


    @property
    def classes_(self):
        if self._mono_target:        
            return self._target_encoder.categories_[0]
        else:
            return self._target_encoder.categories_

class ClassifierFromRegressor(_BaseOrdinalClassifier):
    """ this class transform a regressor into a classifier
    it can be used for ordinal classification
    
    This model will transform the target into an increasing list of integer
    and then fit a regression on that
    """

    def __init__(self,
                 regressor,
                 kernel_windows=0.2,
                 classes="auto"
                 ):
        self.regressor=regressor
        self.classes=classes
        self.kernel_windows=kernel_windows


    def fit(self, X, y):
        
        if not is_regressor(self.regressor):
            raise TypeError("regressor should be a sklearn regressor")

        y_encoded = self._prepare_target(y, klass=OrdinalEncoder, conversion_type=DataTypes.NumpyArray)

        if self._mono_target:
            y_int = y_encoded[:,0]
            assert len(self._target_encoder.categories_) == 1
        else:
            y_int = y_encoded # I keep the multi-dimension            
            assert len(self._target_encoder.categories_) == y.shape[1]

        self.regressor.fit(X, y_int)
        
        return self


    def predict(self, X):
        y_hat = self.regressor.predict(X)     # call regressor
        y_int_hat = (y_hat + 0.5).astype(np.int32)  # conversion to closest int
        
        y_hat = self._target_encoder.inverse_transform(make2dimensions(y_int_hat))        
        
        if self._mono_target:
            y_hat = y_hat[:, 0]

        return y_hat.astype(self._target_dtype)


    def predict_proba(self, X):
        y_hat = self.regressor.predict(X)     # call regressor
        if self._mono_target:
            y_hat_2d = y_hat[:, np.newaxis]
            assert y_hat.ndim == 1
        else:
            y_hat_2d = y_hat
            assert y_hat.ndim == 2

        probas = []
        for j, category in enumerate(self._target_encoder.categories_):
            pivot_integer = np.arange(len(category))[np.newaxis, :]
            distance_to_pivot = np.abs(y_hat_2d[:, j:(j+1)] - pivot_integer)
            proba = self.distance_to_proba(distance_to_pivot)

            probas.append(proba)


        if self._mono_target:
            return probas[0]
    
        else:
            return probas


    def distance_to_proba(self, d):
        """ convert a distance to a probability """
        e = np.exp(-d/self.kernel_windows) # TODO : find a good heuristic for that kernel_windows
        
        return e / e.sum(axis=1, keepdims=True)


class OrdinalClassifier(_BaseOrdinalClassifier):
    """ This class transform a classifier to make it more suited to ordinal classification.
    It does so by changing the Target using the OrdinalOneHotEncoder transformer
    
    Concretely if we have 4 ordered classes 'y=A', 'y=B', 'y=C', 'y=D' with ('A' < 'B' < 'C' < 'D') 
    It creates 3 targets :
        'y > A' , 'y>B' and 'y>C'
        
    The classifier is then fitted on those target.
    
    At the end to make a prediction we call the underlying classifier and recreates the proba
    
    See the paper 
    https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf
    for more detailed explanation
    
    """
    def __init__(self, classifier, classes="auto"):
        self.classifier = classifier
        self.classes=classes
 
       
    def fit(self, X , y):
        
        if not is_classifier(self.classifier):
            raise TypeError("classifier should be a sklearn classifier")

        y_int = self._prepare_target(y, klass = _OrdinalOneHotEncoder, conversion_type=DataTypes.DataFrame)
        y_int = convert_generic(y_int, output_type=DataTypes.NumpyArray)
        
        self.classifier.fit(X, y_int)
        
        return self
    
    
    @staticmethod
    def _aggregate_probas_over(probas_over):
        """ helper method to go from the probabilities that the target is stricly above something to the proba of each class """
        
        # For example, if we have 4 ordered classes 'A', 'B', 'C' and 'D'
        #
        # probas_over = [ proba(y > A), proba(y > B), proba(y > C)] . So a list of 3 (= 4 -1 ) elements
        #        
        # This corresponds to the probas of target above something
        # probas_over[j] := P( Y > classes[j] ) #
        
        # To go back to P( Y == classes[j] ) I need to do
        
        # P( Y == classes[0]   ) = 1- P(Y > classes[0] )   # smallest target
        # P( Y == classes[j]   ) = P( Y > classes[j-1] ) - P( Y > classes[j] )  # intermediate target
        # P( Y == classes[J-1] ) = P( Y > classes[J-2] )   # highest target
        
        J = len(probas_over)
        
        classes_proba = []
        classes_proba.append( 1- probas_over[0] )
        for j in range(1, J):
            classes_proba.append( probas_over[j-1] - probas_over[j] )
        classes_proba.append(probas_over[J-1])
        
        probas = np.concatenate(classes_proba, axis=1)
        
        return probas


    def predict_proba(self, X):
        
        # Call classifier
        y_hat_proba = self.classifier.predict_proba(X)
        

        # Retrive the proba of 1 from proba matrix
        probas_over = []
        for proba, cl in zip(y_hat_proba, self.classifier.classes_):
            if cl[0] == 1:
                p = proba[:,0]
            else:
                p = proba[:,1] # should always be the case
                
            assert len(cl) == 2
            assert cl.tolist() == [0,1]
            assert proba.shape[1] == 2

            probas_over.append(p[:, np.newaxis])
        
        # Now let's re-aggregate the proba of each class
        probas = []
        start_index = 0
        for target_index, categories in enumerate(self._target_encoder.categories_):

            end_index = start_index + len(categories) - 1 # start and end of current target

            probas_over_for_target = probas_over[start_index:end_index]
            p = self._aggregate_probas_over(probas_over_for_target)
            probas.append(p)
            
            start_index = end_index
            
        if self._mono_target:
            return probas[0]
        else:
            return probas
        
        
    def predict(self, X):
        probas = self.predict_proba(X)
        
        classes = self.classes_
        
        if self._mono_target:
            y_hat = classes[probas.argmax(axis=1)]
        
        else:
            
            all_pred = [ c[p.argmax(axis=1)][:, np.newaxis] for p, c in zip(probas, classes)]
            y_hat = np.concatenate(all_pred, axis=1)
            
        return y_hat.astype(self._target_dtype)



class RegressorFromClassifier(BaseEstimator, RegressorMixin):
    """ Transform a Classifier into a regressor
    
    does it by clustering the target and fit a classification
    """
    
    def __init__(self,
                 classifier,
                 strategy="kmeans",
                 n_bins=10,
                 y_clusterer=None
                 ):
        self.classifier=classifier
        self.strategy=strategy
        self.n_bins=n_bins
        self.y_clusterer=y_clusterer
        
        
    def get_default_y_cluster(self, y=None):
        """ this methods returns the default clusterer to use, if y_clusterer is None """
        return KBinsDiscretizer(n_bins=self.n_bins, strategy=self.strategy, encode="ordinal")
  
    
    def fit(self, X, y):
        
        self._mono_target = y.ndim == 1
        
        if self.y_clusterer is None:
            y_clusterer = self.get_default_y_cluster(y)
        else:
            y_clusterer = self.y_clusterer
            # TODO : check that it is a clusterer
            
        if not is_classifier(self.classifier):
            raise TypeError("classifier should be a classifer")
            
        yd2 = make2dimensions(y)
        
        if hasattr(y_clusterer, "fit_predict"):
            y_cl = y_clusterer.fit_predict(yd2)
        else:
            y_cl = y_clusterer.fit_transform(yd2).astype('int32')
        
        if y_cl.ndim == 1:
            y_cl = y_cl[:, np.newaxis]

        if self._mono_target and y_cl.shape[1] > 1:
            raise ValueError("The cluster should return only 1 dimensional clusters")
 
        self._mono_cluster = y_cl.shape[1] == 1
 
        self.classifier.fit(X, y_cl) # fit classifier on result of cluster
        
        if self._mono_cluster:
            classes = [self.classifier.classes_]
        else:
            classes = self.classifier.classes_
            
        all_mean_mapping = self._compute_y_mean(yd2, y_cl)
        
        all_y_mean_mapping_matrix = []
        for classe, y_mean_mapping in zip(classes, all_mean_mapping):
            mat = np.concatenate([y_mean_mapping[cl] for cl in classe], axis=0)
            all_y_mean_mapping_matrix.append(mat)
            
        self._all_y_mean_matrix = all_y_mean_mapping_matrix
        
        return self
    
    
    def _compute_y_mean(self, yd2, y_cl):
        """ compute the mean of each target within each cluster
        Those value will be needed in order to make the final predictions
        """
        assert y_cl.ndim == 2
        
        all_mean_mapping = []
        for j in range(y_cl.shape[1]):      
            index_dico = {cl : g.index for cl, g in pd.DataFrame({"y":y_cl[:, j]}).groupby("y")}
            if self._mono_cluster and not self._mono_target:
                # it means that 
                # 1. I have more than one target ...
                # 2. ... but the cluster returns one dimension only
                mean_mapping = {cl:yd2[index.values,:].mean(axis=0, keepdims=True) for cl, index in index_dico.items()}
            else:
                mean_mapping = {cl:yd2[index.values,j:(j+1)].mean(axis=0, keepdims=True) for cl, index in index_dico.items()}
        
            all_mean_mapping.append(mean_mapping)
            
        return all_mean_mapping # for each cluster, mean of each target
    
    
    def predict(self, X):
        
        y_hat_probas = self.classifier.predict_proba(X)
        
        if self._mono_cluster:
            y_hat_probas = [y_hat_probas]

        y_hats = [ np.dot(y_hat_proba, y_mean_mapping_matrix) for y_hat_proba, y_mean_mapping_matrix in zip(y_hat_probas, self._all_y_mean_matrix) ]
        
        if len(y_hats) > 1:
            y_hat = np.concatenate(y_hats, axis=1)
        else:
            y_hat = y_hats[0]
        
        if self._mono_target:
            return y_hat[:, 0]
        else:
            return y_hat




