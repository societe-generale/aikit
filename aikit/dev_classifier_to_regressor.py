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
            # TODO : check that it is a clusterer
            
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




# In[]
from sklearn.base import TransformerMixin

from sklearn.utils.validation import check_is_fitted
from aikit.tools.data_structure_helper import convert_generic, DataTypes

class OrdinalEncoderV2(BaseEstimator, TransformerMixin):
    
    def __init__(self,
                 categories="auto",
                 dtype=np.int32
                 ):
        self.categories=categories
        self.dtype=dtype
                
    def fit(self, X, y=None):

        X = convert_generic(X, output_type=DataTypes.NumpyArray)
        if X.ndim != 2:
            raise TypeError("This transformer expect a two dimensional array")
            
        self._nb_columns = X.shape[1]
            
        is_auto = isinstance(self.categories, str) and self.categories == "auto"
        
        if not is_auto:
            if len(self.categories) != X.shape[1]:
                raise TypeError("categories should be 'auto' or a list the same size as 'X.shape[1]'")

        all_mappings = []
        all_inv_mappings = []
        categories = []
        for j in range(X.shape[1]):
            
            if is_auto:
                target_classes_j = np.sort(np.unique(X[:, j]))
            else:
                target_classes_j = np.array(self.categories[j])
                uy_j = set(list(X[:, j]))
                if len(set(list(uy_j)).difference(target_classes_j)) > 0:
                    raise ValueError("I have a categories that doesn't exist, please chekc")

            integers = np.arange(len(target_classes_j)).astype(self.dtype)
            mapping = pd.Series(integers, index = target_classes_j)
            inv_mapping = pd.Series(target_classes_j, index = integers)
            
            all_mappings.append(mapping)
            all_inv_mappings.append(inv_mapping)
            categories.append(target_classes_j)
        
        self.categories_ = categories
        self._all_mapping = all_mappings
        self._all_inv_mapping = all_inv_mappings
        
        return self
    

    def transform(self, X):
        check_is_fitted(self)
        
        X = convert_generic(X, output_type=DataTypes.NumpyArray)

        if X.ndim != 2:
            raise TypeError("This transformer expect a two dimensional array")
            
        if X.shape[1] != self._nb_columns:
            raise ValueError("X doesn't have the correct number of columns")
            
        Xres = np.zeros(X.shape, dtype=self.dtype)
        for j in range(X.shape[1]):
            Xres[:, j] = self._all_mapping[j].loc[X[:, j]].values
        
        return Xres
    
    def inverse_transform(self, X):
        check_is_fitted(self)

        X = convert_generic(X, output_type=DataTypes.NumpyArray)

        if X.ndim != 2:
            raise TypeError("This transformer expect a two dimensional array")

        if X.shape[1] != self._nb_columns:
            raise ValueError("X doesn't have the correct number of columns")
        
        Xres = []
        for j in range(X.shape[1]):
            Xres.append(  self._all_inv_mapping[j].loc[ X[:, j]].values[:, np.newaxis] )
            
        return np.concatenate(Xres, axis=1)
            
# In[]

OrdinalEncoderV2(dtype=np.int32).fit_transform(y_ord[:,np.newaxis])
OrdinalEncoderV2(dtype=np.int32).fit_transform(y_ord[:,np.newaxis].astype(str))

y = np.array(["z","a","b"]*3)
enc = OrdinalEncoderV2(dtype=np.int32)
y_enc = enc.fit_transform(y[:, np.newaxis])
assert (enc.inverse_transform(y_enc)[:,0] == y).all()
enc.categories_

y = np.array(["z","a","b"]*3)
enc = OrdinalEncoderV2(dtype=np.int32, categories=[["a","b","c","z"]])

y_enc = enc.fit_transform(y[:, np.newaxis])
assert (enc.inverse_transform(y_enc)[:,0] == y).all()
enc.categories_

y = np.array(["z","a","b"]*3)
enc = OrdinalEncoderV2(dtype=np.int32, categories=[["z","a","b"]])
y_enc = enc.fit_transform(y[:, np.newaxis])
assert (enc.inverse_transform(y_enc)[:,0] == y).all()
enc.categories_

y = np.array(["z","a","b"]*3)
enc = OrdinalEncoder(dtype=np.int32, categories=[["z","a","b"]])
y_enc = enc.fit_transform(y[:, np.newaxis].astype(np.object))
assert (enc.inverse_transform(y_enc)[:,0] == y).all()
enc.categories_



# In[]

class ClassifierFromRegressor(BaseEstimator, ClassifierMixin):
    """ this class transform a regressor into a classifier
    it can be used for ordinal classification
    """

    def __init__(self,
                 regressor_model,
                 kernel_windows=0.2,
                 classes=None
                 ):
        self.regressor_model=regressor_model
        self.classes=classes
        self.kernel_windows=kernel_windows


    def fit(self, X, y):

        self._mono_target = y.ndim == 1

        ## Conversion of target into integer      
        if isinstance(self.classes, str) and self.classes == "auto":
            categories = "auto"
        else:
            if self._mono_target:
                categories = [self.classes] # because OrdinalEncoder expect a list
            else:
                if not isinstance(self.classes, list):
                    raise TypeError("For multi-target classes should be a list, instead I got %s" % str(type(self.classes)))
                
                categories = self.classes
            
        self._target_encoder = OrdinalEncoder(dtype=np.int32, categories=categories) # ca ne marche pas si les classes sont pas ordonnées !
        
        yd2 = convert_generic(make2dimensions(y), output_type=DataTypes.NumpyArray)

        if yd2.dtype.kind == 'U':
            yd2 = yd2.astype(np.object, copy=False)
        
        yd2_int = self._target_encoder.fit_transform(yd2)
        
        if self._mono_target:
            y_int = yd2_int[:,0]
            assert len(self._target_encoder.categories_) == 1
        else:
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
        """ convert a distance to a probability """
        e = np.exp(-d/self.kernel_windows) # TODO : find a good heuristic for that kernel_windows
        
        return e / e.sum(axis=1, keepdims=True)


# In[]    

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

import itertools

import pytest



@pytest.mark.parametrize("string_classes, change_order, multi_target", list(itertools.product((True, False), (True, False), (True, False))))
def test_ClassifierFromRegressor(string_classes, change_order, multi_target):

    if change_order and not string_classes:
        return # this is not a valid test
    
    np.random.seed(123)
    
    X = np.random.randn(9, 2)
    y_int = np.array([2,0,1]*3)
    
    if string_classes:
        y = np.array(["a","b","c"])[y_int]
    else:
        y = y_int
                
    if change_order:
        if string_classes:
            classes = ["c", "a", "b"]
            expected_classes = classes
    else:
        if string_classes:
            classes = "auto"
            expected_classes = ["a", "b", "c"]
        else:
            classes = "auto"
            expected_classes = [0, 1, 2]

    if multi_target:
        y = np.repeat(y[:,np.newaxis], 2, axis=1)
        expected_classes = [expected_classes, expected_classes]

        if classes != "auto":
            classes = [classes, classes]
            

    classifier = ClassifierFromRegressor(regressor_model=DecisionTreeRegressor(random_state=123), classes=classes)
    classifier.fit(X, y)
    
    proba = classifier.predict_proba(X)
    
    yhat = classifier.predict(X)
    
    assert (yhat == y).all() # DecisionTree should be able to overfit easily : so that I can check that 
    assert type(yhat) == type(y)
    assert yhat.dtype == yhat.dtype
    assert yhat.shape == y.shape

    if multi_target:

        assert isinstance(classifier.classes_, list)
        assert len(classifier.classes_) == 2
        for c, e in zip(classifier.classes_, expected_classes):
            assert list(c) == list(e)
        
        assert isinstance(proba, list)
        assert len(proba) == 2
        for j, p in enumerate(proba):
            assert p.shape == (y.shape[0] , 3)
            
            assert (classifier.classes_[j][p.argmax(axis=1)] == yhat[:,j]).all()

            assert p.min() >= 0
            assert p.max() <= 1
            
            assert not pd.isnull(p).any()
            assert np.abs(p.sum(axis=1) - 1).max() <= 0.0001

    else:
        assert list(classifier.classes_) == expected_classes
        assert proba.shape == (y.shape[0] , 3)
        assert (classifier.classes_[proba.argmax(axis=1)] == yhat).all()
        assert proba.min() >= 0
        assert proba.max() <= 1
        

        assert not pd.isnull(proba).any()
        assert np.abs(proba.sum(axis=1) - 1).max() <= 0.0001


# In[]

for string_classes, change_order, multi_target in itertools.product((True, False), (True, False), (True, False)):
    print("a")
    test_ClassifierFromRegressor(string_classes, change_order, multi_target)

#X = np.random.randn(9, 2)
#y = np.array(["c","a","b"]*3)

#self = ClassifierFromRegressor(regressor_model=Ridge(), classes=["c","a","b"])
#self.fit(X, y)

#proba = self.predict_proba(X)

#yhat = self.predict(X)

#assert list(self.classes_) == ["c","a","b"]
#assert proba.shape == (y.shape[0] , 3)
#assert yhat.shape == y.shape
#assert (self.classes_[proba.argmax(axis=1)] == yhat).all()
#assert proba.min() >= 0
#assert proba.max() <= 1
#assert not pd.isnull(proba).any()
#assert np.abs(proba.sum(axis=1) - 1).max() <= 0.0001





# In[]
import matplotlib.pylab as plt


self = RegressorFromClassifier(LogisticRegression())
self.fit(X, y)

yhat = self.predict(X)
yhat
y

assert y.shape == yhat.shape

# In[] 
y = np.array(["z","a","b"]*3)[:, np.newaxis]
enc = OrdinalEncoder(dtype=np.int32)

y_int = enc.fit_transform(y)[:,0]

y_int = np.array([0,1,2,3])

y_one_hot = np.zeros(shape=(y_int.shape[0], len(y_int)), dtype=np.int32)

line = np.arange(y_one_hot.shape[0])
y_one_hot[line, y_int] = 1
y_one_hot[line, np.maximum(y_int-1,0)] = 1
y_one_hot[line, np.maximum(y_int-2,0)] = 1
y_one_hot[line, np.maximum(y_int-3,0)] = 1
y_one_hot[:, 1:]
