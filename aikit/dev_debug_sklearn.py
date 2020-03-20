# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:13:32 2020

@author: LionelMassoulard
"""

from sklearn.preprocessing import OrdinalEncoder
import numpy as np

import pickle

from aikit.tools import save_pkl


if __name__ == "__main__":

    
    X = np.array(["c","a","b"]*3)[:, np.newaxis]


    #X = np.array([10,100,1000]*3)[:, np.newaxis]


    # Cas 0 :    
    enc = OrdinalEncoder(categories="auto", dtype=np.int32)
    X_int = enc.fit_transform(X)
    X2 = enc.inverse_transform(X_int)
    assert X.dtype == X2.dtype
    assert (X2 == X).all()


    # Cas 1 : normal
    enc = OrdinalEncoder(categories="auto", dtype=np.int32)
    X_int = enc.fit_transform(X)
    X2 = enc.inverse_transform(X_int)
    assert type(X2) == type(X)
    assert X.dtype == X2.dtype
    assert (X2 == X).all()
    
    # Cas 2 : change order        
    enc = OrdinalEncoder(categories=[["c","a","b"]], dtype=np.int32)
    X_int = enc.fit_transform(X) # doesn't work ==> # ValueError: Unsorted categories are not supported for numerical categories
    X2 = enc.inverse_transform(X_int)
    assert type(X2) == type(X)
    assert X.dtype == X2.dtype
    assert (X2 == X).all()
    

    # Cas 2 bis : with an array of object    
    Xo = X.astype(np.object)
    enc = OrdinalEncoder(categories=[["c","a","b"]], dtype=np.int32) 
    X_int = enc.fit_transform(Xo) # Here it does work !
    X2 = enc.inverse_transform(X_int)
    assert type(X2) == type(Xo)
    assert Xo.dtype == X2.dtype
    assert (X2 == Xo).all()
    
    
    Xd = pd.DataFrame(X)
    enc = OrdinalEncoder(categories=[["c","a","b"]], dtype=np.int32) 
    X_int = enc.fit_transform(Xd) # Here it does work !
    X2 = enc.inverse_transform(X_int)
    assert type(X2) == type(Xd) # Fails
    assert (Xd.dtypes == X2.dtypes).all() # Fails because X2 is numpy array 
    assert (X2 == Xd).all().all() # but still ok
        
    
    
    
