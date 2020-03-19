# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:17:15 2020

@author: LionelMassoulard
"""


from sklearn.base import clone

from aikit.transformers import ModelWrapper
from aikit.enums import DataTypes

def AutoWrapper_from_klass(klass):
    
    class WrappedKlass(ModelWrapper):
        
        def __init__(self,
                     columns_to_use="all",
                     regex_match=False,
                     desired_output_type=DataTypes.DataFrame,
                     drop_used_columns=True,
                     drop_unused_columns=False,
                     ):
            
            self.columns_to_use=columns_to_use
            self.regex_match=regex_match
            self.desired_output_type=desired_output_type
            self.drop_used_columns=drop_used_columns
            self.drop_unused_columns=drop_unused_columns
            
            
            super(WrappedKlass, self).__init__(self,
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
                 drop_unused_columns=drop_unused_columns
                 )
            
        def _get_model(self, X, y=None):
            return klass()
        

def AutoWrapper_from_klass_with_kwargs(klass):
    
    class WrappedKlass(ModelWrapper):
        
        def __init__(self,
                     columns_to_use="all",
                     regex_match=False,
                     desired_output_type=DataTypes.DataFrame,
                     drop_used_columns=True,
                     drop_unused_columns=False,
                     **kwargs
                     ):
            
            self.columns_to_use=columns_to_use
            self.regex_match=regex_match
            self.desired_output_type=desired_output_type
            self.drop_used_columns=drop_used_columns
            self.drop_unused_columns=drop_unused_columns
            
            self.kwargs = kwargs
            
            # En l'etat on ne peut pas faire ca : 
            # sklearn n'accepte pas le kwargs (ne marche pas avec get_params)
            
            # Solution 1 :
            # bricoler pour que la signature remarche (ce qui doit être possible)
            # cf ce qui est fais dans les decorator
            
            # Solution 2
            # re-coder get_params et set_params pour qu'il marche MEME avec des kwargs
            
            
            super(WrappedKlass, self).__init__(self,
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
                 drop_unused_columns=drop_unused_columns
                 )
            
        def _get_model(self, X, y=None):
            return klass(**self.kwargs)
        
    return WrappedKlass


def AutoWrapper_from_model(model):
    
    class WrappedKlass(ModelWrapper):
        
        def __init__(self,
                     columns_to_use="all",
                     regex_match=False,
                     desired_output_type=DataTypes.DataFrame,
                     drop_used_columns=True,
                     drop_unused_columns=False,
                     ):
            
            self.columns_to_use=columns_to_use
            self.regex_match=regex_match
            self.desired_output_type=desired_output_type
            self.drop_used_columns=drop_used_columns
            self.drop_unused_columns=drop_unused_columns
            
            
            super(WrappedKlass, self).__init__(
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
                 drop_unused_columns=drop_unused_columns
                 )
            
        def _get_model(self, X, y=None):
            return clone(model)
        
    return WrappedKlass
            

# In[] :
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd

X = np.random.randn(100,10)
df = pd.DataFrame(X, columns=[f"NUMBER_{j}" for j in range(X.shape[1])])
df["not_a_number"] = "a"

# Utilisation 1 : from a model
model = AutoWrapper_from_model(TruncatedSVD(n_components=2))(columns_to_use=["NUMBER_"], regex_match=True)
model.fit_transform(df)
