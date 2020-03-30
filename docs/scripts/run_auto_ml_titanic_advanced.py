# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 21:13:14 2019

@author: Lionel Massoulard
"""
import os.path

from aikit.datasets import load_dataset, DatasetEnum
from aikit.ml_machine import MlMachineLauncher

from aikit.logging import _set_logging_to_console

from sklearn.model_selection import StratifiedKFold

from collections import OrderedDict

def loader():
    """ this is the function that should return the DataSet """
    dfX,y,_,_,_ = load_dataset(DatasetEnum.titanic)
    
    return dfX,y


if __name__ == "__main__":
    
    _set_logging_to_console()
    
    def set_configs(launcher):
        """ this is the function that will set the different configurations """
        # Change the CV here :
        launcher.job_config.cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123) # specify CV
        
        # Change the scorer to use :
        launcher.job_config.scoring = ['accuracy', 'log_loss_patched', 'avg_roc_auc', 'f1_macro']
        
        # Change the main scorer (linked with )
        launcher.job_config.main_scorer = 'accuracy'
        
        # Change the base line (for the main scorer)
        launcher.job_config.score_base_line = 0.8
        
        # Allow 'approx cv or not : 
        launcher.job_config.allow_approx_cv = False
        
        # Allow 'block search' or not :
        launcher.job_config.do_blocks_search = True
        
        # Start with default models or not :
        launcher.job_config.start_with_default = True
        
        # Change default 'columns block' : use for block search
        launcher.auto_ml_config.columns_block = OrderedDict([
              ('NUM', ['pclass', 'age', 'sibsp', 'parch', 'fare', 'body']),
             ('TEXT', ['name', 'ticket']),
             ('CAT', ['sex', 'cabin', 'embarked', 'boat', 'home_dest'])])
    
        # Change the list of models/transformers to use :
        launcher.auto_ml_config.models_to_keep = [
                     #('Model', 'LogisticRegression'),
                     ('Model', 'RandomForestClassifier'),
                     #('Model', 'ExtraTreesClassifier'),
                     
                     # Example : keeping only RandomForestClassifer
                     
                     ('FeatureSelection', 'FeaturesSelectorClassifier'),
                     
                     ('TextEncoder', 'CountVectorizerWrapper'),
                     
                     #('TextPreprocessing', 'TextNltkProcessing'),
                     #('TextPreprocessing', 'TextDefaultProcessing'),
                     #('TextPreprocessing', 'TextDigitAnonymizer'),
                     
                     # => Example: removing TextPreprocessing
                     
                     ('CategoryEncoder', 'NumericalEncoder'),
                     ('CategoryEncoder', 'TargetEncoderClassifier'),
                     
                     ('MissingValueImputer', 'NumImputer'),
                     
                     ('DimensionReduction', 'TruncatedSVDWrapper'),
                     ('DimensionReduction', 'PCAWrapper'),
                     
                     ('TextDimensionReduction', 'TruncatedSVDWrapper'),
                     ('DimensionReduction', 'KMeansTransformer'),
                     ('Scaling', 'CdfScaler')
                     ]
        
        # Specify the type of problem
        launcher.auto_ml_config.type_of_problem = 'CLASSIFICATION'
        
        # Specify special hyper parameters : Example 
        launcher.auto_ml_config.specific_hyper = {
                ('Model', 'RandomForestClassifier') : {"n_estimators":[10,20]}
                }
        # Example : only test n_estimators to be 10 or 20
    
        return launcher

    base_folder = os.path.join(os.path.expanduser('~'), "automl","titanic")
    launcher = MlMachineLauncher(base_folder=base_folder,
                                 name="titanic",
                                 loader=loader,
                                 set_configs=set_configs)
                                 
    
    
    launcher.execute_processed_command_argument()
