# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 21:13:14 2019

@author: Lionel Massoulard
"""

import os.path

from aikit.datasets import load_dataset, DatasetEnum
from aikit.ml_machine import MlMachineLauncher

from aikit.logging import _set_logging_to_console

def loader():
    """ this is the function that should return the DataSet """
    dfX,y,_,_,_ = load_dataset(DatasetEnum.titanic)
    
    return dfX,y


if __name__ == "__main__":
    
    _set_logging_to_console()
    
    base_folder = os.path.join(os.path.expanduser('~'), "automl","titanic")
    launcher = MlMachineLauncher(base_folder=base_folder,
                                 name="titanic",
                                 loader=loader)
    
    
    launcher.execute_processed_command_argument()
