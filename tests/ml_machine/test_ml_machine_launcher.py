# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:01:20 2019

@author: lmassoul032513
"""

import numpy as np

from aikit.datasets.datasets import load_dataset, DatasetEnum
from aikit.ml_machine.ml_machine_launcher import MlMachineLauncher
from aikit.enums import TypeOfProblem

from sklearn.model_selection import LeaveOneOut


def loader():
    """ modify this function to load the data
    
    Returns
    -------
    dfX, y 
    
    Or
    dfX, y, groups
    
    """
    dfX, y, _, _, _ = load_dataset(DatasetEnum.titanic)
    return dfX, y


def test_launcher_init(tmpdir):
    def set_configs(launcher):
        """ modify that function to change launcher configuration """
        return launcher

    launcher = MlMachineLauncher(base_folder=tmpdir, name="titanic", loader=loader, set_configs=set_configs)

    launcher.initialize()

    assert launcher.auto_ml_config is not None
    assert launcher.data_persister is not None
    assert launcher.job_config is not None

    # auto-ml-config check
    assert launcher.auto_ml_config.type_of_problem == TypeOfProblem.CLASSIFICATION

    # job-config
    assert launcher.job_config.cv is not None
    assert launcher.job_config.scoring is not None
    assert isinstance(launcher.job_config.scoring, list)


def numpy_loader():
    np.random.seed(123)
    X = np.random.randn(100, 10).astype(np.float32)
    y = X[:,0] + np.random.randn(100)
    return X, y


def test_launcher_init_with_numpy(tmpdir):
    launcher = MlMachineLauncher(base_folder=tmpdir, name="numpy", loader=numpy_loader)

    launcher.initialize()

    assert launcher.auto_ml_config is not None
    assert launcher.data_persister is not None
    assert launcher.job_config is not None

    # auto-ml-config check
    assert launcher.auto_ml_config.type_of_problem == TypeOfProblem.REGRESSION

    # job-config
    assert launcher.job_config.cv is not None
    assert launcher.job_config.scoring is not None
    assert isinstance(launcher.job_config.scoring, list)


def test_launcher_init_persist_reload(tmpdir):
    def set_configs(launcher):
        """ modify that function to change launcher configuration """
        return launcher

    launcher = MlMachineLauncher(base_folder=tmpdir, name="titanic", loader=loader, set_configs=set_configs)

    launcher.initialize()
    launcher.persist()
    launcher.reload()

    assert launcher.auto_ml_config is not None
    assert launcher.data_persister is not None
    assert launcher.job_config is not None

    # auto-ml-config check
    assert launcher.auto_ml_config.type_of_problem == TypeOfProblem.CLASSIFICATION

    # job-config
    assert launcher.job_config.cv is not None
    assert launcher.job_config.scoring is not None
    assert isinstance(launcher.job_config.scoring, list)

    def loader_error():
        assert False  # This function should not be called AGAIN here

    def set_configs_error(launcher):
        assert False  # This function should not be called AGAIN here

    new_launcher = MlMachineLauncher(
        base_folder=tmpdir, name="titanic", loader=loader_error, set_configs=set_configs_error
    )

    new_launcher.reload()  # So that I ma
    assert new_launcher.auto_ml_config is not None
    assert new_launcher.data_persister is not None
    assert new_launcher.job_config is not None

    # auto-ml-config check
    assert new_launcher.auto_ml_config.type_of_problem == TypeOfProblem.CLASSIFICATION

    # job-config
    assert new_launcher.job_config.cv is not None
    assert new_launcher.job_config.scoring is not None
    assert isinstance(new_launcher.job_config.scoring, list)


def test_launcher_init_persist_reload_with_group(tmpdir):
    def set_configs(launcher):
        """ modify that function to change launcher configuration """
        return launcher

    def new_loader():
        dfX, y = loader()
        groups = np.arange(len(y))
        return dfX, y, groups

    launcher = MlMachineLauncher(base_folder=tmpdir, name="titanic", loader=new_loader, set_configs=set_configs)

    launcher.initialize()
    launcher.persist()
    launcher.reload()

    assert launcher.auto_ml_config is not None
    assert launcher.data_persister is not None
    assert launcher.job_config is not None

    assert launcher.dfX is not None
    assert launcher.y is not None
    assert launcher.groups is not None

    # auto-ml-config check
    assert launcher.auto_ml_config.type_of_problem == TypeOfProblem.CLASSIFICATION

    # job-config
    assert launcher.job_config.cv is not None
    assert launcher.job_config.scoring is not None
    assert isinstance(launcher.job_config.scoring, list)

    def loader_error():
        assert False  # This function should not be called AGAIN here

    def set_configs_error(launcher):
        assert False  # This function should not be called AGAIN here

    new_launcher = MlMachineLauncher(
        base_folder=tmpdir, name="titanic", loader=loader_error, set_configs=set_configs_error
    )

    new_launcher.reload()  # So that I ma
    assert new_launcher.auto_ml_config is not None
    assert new_launcher.data_persister is not None
    assert new_launcher.job_config is not None

    # auto-ml-config check
    assert new_launcher.auto_ml_config.type_of_problem == TypeOfProblem.CLASSIFICATION

    # job-config
    assert new_launcher.job_config.cv is not None
    assert new_launcher.job_config.scoring is not None
    assert isinstance(new_launcher.job_config.scoring, list)


def test_launcher_set_configs(tmpdir):
    def set_configs(launcher):
        """ modify that function to change launcher configuration """

        launcher.job_config.cv = LeaveOneOut()
        return launcher

    launcher = MlMachineLauncher(base_folder=tmpdir, name="titanic", loader=loader, set_configs=set_configs)

    launcher.initialize()
    assert isinstance(launcher.job_config.cv, LeaveOneOut)


def test_launcher_init_clustering(tmpdir):
    def set_configs(launcher):
        """ modify that function to change launcher configuration """
        return launcher

    def new_loader():
        dfX, y = loader()
        return dfX, None

    launcher = MlMachineLauncher(base_folder=tmpdir, name="titanic", loader=new_loader, set_configs=set_configs)

    launcher.initialize()

    assert launcher.auto_ml_config is not None
    assert launcher.data_persister is not None
    assert launcher.job_config is not None

    # auto-ml-config check
    assert launcher.auto_ml_config.type_of_problem == TypeOfProblem.CLUSTERING

    # job-config
    assert launcher.job_config.cv is not None
    assert launcher.job_config.scoring is not None
    assert isinstance(launcher.job_config.scoring, list)
