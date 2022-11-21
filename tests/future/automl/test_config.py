# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:20:02 2019

@author: lmassoul032513
"""
import numpy as np
import pandas as pd
import pytest

from aikit.future.automl import AutoMlConfig
from aikit.future.enums import ProblemType


def test_automl_config_raise_if_wrong_nb_observations():
    df = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [0, 10, 20, 30, 40, 50]})
    y = np.array([0, 0, 0, 1, 1, 1])

    auto_ml_config = AutoMlConfig(df, y[0:3])
    with pytest.raises(ValueError):
        auto_ml_config.guess_everything()  # raise because y doesn't have the correct number of observations


def test_automl_config_raise_multi_output():
    df = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [0, 10, 20, 30, 40, 50]})
    y = np.array([0, 0, 0, 1, 1, 1])
    y2d = np.concatenate((y[:, np.newaxis], y[:, np.newaxis]), axis=1)

    auto_ml_config = AutoMlConfig(df, y2d)
    with pytest.raises(ValueError):
        auto_ml_config.guess_everything()  # raise because y has 2 dimensions


def test_automl_config(dataset_and_automl_config):
    df, y, automl_config = dataset_and_automl_config

    assert automl_config.problem_type == ProblemType.CLASSIFICATION
    assert automl_config.columns_informations is not None

    # Tests on needed steps
    def _check_steps(_automl_config):
        assert hasattr(_automl_config, "needed_steps")
        assert isinstance(_automl_config.needed_steps, list)
        for step in _automl_config.needed_steps:
            assert isinstance(step, dict)
            assert set(step.keys()) == {"optional", "step"}
            assert isinstance(step["optional"], bool)
            assert isinstance(step["step"], str)

    _check_steps(automl_config)
    assert "Model" in [step["step"] for step in automl_config.needed_steps]
    assert "Scaling" in [step["step"] for step in automl_config.needed_steps]

    # Try assigning to needed steps
    automl_config.needed_steps = [s for s in automl_config.needed_steps if s["step"] != "Scaling"]

    _check_steps(automl_config)
    assert "Model" in [step["step"] for step in automl_config.needed_steps]
    assert "Scaling" not in [step["step"] for step in automl_config.needed_steps]

    with pytest.raises(TypeError):
        automl_config.needed_steps = "this shouldn't be accepted has steps"

    _check_steps(automl_config)

    # Tests on models to keep
    def _check_models(_automl_config):
        assert hasattr(_automl_config, "models_to_keep")
        assert isinstance(_automl_config.models_to_keep, list)
        for model in _automl_config.models_to_keep:
            assert isinstance(model, tuple)
            assert len(model) == 2
            assert isinstance(model[0], str)
            assert isinstance(model[1], str)

    _check_models(automl_config)

    assert ("Model", "LogisticRegression") in automl_config.models_to_keep
    assert ("Model", "RandomForestClassifier") in automl_config.models_to_keep
    assert ("Model", "ExtraTreesClassifier") in automl_config.models_to_keep

    # try assignation
    automl_config.models_to_keep = [m for m in automl_config.models_to_keep if m[1] != "LogisticRegression"]

    with pytest.raises(TypeError):
        automl_config.models_to_keep = "this shouldn't be accepted has models_to_keep"

    _check_models(automl_config)
    assert ("Model", "LogisticRegression") not in automl_config.models_to_keep
    assert ("Model", "RandomForestClassifier") in automl_config.models_to_keep
    assert ("Model", "ExtraTreesClassifier") in automl_config.models_to_keep

    automl_config.filter_models(Model="ExtraTreesClassifier")

    _check_models(automl_config)
    assert ("Model", "LogisticRegression") not in automl_config.models_to_keep
    assert ("Model", "RandomForestClassifier") not in automl_config.models_to_keep
    assert ("Model", "ExtraTreesClassifier") in automl_config.models_to_keep


def test_automl_config_change_type_of_problem(numeric_dataset_automl_config):
    X, y, automl_config = numeric_dataset_automl_config

    assert automl_config.problem_type == "CLASSIFICATION"
    assert ("Model", "RandomForestClassifier") in automl_config.models_to_keep
    assert ("Model", "RandomForestRegressor") not in automl_config.models_to_keep

    with pytest.raises(ValueError):
        automl_config.problem_type = "NOT_ALLOWED"

    automl_config.problem_type = "REGRESSION"
    assert automl_config.problem_type == "REGRESSION"
    assert ("Model", "RandomForestClassifier") not in automl_config.models_to_keep
    assert ("Model", "RandomForestRegressor") in automl_config.models_to_keep
