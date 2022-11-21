import pytest

from aikit.future.automl import AutoMlConfig
from tests.future.conftest import get_numeric_dataset, get_titanic_dataset


@pytest.fixture(params=["numeric", "titanic"])
def dataset_and_automl_config(request):
    if request.param == "numeric":
        df, y = get_numeric_dataset()
    elif request.param == "titanic":
        df, y = get_titanic_dataset()
    else:
        raise NotImplementedError(f"Unknown dataset type: {request.param}")
    automl_config = AutoMlConfig(df, y)
    automl_config.guess_everything()
    return df, y, automl_config


@pytest.fixture
def numeric_dataset_automl_config():
    df, y = get_numeric_dataset()
    automl_config = AutoMlConfig(df, y)
    automl_config.guess_everything()
    return df, y, automl_config
