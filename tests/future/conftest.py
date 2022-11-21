import numpy as np
import pandas as pd
import pytest

from aikit.datasets import load_dataset, DatasetEnum


def get_numeric_dataset():
    np.random.seed(123)
    df = pd.DataFrame(np.random.randn(100, 10), columns=["COL_%d" % d for d in range(10)])
    y = 1 * (np.random.randn(100) > 0)
    return df, y


def get_titanic_dataset():
    df, y, _, _, _ = load_dataset(DatasetEnum.titanic)
    return df, y


@pytest.fixture
def numeric_dataset():
    return get_numeric_dataset()


@pytest.fixture
def titanic_dataset():
    return get_titanic_dataset()
