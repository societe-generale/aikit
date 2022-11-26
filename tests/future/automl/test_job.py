import pytest

from aikit.future.automl._job import JobConfig


def test_job_config(numeric_dataset_automl_config):
    df, y, _, automl_config = numeric_dataset_automl_config

    job_config = JobConfig()
    job_config.guess_cv(automl_config)

    assert job_config.cv is not None
    assert hasattr(job_config.cv, "split")

    job_config.guess_scoring(automl_config)
    assert isinstance(job_config.scoring, list)

    assert hasattr(job_config, "allow_approx_cv")
    assert hasattr(job_config, "start_with_default")
    assert hasattr(job_config, "do_blocks_search")

    assert isinstance(job_config.allow_approx_cv, bool)
    assert isinstance(job_config.start_with_default, bool)
    assert isinstance(job_config.do_blocks_search, bool)

    with pytest.raises(ValueError):
        job_config.cv = "this is not a cv"


def test_job_config_additional_scoring_function():
    job_config = JobConfig()

    assert job_config.additional_scoring_function is None

    def f(x):
        return x + 1

    job_config.additional_scoring_function = f
    assert job_config.additional_scoring_function is not None
    assert job_config.additional_scoring_function(1) == 2

    with pytest.raises(TypeError):
        job_config.additional_scoring_function = 10  # no a function

    def f(x):
        return x + 1

    job_config.additional_scoring_function = f
    assert job_config.additional_scoring_function is not None
    assert job_config.additional_scoring_function(1) == 2

    with pytest.raises(TypeError):
        job_config.additional_scoring_function = 10  # no a function
