from aikit import scorer  # noqa
from aikit.datasets import load_dataset
from aikit.future.automl import AutoMlConfig, JobConfig, AutoMl, TimeBudget
from aikit.future.automl.backends import BACKEND_REGISTRY
from aikit.future.automl.guider import AutoMlModelGuider
from aikit.future.automl.result import AutoMlResultReader
from aikit.future.automl.serialization import Format


def test_automl_titanic():
    df_train, y_train, _, _, _ = load_dataset("titanic")

    automl_config = AutoMlConfig(X=df_train, y=y_train)
    automl_config.guess_everything()

    job_config = JobConfig()
    job_config.cv = 3
    job_config.guess_scoring(automl_config)

    backend = BACKEND_REGISTRY["sequential"]()

    backend.get_data_loader().write(key="X", path="data", data=df_train, serialization_format=Format.PICKLE)
    backend.get_data_loader().write(key="y", path="data", data=y_train, serialization_format=Format.PICKLE)
    backend.get_data_loader().write(key="groups", path="data", data=None, serialization_format=Format.PICKLE)
    backend.get_data_loader().write(key="automl_config", path="data", data=automl_config,
                                    serialization_format=Format.PICKLE)
    backend.get_data_loader().write(key="job_config", path="data", data=job_config, serialization_format=Format.PICKLE)

    result_reader = AutoMlResultReader(backend.get_data_loader())

    automl_guider = AutoMlModelGuider(result_reader=result_reader,
                                      job_config=job_config)

    automl = AutoMl(automl_config=automl_config,
                    job_config=job_config,
                    backend=backend,
                    automl_guider=automl_guider,
                    budget=TimeBudget(10),
                    random_state=123)

    automl.search_models()

    df_result = result_reader.load_all_results(aggregate=True)
    assert len(df_result) > 0
