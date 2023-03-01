import logging

import typer
from sklearn.exceptions import ConvergenceWarning

# import scorers to add custom aikit scorers in scikit-learn SCORERS list
import aikit.scorer  # noqa
from aikit.datasets import load_dataset, DatasetEnum
from aikit.future.automl import AutoMl, TimeBudget, AutoMlConfig, load_job_config_from_json, JobConfig
from aikit.future.automl.backends import BACKEND_REGISTRY
from aikit.future.automl.guider import AutoMlModelGuider
from aikit.future.automl.result import AutoMlResultReader
from aikit.future.automl.serialization import Format

# Remove some warning categories for debugging purpose
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=ConvergenceWarning)

app = typer.Typer()

logging.basicConfig(level=logging.INFO)
# Configure some custom level for thirdparties for debugging purpose
logging.getLogger("gensim").setLevel(logging.WARNING)


@app.command()
def run(data: str, config_path: str = None, target: str = "target", backend: str = "sequential",
        cv: int = None, budget: int = 3600):
    if data in DatasetEnum.alls:
        df_train, y_train, _, _, _ = load_dataset(data)
    else:
        # TODO: load data from filesystem
        raise NotImplementedError(f"Unknown dataset: {data}")
    automl_config = AutoMlConfig(X=df_train, y=y_train)
    automl_config.guess_everything()

    if config_path is not None:
        job_config = load_job_config_from_json(config_path)
    else:
        job_config = JobConfig()
        if cv is not None:
            job_config.cv = cv
    if job_config.cv is None:
        job_config.guess_cv(automl_config)
    if job_config.scoring is None:
        job_config.guess_scoring(automl_config)

    backend = BACKEND_REGISTRY[backend.lower()]()

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
                    budget=TimeBudget(budget),
                    random_state=123)

    automl.search_models()

    df_result = result_reader.load_all_results(aggregate=True)
    print(df_result)


app()
