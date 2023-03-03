import logging
import os
import uuid
# Remove some warning categories for debugging purpose
from warnings import simplefilter

import pandas as pd
import typer
from sklearn.exceptions import ConvergenceWarning

# import scorers to add custom aikit scorers in scikit-learn SCORERS list
import aikit.scorer  # noqa
from aikit.datasets import load_dataset, DatasetEnum
from aikit.future.automl import AutoMl, TimeBudget, AutoMlConfig, load_job_config_from_json, JobConfig
from aikit.future.automl._automl import ModelCountBudget
from aikit.future.automl.backends import get_backend, filter_backend_kwargs
from aikit.future.automl.guider import AutoMlModelGuider
from aikit.future.automl.result import AutoMlResultReader
from aikit.future.automl.serialization import Format

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=ConvergenceWarning)
simplefilter(action='ignore', category=UserWarning)


app = typer.Typer()

# Configure logging
logging.basicConfig(level=logging.INFO)
# Configure some custom level for third-parties for debugging purpose
logging.getLogger("gensim").setLevel(logging.WARNING)

_logger = logging.getLogger(__name__)


@app.command()
def run(data: str,
        config_path: str = None,
        target: str = "target",
        backend: str = "sequential",
        session: str = None,
        cv: int = None,
        baseline: float = None,
        budget_model_count: int = None,
        budget_time: int = None,
        dask_storage_path: str = os.path.join(os.path.expanduser("~"), ".aikit", "working_dir"),
        dask_cluster: str = "local",
        dask_num_workers: int = 1):

    if session is None:
        session = str(uuid.uuid4())
    _logger.info(f"Start AutoML, session: {session}")

    # Register in this dictionary all arguments that must be passed to the backend
    backend_kwargs = {
        "dask_storage_path": dask_storage_path,
        "dask_cluster": dask_cluster,
        "dask_num_workers": dask_num_workers,
    }
    backend_kwargs = filter_backend_kwargs(backend, **backend_kwargs)

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
        if baseline is not None:
            job_config.baseline_score = baseline
    if job_config.cv is None:
        job_config.guess_cv(automl_config)
    if job_config.scoring is None:
        job_config.guess_scoring(automl_config)

    if budget_time is not None:
        budget = TimeBudget(budget_time)
    elif budget_model_count is not None:
        budget = ModelCountBudget(budget_model_count)
    else:
        raise ValueError("'budget_time' or 'budget_model_count' must be set")

    # TODO: force seed of workers in the backend
    with get_backend(backend, session=session, **backend_kwargs) as backend:
        # TODO: add dedicated methods in backend to write common data
        backend.get_data_loader().write(key="X", path="data", data=df_train, serialization_format=Format.PICKLE)
        backend.get_data_loader().write(key="y", path="data", data=y_train, serialization_format=Format.PICKLE)
        backend.get_data_loader().write(key="groups", path="data", data=None, serialization_format=Format.PICKLE)
        backend.get_data_loader().write(key="automl_config", path="data", data=automl_config,
                                        serialization_format=Format.PICKLE)
        backend.get_data_loader().write(key="job_config", path="data", data=job_config,
                                        serialization_format=Format.PICKLE)

        result_reader = AutoMlResultReader(backend.get_data_loader())

        automl_guider = AutoMlModelGuider(result_reader=result_reader,
                                          job_config=job_config)

        automl = AutoMl(automl_config=automl_config,
                        job_config=job_config,
                        backend=backend,
                        automl_guider=automl_guider,
                        budget=budget,
                        random_state=123)

        automl.search_models()

        df_result = result_reader.load_all_results(aggregate=True)
        print(df_result)

    _logger.info(f"Finished searching models, session: {session}")


@app.command()
def result(session: str,
           output_path: str = ".",
           backend: str = "sequential",
           dask_storage_path: str = os.path.join(os.path.expanduser("~"), ".aikit", "working_dir")):

    # Register in this dictionary all arguments that must be passed to the backend
    backend_kwargs = {
        "dask_storage_path": dask_storage_path,
    }
    backend_kwargs = filter_backend_kwargs(backend, **backend_kwargs)

    with get_backend(backend, session=session, **backend_kwargs) as backend:
        result_reader = AutoMlResultReader(backend.get_data_loader())

        df_results = result_reader.load_all_results()
        df_additional_results = result_reader.load_additional_results()
        df_params = result_reader.load_all_params()
        df_errors = result_reader.load_all_errors()
        df_params_other = result_reader.load_other_params()

        df_merged_result = pd.merge(df_params, df_results, how="inner", on="job_id")
        df_merged_result = pd.merge(df_merged_result, df_params_other, how="inner", on="job_id")
        if df_additional_results.shape[0] > 0:
            df_merged_result = pd.merge(df_merged_result, df_additional_results, how="inner", on="job_id")

        df_merged_error = pd.merge(df_params, df_errors, how="inner", on="job_id")

        result_filename = os.path.join(output_path, "result.xlsx")
        try:
            df_merged_result.to_excel(result_filename, index=False)
            _logger.info(f"Result file saved: {result_filename}")
        except:  # noqa
            _logger.warning(f"Error saving result file ({result_filename})", exc_info=True)

        error_filename = os.path.join(output_path, "error.xlsx")
        try:
            df_merged_error.to_excel(error_filename, index=False)
            _logger.info(f"Error file saved: {error_filename}")
        except:  # noqa
            _logger.warning(f"Error saving error file ({error_filename})", exc_info=True)


if __name__ == '__main__':
    app()
