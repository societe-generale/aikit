import gc
import logging
import traceback
from collections import OrderedDict

from sklearn.metrics import SCORERS

from aikit.cross_validation import create_cv, score_from_params_clustering, cross_validation
from .serialization import DataLoader, Format
from ..enums import ProblemType
from ..util.serialization import sklearn_model_from_param

_logger = logging.getLogger(__name__)


def load_X_y(data_loader: DataLoader):  # noqa
    # TODO: add caching to avoid reloading data in each job
    X = data_loader.read(key="X", path="data", serialization_format=Format.PICKLE)  # noqa
    y = data_loader.read(key="y", path="data", serialization_format=Format.PICKLE)
    groups = data_loader.read(key="groups", path="data", serialization_format=Format.PICKLE)
    return X, y, groups


def load_automl_config(data_loader: DataLoader):
    automl_config = data_loader.read(key="automl_config", path="data", serialization_format=Format.PICKLE)
    job_config = data_loader.read(key="job_config", path="data", serialization_format=Format.PICKLE)
    return automl_config, job_config


def run_model_from_job_id(job_id: str, data_loader: DataLoader):
    _logger.info(f"Run job from queue: {job_id}")

    # Load data
    X, y, groups = load_X_y(data_loader)  # noqa

    # Load config
    automl_config, job_config = load_automl_config(data_loader)

    # Load job parameters
    job_param = data_loader.read(key=job_id, path="job_param", serialization_format=Format.JSON)

    # Create sklearn model from serialized form
    model = sklearn_model_from_param(job_param["model_json"])

    # Run model fit and cross-validation
    run_model(model,
              df=X,
              y=y,
              groups=groups,
              automl_config=automl_config,
              job_config=job_config,
              job_param=job_param,
              job_id=job_id,
              data_loader=data_loader)


def run_model(model, df, y, groups, automl_config, job_config, job_param, job_id, data_loader):
    """ Runs a given model """
    # Create cross-validation object related to target
    cv = create_cv(
            cv=job_config.cv,
            y=y,
            classifier=automl_config.problem_type == ProblemType.CLASSIFICATION,
            shuffle=True,
            random_state=123)

    # Create scorers
    scorers = OrderedDict()
    for i, s in enumerate(job_config.scoring):
        if isinstance(s, str):
            scorers[s] = SCORERS[s]
        else:
            scorers["scorer_%d" % i] = SCORERS[s]

    # Get params related to current job
    stopping_round = job_param.get("stopping_round", None)
    stopping_threshold = job_param.get("stopping_threshold", None)
    nodes_not_to_cross_validate = job_param.get("nodes_not_to_crossvalidate", None)
    cv_type = job_param.get("cv_type", "full")
    if cv_type not in ("approximate", "full"):
        raise NotImplementedError(f"Unknown CV type: {cv_type}, must be 'approximate' or 'full'")

    if automl_config.problem_type == ProblemType.CLASSIFICATION:
        method = "predict_proba"
    elif automl_config.problem_type == ProblemType.CLUSTERING:
        method = "fit_predict"
    else:
        method = "predict"

    # Run CV
    has_error = False
    error_traceback = error_as_string = None
    cv_result = y_pred = None
    try:
        # TODO: Avoid ifs per problem type
        if automl_config.problem_type == ProblemType.CLUSTERING:
            cv_result, y_pred = score_from_params_clustering(
                model, X=df, scoring=scorers, return_predict=True, method=method)
        else:
            approximate_cv = cv_type == "approximate"
            cv_result, y_pred = cross_validation(
                model,
                X=df,
                y=y,
                groups=groups,
                cv=cv,
                scoring=scorers,
                return_predict=True,
                method=method,
                n_jobs=1,  # Force n_jobs to 1 since we already parallelize at the worker level
                stopping_round=stopping_round,
                stopping_threshold=stopping_threshold,
                nodes_not_to_crossvalidate=nodes_not_to_cross_validate,
                approximate_cv=approximate_cv)
    except Exception as e:
        if isinstance(e, MemoryError):
            gc.collect()
        error_traceback = traceback.format_exc()
        error_as_string = repr(e)
        has_error = True

    # Return result
    if not has_error:
        # TODO: Avoid ifs per problem type
        if automl_config.problem_type != ProblemType.CLUSTERING:
            test_metric = 100 * cv_result[f"test_{job_config.main_scorer}"].mean()
            train_metric = 100 * cv_result[f"train_{job_config.main_scorer}"].mean()
            _logger.info(f"Train {job_config.main_scorer}: {train_metric:.2f}%")
            _logger.info(f"Test  {job_config.main_scorer}: {test_metric:.2f}%")

        data_loader.write(data=cv_result, key=job_id, path="result", serialization_format=Format.CSV)

        if job_config.additional_scoring_function is not None:
            additional_result = job_config.additional_scoring_function(cv_result, y_pred, y, groups)
            data_loader.write(data=additional_result,
                              key=job_id,
                              path="additional_result",
                              serialization_format=Format.JSON)

        if automl_config.problem_type == ProblemType.CLUSTERING:
            data_loader.write(data=y_pred, key=job_id, path="labels", serialization_format=Format.CSV)

        return has_error, (cv_result, y_pred)
    else:
        _logger.warning(f"Error on job_id {job_id}")
        _logger.warning(error_as_string)
        _logger.warning(error_traceback)
        data_loader.write(data=error_as_string, key=job_id, path="error", serialization_format=Format.TEXT)
        return has_error, error_as_string
