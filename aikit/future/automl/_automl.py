import abc
import datetime
import logging

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from ._registry import MODEL_REGISTRY
from ._config import AutoMlConfig
from ._job import JobConfig
from .backends import AutoMlBackend
from .guider import AutoMlModelGuider
from .random_model_generator import RandomModelGenerator
from ..enums import SpecialModels
from ..graph import convert_graph_to_code
from ..util.hash import md5_hash

_logger = logging.getLogger(__name__)


class AutoMlBudget(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def start(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def has_budget(self):
        raise NotImplementedError()


class TimeBudget(AutoMlBudget):
    """ A time budget class with a budget in seconds. """

    def __init__(self, time_budget: int):
        self.time_budget = time_budget
        self.start_time = None

    def start(self):
        self.start_time = datetime.datetime.now()

    def has_budget(self):
        return (datetime.datetime.now() - self.start_time).total_seconds() < self.time_budget


class AutoMl:
    """
    Main class for AutoML.

    Generate models and pipelines for a given dataset.
    Automatic model generation is both random and guided by previous model scores.

    The AutoMl object uses a backend to send tasks, retrieve results and store data.
    """

    def __init__(self,
                 automl_config: AutoMlConfig,
                 job_config: JobConfig,
                 automl_guider: AutoMlModelGuider,
                 backend: AutoMlBackend,
                 budget: AutoMlBudget,
                 random_state=None):

        self.automl_config = automl_config
        self.job_config = job_config
        self.automl_guider = automl_guider
        self.backend = backend
        self.random_model_generator = None
        self.random_state = random_state
        self.budget = budget

        self._last_metric_threshold = None

        self.random_model_generator = RandomModelGenerator(automl_config=self.automl_config,
                                                           random_state=self.random_state)

        if self.job_config.start_with_default:
            self._default_iterator = self.random_model_generator.default_models_iterator()
            self._default_iterator_empty = False
        else:
            self._default_iterator = None  # Will never be used
            self._default_iterator_empty = True

        if self.job_config.do_blocks_search:
            self._block_search_iterator = self.random_model_generator.block_search_iterator(random_order=True)
            self._block_search_iterator_empty = False
        else:
            self._block_search_iterator = None
            self._block_search_iterator_empty = True

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, new_random_state):
        self._random_state = check_random_state(new_random_state)
        if self.random_model_generator is not None:
            self.random_model_generator.random_state = self._random_state

    def _get_next_job_type(self):
        """ Return the next job type to be generated and the next model in an iterator. """
        job_type = None
        iter_next = None
        # Generate a model from the default iterator
        if not self._default_iterator_empty:
            phase = self.backend.get_current_phase()
            # Check if we are in the default phase
            if phase == "default":
                try:
                    iter_next = next(self._default_iterator)
                except StopIteration:
                    _logger.debug("no more default model to generate...")
                    iter_next = None

                if iter_next is None:
                    _logger.info("default_iterator is empty, switch to 'random' phase")
                    job_type = None
                    self._default_iterator_empty = True
                    phase = "random"
                    self.backend.set_current_phase(phase)
                else:
                    job_type = "default"
            else:
                job_type = None
                iter_next = None

        # No job type: try to generate a model from the block search iterator
        if job_type is None and not self._block_search_iterator_empty:
            p_block_search = 0.2
            if self.random_state.rand(1)[0] <= p_block_search:
                try:
                    iter_next = next(self._block_search_iterator)
                except StopIteration:
                    _logger.debug("no more block_search model to generate...")
                    iter_next = None

                if iter_next is None:
                    _logger.info("block_search iterator is empty, will generate a model using guider")
                    self._block_search_iterator_empty = True
                    job_type = None
                else:
                    job_type = "block_search"

        # No job type: generate a model using the guider
        if job_type is None:
            if self.automl_guider is None:
                # no guider: switch to exploration task
                job_type = "exploration"
            else:
                p_exploration = self.automl_guider.find_exploration_proba()
                _logger.info(f"AutoML guider exploration proba: {p_exploration:0.4f}")
                if self.random_state.rand(1)[0] <= p_exploration:
                    job_type = "exploration"
                else:
                    job_type = "guided"

        return job_type, iter_next

    def _get_next_model(self, job_type, next_model):

        if job_type == "exploration":
            _logger.info("Create a random model")

            # Draw a random model
            graph, all_models_params, blocks_to_use = self.random_model_generator.draw_random_graph()

            model_dict = convert_graph_to_code(graph, all_models_params, return_mapping=True)
            model_json_code = model_dict["json_code"]
            name_mapping = model_dict["name_mapping"]

            job_id = md5_hash(model_json_code)

            json_param = {
                "Graph": {
                    "nodes": list(graph.nodes),
                    "edges": list(graph.edges)
                },
                "all_models_params": all_models_params,
                "blocks_to_use": blocks_to_use,
                "job_id": job_id,
            }

        elif job_type == "guided":
            _logger.info("Create a model using the AutoML guider")

            # 'Guided' model: draw several random models and use the guider to select the best one
            all_params1 = []
            all_params2 = []
            all_names_mapping = []

            # TODO: put 100 into some default AutoML configuration
            for nb in range(100):
                # Draw 100 random models
                graph, all_models_params, blocks_to_use = self.random_model_generator.draw_random_graph()

                model_dict = convert_graph_to_code(graph, all_models_params, return_mapping=True)
                model_json_code = model_dict["json_code"]
                name_mapping = model_dict["name_mapping"]

                job_id = md5_hash(model_json_code)

                # TODO: verify if job already exists ?
                json_param = {
                    "Graph": {
                        "nodes": list(graph.nodes),
                        "edges": list(graph.edges)
                    },
                    "all_models_params": all_models_params,
                    "blocks_to_use": blocks_to_use,
                    "job_id": job_id,
                }

                all_params1.append(json_param)
                all_params2.append(model_json_code)
                all_names_mapping.append(name_mapping)

            # Refit the guider model
            self.automl_guider.fit_metric_model()

            # Use guider to have a mean/variance prediction for the metric
            metric_prediction, metric_variance_prediction = self.automl_guider.predict_metric(all_params1)

            if metric_prediction is None or metric_variance_prediction is None:
                # Should not append
                # Just take the first random model
                _logger.warning("AutoML guider failed to predict metric, take the first random model")
                model_index = 0

            else:
                # Compute a model benchmark metric

                # 1) Mean + 2 * Std
                benchmark = metric_prediction + 2 * np.sqrt(metric_variance_prediction)

                # 2) Proba new >= best =>
                # (Mean - Best) / Std

                # 3) E( New * (1[new >= best]) )

                # If we use ranks, the best is almost the rank 1

                def softmax(_benchmark, T=1.0):  # noqa
                    ss = np.std(_benchmark)
                    if ss == 0:
                        return 1 / len(_benchmark) * np.ones(len(_benchmark), dtype=np.float32)
                    else:
                        n_benchmark = (_benchmark - np.mean(_benchmark)) / ss
                        exp_n_benchmark = np.exp(n_benchmark / T)
                        return exp_n_benchmark / exp_n_benchmark.sum()

                probas = softmax(benchmark, T=0.1)

                # Remark:
                # Takes the "best" with a 0.25 probability
                # TODO: temperature param T can also decay in time
                probas[pd.isnull(probas)] = 0.0
                probas[np.isinf(probas)] = 0.0

                if probas.sum() == 0:
                    model_index = np.random.choice(len(probas), size=1)[0]
                else:
                    probas = probas / probas.sum()
                    model_index = np.random.choice(len(probas), size=1, p=probas)[0]

                # TODO: we can also remove 'bad' models
                # TODO: we can also make a random choice among the best models

            model_json_code = all_params2[model_index]
            json_param = all_params1[model_index]
            name_mapping = all_names_mapping[model_index]

        elif job_type in ("default", "block_search"):

            if job_type == "default":
                _logger.info("Create default model...")
            else:
                _logger.info("Create create a block-search model...")

            graph, all_models_params, blocks_to_use = next_model

            model_dict = convert_graph_to_code(graph, all_models_params, return_mapping=True)
            model_json_code = model_dict["json_code"]
            name_mapping = model_dict["name_mapping"]

            job_id = md5_hash(model_json_code)

            json_param = {
                "Graph": {
                    "nodes": list(graph.nodes),
                    "edges": list(graph.edges)
                },
                "all_models_params": all_models_params,
                "blocks_to_use": blocks_to_use,
                "job_id": job_id,
            }

        else:
            raise NotImplementedError(f"Unknown job_type {job_type}, please check the _get_next_job_type() method")

        return json_param, model_json_code, name_mapping

    def _get_next_model_job(self):
        """ main method that will generate a new job to be given to ml machine """
        # nb_models_done = self.auto_ml_guider.get_nb_models_done()
        # proba_exploration = XXX

        # Discover next job type and next model
        job_type, next_model = self._get_next_job_type()

        # Get next model
        json_param, model_json_code, name_mapping = self._get_next_model(job_type, next_model)

        # Parameterize cross-validation
        job_param = {
            "model_json": model_json_code,
            "job_type": job_type,
            "cv_type": "full"  # default
        }
        if self.job_config.allow_approx_cv:
            # If the model is a GraphPipeline instance
            if model_json_code[0] == SpecialModels.GraphPipeline:
                # Loop through the nodes to figure out which one uses y, if not they won't be cross-validated
                nodes_not_to_crossvalidate = set()
                for node in json_param["Graph"]["nodes"]:
                    infos = MODEL_REGISTRY.informations.get(node[1], None)
                    if infos is not None:
                        if not infos.get("use_y", True):
                            nodes_not_to_crossvalidate.add(name_mapping[node])
                # Update job parameters
                job_param["cv_type"] = "approximate"
                job_param["nodes_not_to_crossvalidate"] = nodes_not_to_crossvalidate

        # Sets the cross-validation threshold
        if self.automl_guider is None:
            metric_threshold = None
        else:
            metric_threshold = self.automl_guider.find_metric_threshold()
            if self._last_metric_threshold is not None:
                # Take the max so that the metric threshold ALWAYS increases which makes more sense
                # Otherwise it could decrease because it is a quantile which can vary
                metric_threshold = max(metric_threshold, self._last_metric_threshold)
            self._last_metric_threshold = metric_threshold

        if self.job_config.baseline_score is not None:
            job_param["stopping_round"] = 0
            job_param["stopping_threshold"] = self.job_config.baseline_score

            if metric_threshold is not None:
                # Warning: will not work if the scorer is a loss
                job_param["stopping_threshold"] = max(metric_threshold, job_param["stopping_threshold"])
        else:
            if metric_threshold is not None:
                job_param["stopping_round"] = 0
                job_param["stopping_threshold"] = metric_threshold
            else:
                job_param["stopping_round"] = None
                job_param["stopping_threshold"] = None

        # Remark: we can disable CV by setting a starting stopping threshold with a high value
        _logger.info(f"stopping_threshold: {job_param['stopping_threshold']}")
        _logger.info(f"cv_type: {job_param['cv_type']}")

        job_param["job_creation_time"] = datetime.datetime.strftime(
            datetime.datetime.now(tz=datetime.timezone.utc), "%Y-%m-%d %H:%M:%S")

        return json_param["job_id"], job_param, json_param

    def search_models(self):
        _logger.info("Start searching models...")
        self.backend.start()
        self.budget.start()

        while self.budget.has_budget():
            job_id, job_param, json_param = self._get_next_model_job()
            _logger.info(f"Put next job in queue, job_id={job_id}")
            self.backend.send_job(job_id, job_param, json_param)
            self.backend.wait_for_job_queue()

        _logger.info("No more budget, stop searching models...")

        self.backend.stop()
