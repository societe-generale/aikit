import time
import logging
import datetime
import numpy as np
import pandas as pd

from sklearn.utils.validation import check_random_state

from aikit import enums
from aikit.tools.helper_functions import md5_hash
from aikit.automl.config import AutoMlConfig, JobConfig
from aikit.automl.persistence.storage import Storage
from aikit.automl.persistence.job_queue import JobsQueue
from aikit.automl.persistence.reader import ResultsReader
from aikit.automl.model_generation.guider import AutoMlModelGuider

from aikit.automl.model_generation.generator import RandomModelGenerator
from aikit.automl.model_generation.model_graph import convert_graph_to_code
from aikit.automl.model_registration import MODEL_REGISTER


logger = logging.getLogger('aikit')


class Controller:

    def __init__(self, data_path, data_key, storage_path, queue_path, random_state=None):
        self.data = Storage(data_path)
        self.data_key = data_key
        self.storage = Storage(storage_path)
        self.reader = ResultsReader(self.storage)
        self.queue = JobsQueue(queue_path)
        self.random_state = check_random_state(random_state)

        self.automl_config: AutoMlConfig = None
        self.job_config: JobConfig = None
        self.automl_guider: AutoMlModelGuider = None
        self.generator: RandomModelGenerator = None
        self._default_iterator = None
        self._default_iterator_empty = True
        self._block_search_iterator = None
        self._block_search_iterator_empty = True

        self.queue_size = 20
        self.jobs_update_time = 5
        self.p_block_search = 0.2
        self.status = {
            'mode': 'default',
            'metric_threshold': None
        }

        self._init_config()
        self._init_generators()

    def _init_config(self):
        if self.storage.exists('status.json'):
            self.status = self.storage.load_json('status')
            self.automl_config = self.storage.load_pickle('automl_config')
            self.job_config = self.storage.load_pickle('job_config')
            self.reader.load()
        else:
            X, y, *_ = self.data.load_pickle(self.data_key)
            self.automl_config = AutoMlConfig()
            self.automl_config.guess_everything(X, y)
            self.job_config = JobConfig(self.automl_config.type_of_problem)
            self.job_config.guess_cv(n_splits=5)
            self.job_config.guess_scoring()

            self.storage.save_pickle(self.automl_config, 'automl_config')
            self.storage.save_pickle(self.job_config, 'job_config')
            self.storage.save_json(self.status, 'status')

    def _init_generators(self):
        self.automl_guider = AutoMlModelGuider(self.job_config)
        self.generator = RandomModelGenerator(self.automl_config, random_state=self.random_state)

        if self.job_config.start_with_default:
            self._default_iterator = self.generator.iterator_default_models()
            self._default_iterator_empty = False

        if self.job_config.do_blocks_search:
            self._block_search_iterator = self.generator.iterate_block_search(random_order=True)
            self._block_search_iterator_empty = False

    def run(self):
        while self.queue.size() < self.queue_size:
            self.create_job()

        logger.info('Queue is full. Adding new jobs every {} seconds'.format(self.jobs_update_time))
        while True:
            current_queue_size = self.queue.size()
            for _ in range(self.queue_size - current_queue_size):
                self.create_job()
            time.sleep(self.jobs_update_time)

    def create_job(self):
        job = self.get_next_job()

        if job['job_id'] in self.reader.jobs:
            return

        job = self.compute_job_cv_type(job)
        job = self.compute_job_metric_threshold(job)

        self.storage.save_json(self.status, 'status')
        self.storage.save_special_json(job, job['job_id'], 'jobs')
        self.queue.enqueue(job['job_id'])
        logger.info('Generating {} job [{}]'.format(job['job_type'], job['job_id']))

    def get_next_job(self):
        # iterate on default iterator
        if self.status['mode'] == 'default' and not self._default_iterator_empty:
            try:
                next_job = next(self._default_iterator)
                return self.get_job('default', *next_job)
            except StopIteration:
                logger.info("Default iterator cleared out")
                self._default_iterator_empty = True
                self.status['mode'] = 'random'

        # iterate on block search iterator
        if not self._block_search_iterator_empty and self.random_state.rand(1)[0] <= self.p_block_search:
            try:
                next_job = next(self._block_search_iterator)
                return self.get_job('block-search', *next_job)
            except StopIteration:
                logger.info("Block-search iterator cleared out")
                self._block_search_iterator_empty = True

        p_exploration = self.find_exploration_proba()
        logger.debug('Exploration probability: {:.2%}'.format(p_exploration))

        if self.random_state.rand(1)[0] <= p_exploration:
            return self.get_job('exploration', *self.generator.draw_random_graph())
        else:
            return self.get_guided_job()

    def get_job(self, job_type, graph, all_models_params, blocks_to_use):
        code = convert_graph_to_code(graph, all_models_params, also_returns_mapping=True)
        job_id = md5_hash(code['json_code'])
        return {
            'job_id': job_id,
            'job_type': job_type,
            'job_creation_time': datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S"),
            'graph': {'nodes': list(graph.nodes), 'edges': list(graph.edges)},
            'all_models_params': all_models_params,
            'blocks_to_use': blocks_to_use,
            'model_code': code['json_code'],
            'name_mapping': code['name_mapping']
        }

    def find_exploration_proba(self):
        """ function to retrieve the probability to run a random model, proba decrease with number of model tested """
        n_jobs = self.reader.get_number_of_successful_jobs()
        if n_jobs <= self.automl_guider.min_nb_of_models:
            return 1
        elif n_jobs <= 100:
            return 0.5
        elif n_jobs <= 500:
            return 0.25
        else:
            return 0.1

    def get_guided_job(self):
        """ Guided job : draw several models and guess which is best

        1) Mean + 2 * Std
        benchmark = metric_prediction + 2 * np.sqrt(metric_variance_prediction)
        2) Proba new >= best =>
        (Mean - Best) / Std
        3) E(New * (1[new >= best])
        Rmk : si on utilise des rang, best presque 1
        ii = np.argmax(benchmark)

        NOTE: on va prendre a peu pres le plus best avec une proba 1/4 (suivant les tests)
        # On peut aussi faire une heuristic en suppossant benchmark uniform (pas tres loin de la verite vu qu'on a fitter un rank...)

        # Comme ca je prend pas l'argmax, mais quelque chose d'un peu plus exploratoir
        # ... peut etre que argmax ca marcherait mieux (surement plus petite variation autour du meilleurs model)

        # On peut aussi virer les trucs vraiment pas bon...
        # Ou tirer au hasard parmis les meilleurs ..

        TODO : on peut faire descendre la temperature en court de route.... a peu pres equivalent à gérer l'explortion...
        """
        jobs = [self.get_job('guided', *self.generator.draw_random_graph()) for _ in range(100)]

        # refit guider
        self.automl_guider.fit_metric_model(self.reader)

        # Predict score + var using the guider
        metric_prediction, metric_variance_prediction = self.automl_guider.predict_metric(jobs)

        if metric_prediction is None or metric_variance_prediction is None:
            return jobs[0]

        def softmax(benchmark, T=1):
            ss = np.std(benchmark)
            if ss == 0:
                return 1 / len(benchmark) * np.ones(len(benchmark), dtype=np.float32)
            else:
                nbenchmark = (benchmark - np.mean(benchmark)) / ss
                exp_nbenchmark = np.exp(nbenchmark / T)
                return exp_nbenchmark / exp_nbenchmark.sum()

        benchmark = metric_prediction + 2 * np.sqrt(metric_variance_prediction)
        probas = softmax(benchmark, T=0.1)
        probas[pd.isnull(probas)] = 0.0
        probas[np.isinf(probas)] = 0.0

        if probas.sum() == 0:
            job_index = np.random.choice(len(probas), size=1)[0]
        else:
            probas = probas / probas.sum()
            job_index = np.random.choice(len(probas), size=1, p=probas)[0]

        return jobs[job_index]

    def compute_job_cv_type(self, job):
        if self.job_config.allow_approx_cv and job['model_code'][0] == enums.SpecialModels.GraphPipeline:
            # Loop throught the nodes to figure out which one don't use y, and won't be cross-validated
            nodes_not_to_crossvalidate = set()
            for node in job["graph"]["nodes"]:
                infos = MODEL_REGISTER.informations.get(node[1], None)
                if infos is not None:
                    if not infos.get("use_y", True):
                        nodes_not_to_crossvalidate.add(job['name_mapping'][node])

            job["cv_type"] = "approximate"
            job["nodes_not_to_crossvalidate"] = nodes_not_to_crossvalidate
        else:
            job['cv_type'] = 'full'
            job['nodes_not_to_crossvalidate'] = None

        return job

    def compute_job_metric_threshold(self, job):
        # TODO: si le scorer est une loss ca ne fonctionne pas de mettre max(threshold) !
        # Rmk : on peut aussi ne pas faire de CV au depart, en mettant ici un stopping_threshold trop haut quoiqu'il arrive

        metric_threshold = self.find_metric_threshold(min_nb_of_models=10)
        if self.status['metric_threshold'] is not None:
            metric_threshold = max(metric_threshold, self.status['metric_threshold'])
            # I do that so that, the metric threshold ALWAYS increases which make more sense.
            # Otherwise it could decrease because it is a quantile which can vary
        self.status['metric_threshold'] = metric_threshold

        if self.job_config.score_base_line is not None:
            job["stopping_round"] = 0
            job["stopping_threshold"] = self.job_config.score_base_line

            if metric_threshold is not None:
                job["stopping_threshold"] = max(metric_threshold, job["stopping_threshold"])

        else:
            if metric_threshold is not None:
                job["stopping_round"] = 0
                job["stopping_threshold"] = metric_threshold
            else:
                job["stopping_round"] = None
                job["stopping_threshold"] = None

        return job

    def find_metric_threshold(self, min_nb_of_models):
        """ find a threshold on the metric to use to stop crossvalidation
        TODO:
        - ici peut etre faire une estimation parametric du quantile avec un Kernel, plus smooth et moins sensible quand peu de valeurs
            result = np.percentile( min_cv, self._get_quantile(len(min_cv)) * 100)
        - pourquoi utilise t'on le même min_nb_of_models ici ?
        """
        logger.debug("Compute metric threshold")

        results = self.reader.get_results()
        if results['job_id'].nunique() <= min_nb_of_models:
            logger.debug("threshold = None")
            return None

        main_scorer = "test_%s" % self.job_config.main_scorer
        results[main_scorer] = results[main_scorer].fillna(results[main_scorer].min())
        min_cv = results.groupby("job_id")[main_scorer].min().values
        delta_min_max_cv = np.median(results.groupby("job_id")[main_scorer].apply(lambda x: x.max() - x.min()))

        min_cv = -np.sort(-min_cv)
        result = min_cv[min_nb_of_models] - delta_min_max_cv
        logger.debug("threshold : %2.2f" % result)
        return result
