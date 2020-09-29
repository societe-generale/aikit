import os
import gc
import uuid
import time
import logging
import traceback
from collections import OrderedDict

from sklearn.utils.validation import check_random_state

from aikit import enums
from aikit.model_definition import sklearn_model_from_param
from aikit.cross_validation import create_cv, cross_validation, score_from_params_clustering
from aikit.scorer import SCORERS
from aikit.tools.helper_functions import system_and_caller_information, md5_hash

from aikit.automl.storage import Storage
from aikit.automl.jobs import JobsQueue


logger = logging.getLogger('aikit')


class Worker:

    def __init__(self, storage_path, queue_path, data_path, data_key, seed=None):
        self.uuid = str(uuid.uuid4())
        self.data = Storage(data_path)
        self.data_key = data_key
        self.storage = Storage(storage_path)
        self.queue = JobsQueue(queue_path)
        self.seed = seed

        self.random_state = None
        self.automl_config = self.storage.load_pickle('automl_config')
        self.job_config = self.storage.load_pickle('job_config')
        self._init_worker()

    def _init_worker(self):
        infos = system_and_caller_information()
        infos["id"] = self.uuid
        infos["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        infos["seed"] = self.seed or int(md5_hash(infos), 16) % (2 ** 32 - 1)

        self.infos = infos
        self.storage.save_json(infos, self.uuid, 'workers')
        self.random_state = check_random_state(infos["seed"])
        self.scorers = OrderedDict()
        for i, s in enumerate(self.job_config.scoring):
            if isinstance(s, str):
                self.scorers[s] = SCORERS[s]
            else:
                self.scorers["scorer_%d" % i] = SCORERS[s]

    def run(self):
        X, y, *_ = self.data.load_pickle(self.data_key)
        self.cv = create_cv(self.job_config.cv, y, classifier=self.automl_config.is_classification(),
                            shuffle=True, random_state=123)

        while True:
            job_id = self.queue.dequeue()
            if job_id is not None:
                self.compute_job(job_id, X, y)
            else:
                logging.info('Waiting for new job...')
                time.sleep(10)

    def compute_job(self, job_id, X, y, groups=None):
        logger.info('Running job {}'.format(job_id))
        infos = {
            'job_id': job_id,
            'worker_id': self.uuid,
            'start_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.storage.save_json(infos, job_id, 'running')

        try:
            job = self.storage.load_special_json(job_id, 'jobs')
            self.validate_job(job)
            cv_result, yhat = self.compute_cv(job, X, y, groups)
            self.save_results(job_id, cv_result, yhat, infos)

            if self.job_config.additional_scoring_function is not None:
                additional_result = self.job_config.additional_scoring_function(cv_result, yhat, y, groups)
                self.storage.save_json(additional_result, job_id, 'additional_results')

        except Exception as e:
            if isinstance(e, MemoryError):
                gc.collect()

            error = {
                'job_id': job_id,
                'error': repr(e),
                'traceback': traceback.format_exc()
            }
            infos['error'] = error['error']
            logger.error(error['error'])
            self.storage.save_json(error, job_id, 'errors')

        finally:
            self.storage.remove(job_id, 'json', 'running')
            infos['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.storage.save_json(infos, job_id, 'finished')

    def validate_job(self, job):
        if job['cv_type'] not in ["approximate", "full"]:
            raise ValueError('CV type {} is not allowed'.format(job['cv_type']))

    def compute_cv(self, job, X, y, groups):
        model = sklearn_model_from_param(job['model_code'])
        method = self.automl_config.get_predict_method_name()

        if self.automl_config.is_clustering():
            cv_result, yhat = score_from_params_clustering(
                model, X, scoring=self.scorers, return_predict=True, method=method, verbose=False
            )
        else:
            cv_result, yhat = cross_validation(
                model, X, y,
                groups=groups,
                cv=self.cv,
                scoring=self.scorers,
                return_predict=True,
                method=method,
                stopping_round=job['stopping_round'],
                stopping_threshold=job['stopping_threshold'],
                nodes_not_to_crossvalidate=job.get('nodes_not_to_crossvalidate', None),
                approximate_cv=(job['cv_type'] == 'approximate'),
                verbose=False
            )

        cv_result['job_id'] = job['job_id']
        return cv_result, yhat

    def save_results(self, job_id, cv_result, yhat, infos):
        infos['results'] = cv_result.to_dict(orient='records')
        self.storage.save_csv(cv_result, job_id, 'results')

        if not self.automl_config.is_clustering():
            train_metric = cv_result["train_%s" % self.job_config.main_scorer].mean()
            test_metric = cv_result["test_%s" % self.job_config.main_scorer].mean()
            infos['train_metric'] = train_metric
            infos['test_metric'] = test_metric
            logger.info('Train {}: {:.2%}'.format(self.job_config.main_scorer, train_metric))
            logger.info('Test {}: {:.2%}'.format(self.job_config.main_scorer, test_metric))

        if self.automl_config.is_clustering():
            self.storage.save_json(yhat, job_id, 'labels')
