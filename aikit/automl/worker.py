import datetime
import gc
import sys
import uuid
import time
import logging
import traceback
from collections import OrderedDict

from aikit import enums
from aikit.model_definition import sklearn_model_from_param
from aikit.cross_validation import create_cv, cross_validation, score_from_params_clustering
from aikit.scorer import SCORERS
from aikit.tools.helper_functions import system_and_caller_information

from aikit.automl.utils import unpack_data
from aikit.automl.persistence.storage import Storage
from aikit.automl.persistence.job_queue import JobsQueue


logger = logging.getLogger('aikit')


class Worker:

    def __init__(self, storage_path, queue_path, data_path, data_key):
        self.uuid = str(uuid.uuid4())
        self.infos = {}
        self.data = Storage(data_path)
        self.data_key = data_key
        self.storage = Storage(storage_path)
        self.queue = JobsQueue(queue_path)

        self.job_config = self.storage.load_pickle('job_config')
        self._init_worker()

    def _init_worker(self):
        self.infos = system_and_caller_information()
        self.infos["id"] = self.uuid
        self.infos["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.infos["status"] = "running"
        self.storage.save_json(self.infos, self.uuid, 'workers')

        self.scorers = OrderedDict()
        for i, s in enumerate(self.job_config.scoring):
            if isinstance(s, str):
                self.scorers[s] = SCORERS[s]
            else:
                self.scorers["scorer_%d" % i] = SCORERS[s]

    def is_running(self):
        self.infos = self.storage.load_json(self.uuid, 'workers')
        return self.infos["status"] != "stopped"

    def run(self):
        X, y, groups = unpack_data(self.data.load_pickle(self.data_key))
        self.cv = create_cv(self.job_config.cv, y,
                            classifier=self.job_config.is_classification(),
                            shuffle=True,
                            random_state=123)

        while self.is_running():
            job_id = self.queue.dequeue()
            if job_id is not None:
                self.compute_job(job_id, X, y, groups)
            else:
                logging.info('Waiting for new job...')
                time.sleep(5)
        logger.info('Finished fitting models, worker exiting.')

    def compute_job(self, job_id, X, y, groups=None):
        logger.info('Running job {}'.format(job_id))

        status = {
            'job_id': job_id,
            'worker_id': self.uuid,
            'start_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.storage.save_json(status, job_id, 'running')

        try:
            job = self.storage.load_special_json(job_id, 'jobs')
            self.validate_job(job)
            cv_result, yhat = self.compute_cv(job, X, y, groups)
            status['results'] = cv_result.to_dict(orient='records')

            if self.job_config.additional_scoring_function is not None:
                additional_result = self.job_config.additional_scoring_function(cv_result, yhat, y, groups)
                status['additional_results'] = additional_result.to_dict(orient='records')

            if not self.job_config.is_clustering():
                status['train_metric'] = cv_result["train_%s" % self.job_config.main_scorer].mean()
                status['test_metric'] = cv_result["test_%s" % self.job_config.main_scorer].mean()
                logger.info('Train {}: {:.2%}'.format(self.job_config.main_scorer, status['train_metric']))
                logger.info('Test {}: {:.2%}'.format(self.job_config.main_scorer, status['test_metric']))

            if self.job_config.is_clustering():
                status['labels'] = yhat

        except Exception as e:
            if isinstance(e, MemoryError):
                gc.collect()

            status['error'] = repr(e)
            status['traceback'] = traceback.format_exc()
            logger.error(status['error'])

        finally:
            self.storage.remove(job_id + '.json', 'running')
            status['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.storage.save_json(status, job_id, 'finished')

    def validate_job(self, job):
        if job['cv_type'] not in enums.TypeOfCv.alls:
            raise ValueError('CV type {} is not allowed'.format(job['cv_type']))

    def compute_cv(self, job, X, y, groups):
        model = sklearn_model_from_param(job['model_code'])
        method = self.job_config.get_predict_method_name()

        if self.job_config.is_clustering():
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
                nodes_not_to_crossvalidate=job['nodes_not_to_crossvalidate'],
                approximate_cv=(job['cv_type'] == 'approximate'),
                verbose=False
            )

        return cv_result, yhat
