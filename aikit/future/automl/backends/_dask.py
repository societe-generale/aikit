import logging
import time
import uuid
from functools import partial

from distributed import Client, LocalCluster

from ._base import AutoMlBackend, register_backend
from ..serialization import FsDataLoader, Format
from ..worker import run_model_from_job_id

_logger = logging.getLogger(__name__)


class DaskBackend(AutoMlBackend):

    def __init__(self, session: str, *, cluster: str = None, num_workers: int = 1, storage_path: str = None, max_jobs: int = 10):
        if cluster == "local":
            self.cluster = LocalCluster(n_workers=num_workers,
                                        threads_per_worker=1,
                                        processes=True)
        else:
            self.cluster = cluster

        self.data_loader = FsDataLoader(storage_path, session)
        self.dask_client = None
        self.max_jobs = max_jobs
        self.futures = {}
        self.phase = None
        self.job_done_callback = None

    def __enter__(self):
        self.dask_client = Client(self.cluster).__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dask_client.__exit__(exc_type, exc_val, exc_tb)

    def __del__(self):
        if self.dask_client is not None:
            self.dask_client.__del__()

    def get_data_loader(self):
        return self.data_loader

    def get_current_phase(self):
        if self.data_loader.exists("phase", path="infos", serialization_format=Format.JSON):
            self.phase = self.data_loader.read("phase", path="infos", serialization_format=Format.JSON)
        else:
            self.phase = "default"
            self.set_current_phase(self.phase)
        return self.phase

    def set_current_phase(self, phase: str):
        self.phase = phase
        self.data_loader.write(self.phase, "phase", path="infos", serialization_format=Format.JSON)

    def _future_callback(self, future, job_id):
        _logger.info(f"Job {job_id} finished with status {future.status}")
        self.futures.pop(job_id)
        if self.job_done_callback is not None:
            self.job_done_callback(job_id)

    def send_job(self, job_id: str, job_param: dict, json_param: dict):
        self.data_loader.write(data=json_param, key=job_id, path="param", serialization_format=Format.JSON)
        self.data_loader.write(data=job_param, key=job_id, path="job_param", serialization_format=Format.JSON)
        future = self.dask_client.submit(run_model_from_job_id, job_id, self.data_loader)
        future.add_done_callback(partial(self._future_callback, job_id=job_id))
        self.futures[job_id] = future

    def set_done_callback(self, callback):
        self.job_done_callback = callback

    def wait_for_job_queue(self):
        while len(self.futures) >= self.max_jobs:
            _logger.info("Waiting 5s for running jobs to complete...")
            time.sleep(5)


register_backend("dask", DaskBackend)
