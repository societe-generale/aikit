import logging
import queue
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from ._base import AutoMlBackend, register_backend
from ..serialization import MemoryDataLoader, Format
from ..worker import run_model_from_job_id

_logger = logging.getLogger(__name__)


class SequentialBackend(AutoMlBackend):
    """ A simple sequential backend that processes jobs one at a time. """

    def __init__(self, session: str):
        self.data_loader = MemoryDataLoader()
        self.phase = "default"
        self.queue = queue.Queue()
        self.executor = None
        self.job_done_callback = None

    def __enter__(self):
        self.executor = ThreadPoolExecutor(max_workers=1).__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.queue = queue.Queue()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.executor.__exit__(exc_type, exc_val, exc_tb)

    def get_data_loader(self):
        return self.data_loader

    def get_current_phase(self):
        return self.phase

    def set_current_phase(self, phase: str):
        self.phase = phase

    def send_job(self, job_id: str, job_param: dict, json_param: dict):
        self.data_loader.write(data=json_param, key=job_id, path="param", serialization_format=Format.JSON)
        self.data_loader.write(data=job_param, key=job_id, path="job_param", serialization_format=Format.JSON)
        self.queue.put(job_id)
        self.executor.submit(self._process_queue)

    def set_done_callback(self, callback):
        self.job_done_callback = callback

    def wait_for_job_queue(self):
        while not self.queue.empty():
            _logger.info("Waiting 5s for job queue to be empty...")
            time.sleep(5)

    def _process_queue(self):
        try:
            job_id = self.queue.get(block=True, timeout=5)
        except queue.Empty:
            _logger.info("No job in queue, skip")
            return

        try:
            run_model_from_job_id(job_id, self.data_loader)
        except:  # noqa
            _logger.warning("Error while processing a job", exc_info=sys.exc_info())
        finally:
            if self.job_done_callback is not None:
                self.job_done_callback(job_id)


register_backend("sequential", SequentialBackend)
