import logging
import os

from aikit.automl import Controller, Worker
from aikit.automl.persistence.storage import Storage


class AutoMlLauncher:
    """
    TODO:
    - review serialize and deserialize automl config and job config to JSON
    - review AutoMlModelGuider
    """

    def __init__(self, storage_path, queue_path=None, data_path=None, data_key=None):
        self.storage_path = storage_path
        self.queue_path = queue_path or storage_path
        self.data_path = data_path or storage_path
        self.data_key = data_key or 'data'

    def start_controller(self, max_runtime_seconds=None, max_model_count=None):
        controller = Controller(
            data_path=self.data_path,
            data_key=self.data_key,
            storage_path=self.storage_path,
            queue_path=self.queue_path
        )
        controller.run(max_runtime_seconds=max_runtime_seconds,
                       max_model_count=max_model_count)
        return controller

    def start_worker(self):
        worker = Worker(
            data_path=self.data_path,
            data_key=self.data_key,
            storage_path=self.storage_path,
            queue_path=self.queue_path
        )
        worker.run()
        return worker

    def stop_workers(self):
        storage = Storage(self.storage_path)
        for filename in storage.listdir('workers'):
            if filename.endswith('.json'):
                worker_id = os.path.splitext(filename)[0]
                logging.getLogger("aikit").info(f"Stop worker {worker_id}...")
                jso = storage.load_json(worker_id, 'workers')
                jso['status'] = "stopped"
                storage.save_json(jso, worker_id, 'workers')
