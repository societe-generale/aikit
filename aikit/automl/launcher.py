from aikit.automl import Controller, Worker


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

    def start_controller(self):
        controller = Controller(
            data_path=self.data_path,
            data_key=self.data_key,
            storage_path=self.storage_path,
            queue_path=self.queue_path
        )
        controller.run()

    def start_worker(self):
        worker = Worker(
            data_path=self.data_path,
            data_key=self.data_key,
            storage_path=self.storage_path,
            queue_path=self.queue_path
        )
        worker.run()
