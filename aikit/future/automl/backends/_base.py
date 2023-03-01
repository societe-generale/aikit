import abc


BACKEND_REGISTRY = {}


class AutoMlBackend(metaclass=abc.ABCMeta):
    """
    Base class for AutoML backends.

    An AutoML backend is responsible for processing jobs in parallel.
    It can rely on the `run_model_from_job_id` method to process jobs.
    """
    # TODO: backends must implement __enter__ and __exit__ methods

    @abc.abstractmethod
    def start(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def stop(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_data_loader(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_current_phase(self):
        """ Returns the current AutoML phase (e.g. 'default', ) """
        raise NotImplementedError()
        """
                    if self.data_persister.exists("phase", path="infos", write_type=SavingType.json):
                phase = self.data_persister.read("phase", path="infos", write_type=SavingType.json)
            else:
                phase = "default"
                self.data_persister.write(phase, "phase", path="infos", write_type=SavingType.json)
        """

    @abc.abstractmethod
    def set_current_phase(self, phase: str):
        """ Updates the current phase. """
        raise NotImplementedError()
    """
                        self.data_persister.write(phase, "phase", path="infos", write_type=SavingType.json)
    """

    @abc.abstractmethod
    def send_job(self, job_id: str, job_param: dict, json_param: dict):
        """ Sends a job for processing. """
        raise NotImplementedError()

    @abc.abstractmethod
    def wait_for_job_queue(self):
        """ Wait until there is a place in the queue for the next job. """
        raise NotImplementedError()
