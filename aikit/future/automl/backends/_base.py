import abc
import inspect

_BACKEND_REGISTRY = {}
_BACKEND_REGISTRY_KWARGS = {}


def filter_backend_kwargs(name: str, **kwargs):
    name = name.lower()
    if name not in _BACKEND_REGISTRY_KWARGS:
        raise ValueError(f"Unknown backend: {name}")
    kwarg_names = _BACKEND_REGISTRY_KWARGS[name]
    return {k.replace(f"{name}_", ""): v for k, v in kwargs.items() if k.replace(f"{name}_", "") in kwarg_names}


def register_backend(name: str, backend_class):
    _BACKEND_REGISTRY[name] = backend_class
    kwarg_names = []
    for param in inspect.signature(backend_class.__init__).parameters.values():
        if param.name == "self":
            continue
        if param.kind == param.KEYWORD_ONLY:
            kwarg_names.append(param.name)
    _BACKEND_REGISTRY_KWARGS[name] = kwarg_names


def get_backend(name: str, session: str, **kwargs):
    name = name.lower()
    if name not in _BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend: {name}")
    backend_class = _BACKEND_REGISTRY[name]
    return backend_class(session=session, **kwargs)


class AutoMlBackend(metaclass=abc.ABCMeta):
    """
    Base class for AutoML backends.

    An AutoML backend is responsible for processing jobs in parallel.
    It should use the `run_model_from_job_id` method to process jobs.
    """

    @abc.abstractmethod
    def get_data_loader(self):
        """ Returns the data loader used by the backend. """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_current_phase(self):
        """ Returns the current AutoML phase (e.g. 'default', ) """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_current_phase(self, phase: str):
        """ Updates the current phase. """
        raise NotImplementedError()

    @abc.abstractmethod
    def send_job(self, job_id: str, job_param: dict, json_param: dict):
        """ Sends a job for processing. """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_done_callback(self, callback):
        """
        Sets a callback to be called when a job is done.
        The callback is a function that takes a job_id as argument.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def wait_for_job_queue(self):
        """ Wait until there is a place in the queue for the next job. """
        raise NotImplementedError()
