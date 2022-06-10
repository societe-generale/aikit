import sys
import logging


def unpack_data(data):
    X, *params = data
    if len(params) == 0:
        return X, None, None
    elif len(params) == 1:
        return X, params[0], None
    elif len(params) == 2:
        return X, params[0], params[1]
    else:
        raise ValueError('Data only expect X, y and groups')


def deactivate_warnings():
    import warnings
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

    logging.getLogger('gensim.models.word2vec').setLevel(logging.ERROR)
    logging.getLogger('gensim.models.base_any2vec').setLevel(logging.ERROR)
    logging.getLogger('numexpr.utils').setLevel(logging.ERROR)


def configure_console_logging(name=None):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        fmt='%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
