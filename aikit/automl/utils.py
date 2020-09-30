import sys
import logging


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
