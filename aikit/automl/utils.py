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


def unnest_job_parameters(job):
    """ unnest a param dictionnary,
    skipping 'ColumnsSelector' and 'columns_to_use'
    """
    res = {
        'job_id': job["job_id"]
    }
    for k, v in job["all_models_params"].items():
        if k[1][1] != "ColumnsSelector":
            for k2, v2 in v.items():
                if k2 != "columns_to_use":
                    res["__".join(k[1]) + "__" + k2] = v2

            res[k[1][0]] = k[1][1]

    for b in job["blocks_to_use"]:
        res["hasblock_%s" % b] = 1

    return res
