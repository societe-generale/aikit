import numpy as np


def is_number(x):
    """ small function to test if something is a python number """
    return issubclass(type(x), (float, int, np.number))
