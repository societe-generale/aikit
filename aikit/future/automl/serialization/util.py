import os


def path_join(*paths, pathsep=os.path.pathsep):
    """ Join paths using the given path separator. """
    return pathsep.join(paths)
