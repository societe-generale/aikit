import os


def path_join(*paths, pathsep=os.path.sep):
    """ Join paths using the given path separator. """
    return pathsep.join(paths)


def path_split(path: str, pathsep=os.path.sep):
    """ Split a path using the given path separator. """
    last_sep_index = path.rindex(pathsep)
    return path[:last_sep_index], path[last_sep_index + 1:]
