

def lunique(original_list):
    """ Keeps unique items from original list """
    result = []
    for o in original_list:
        if o not in result:
            result.append(o)
    return result


def unlist(list_of_list):
    """ transform a list of list into one list with all the elements

    Examples
    --------
    >>> unlist([[1, 10], [32]])
    [1, 10, 32]
    >>> unlist([[10], [11], [], [45]])
    [10, 11, 45]
    """

    res = []
    for l in list_of_list:
        res += l
    return res


def diff(list1, list2):
    """ difference list1 minus list2 """
    res = [ll for ll in list1 if ll not in list2]  # copied from useful functions
    if isinstance(list1, tuple):
        res = tuple(res)
    return res


def intersect(list1, list2):
    """ intersection of 2 lists """
    res = [ll for ll in list1 if ll in list2]
    if isinstance(list1, tuple):
        res = tuple(res)
    return res


def unnest_tuple(nested_tuple):
    """ helper function to un-nest a nested tuple """
    def _rec(_nested_tuple):
        if isinstance(_nested_tuple, (tuple, list)):
            res = []
            for o in _nested_tuple:
                res += _rec(o)
            return res
        else:
            return [_nested_tuple]
    return tuple(_rec(nested_tuple))


def tuple_include(t1, t2):
    return all([t in t2 for t in t1])
