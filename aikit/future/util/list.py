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
    # return reduce(lambda x,y:x+y,list_of_list,[])
