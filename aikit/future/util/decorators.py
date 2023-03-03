from functools import wraps


def enforce_init(cls):
    """
    decorator that prevent setting attributes that are not set
    in the __init__ function.
    """

    def __setattr__(self, key, value):
        if key[0] != "_":
            if self.__frozen:
                if key not in self.__allowed_attributes:
                    raise TypeError(f"This attribute cannot be set: {key}")
            else:
                self.__allowed_attributes.add(key)

        object.__setattr__(self, key, value)

    def init_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self.__frozen = False
            self.__allowed_attributes = set()
            func(self, *args, **kwargs)
            self.__frozen = True

        return wrapper

    cls.__setattr__ = __setattr__
    cls.__init__ = init_decorator(cls.__init__)

    return cls


def singleton(cls):
    """ Singleton decorator """
    instances = {}

    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return getinstance
