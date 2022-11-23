# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:34:01 2018

@author: Lionel Massoulard
"""
import abc

import numpy as np

import itertools
import scipy.stats

from sklearn.utils.validation import check_random_state


def _try_set_random_state(dist, random_state):
    if hasattr(dist, "random_state"):
        dist.random_state = random_state
    return dist


def _get_rand(hyper, random_state=None):
    """ Draw a random sample from a hyperparameter, a constant or a tuple.
    This allows implicit use of list/tuple to model a choice and object to model a constant.

    Parameters
    ----------
    hyper : list, tuple, hyperparameter class or constant
    """
    if random_state is not None:
        _try_set_random_state(hyper, random_state)  # Will modify the object

    if hasattr(hyper, "get_rand"):
        return hyper.get_rand()

    elif isinstance(hyper, (list, tuple)):
        gen = check_random_state(random_state)  # Here I won't have the seed set...
        # Warning: never use np.random.choice(hyper) as it puts everything
        # into a numpy array and makes a wrong type cast
        return hyper[gen.choice(len(hyper))]

    elif hasattr(hyper, "rvs"):
        return hyper.rvs(random_state=random_state)
    else:
        return hyper  # constant


def _get_size(hyper):
    """ return the number of choices from 'something',
    something can be either a hyperparameter but also a constant or a tuple
    """
    if hasattr(hyper, "size"):
        return hyper.size

    elif isinstance(hyper, (list, tuple)):
        return len(hyper)

    elif hasattr(hyper, "rvs"):
        return 10  # heuristic

    else:
        return 1


class _AbstractHyper(metaclass=abc.ABCMeta):
    """ Abstract class representing a hyperparameter (or a set of hyperparameter).

    When inheriting this class, subclasses may implement:
      * _size attribute if the size is known, otherwise _get_size function will be used
      * value attribute is mandatory since samples will be generated from this list
    """

    def __init__(self, random_state=None):
        self.random_state = random_state

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, new_random_state):
        self._random_state = check_random_state(new_random_state)
        self._set_random_state()

    def _set_random_state(self):
        # Do nothing in that class
        return self

    def get_rand(self):
        """
        Generates one random sample based on values.

        Returns
        -------
        one sample
        """
        index = self._random_state.choice(len(self.values))
        # I'd rather not use directly np.random.choice because it is making conversion to np.int/np.float
        # Example: type(np.random.choice([0,1])) == np.int32
        return self.values[index]

    def get_rands(self, n):
        """ generates n samples """
        return [self.get_rand() for _ in range(n)]

    def get_size(self):
        """ return the number of choices, or a proxy if parameter is continuous """
        if hasattr(self, "values"):
            return len(self.values)
        else:
            # Rmk : in case of non-uniform choice, entropy can be used to derive an equivalent number of choice
            raise NotImplementedError("Please implement that in classes")

    @property
    def size(self):
        if not hasattr(self, "_size"):
            setattr(self, "_size", self.get_size())
        return self._size  # noqa

    def distplot(self):
        """ plot the distribution """
        import seaborn as sns
        sns.distplot([self.get_rand() for _ in range(10000)])


class HyperChoice(_AbstractHyper):
    """ Random choice among a set of values.

    Examples
    --------
    >>> hp = HyperChoice(('a','b','c'))
    >>> hp.get_rand()

    Remark: use 'HyperComposition' to make weighting choices
    """

    def __init__(self, values, random_state=None):
        super().__init__(random_state)
        self.values = values


class HyperMultipleChoice(_AbstractHyper):
    """ Select one or many items among a set of values.

    Examples
    --------
    >>> hp = HyperMultipleChoice(("a","b","c","d","e"))
    >>> hp.get_rand()
    """

    def __init__(self, possible_choices, min_number=1, max_number=None, proba_choice=0.9, random_state=None):
        super().__init__(random_state)
        self.possible_choices = possible_choices
        self.proba_choice = proba_choice
        self.min_number = min_number
        self.max_number = max_number

        if not isinstance(self.proba_choice, (list, tuple)):
            self.proba_choice = [self.proba_choice] * len(self.possible_choices)

        if self.min_number is None:
            self.min_number = 0

        if self.min_number < 0:
            raise ValueError(f"min_number ({self.min_number}) should be >= 0")

        if self.min_number > len(self.possible_choices):
            raise ValueError(f"min_number ({self.min_number}) should be <= len of choice ({self.possible_choices})")

        if self.max_number is not None and self.max_number < self.min_number:
            raise ValueError(f"max_number ({self.max_number}) should be > than min_number ({self.min_number})")

        if self.max_number is not None and self.max_number > len(self.possible_choices):
            raise ValueError(f"max_number ({self.max_number}) should be <= len of choices ({self.possible_choices})")

        self._precomputed_choices = None
        self._precomputed_probas = None
        self._use_precomputed = len(self.possible_choices) <= 10

    def _precompute(self):
        if self._precomputed_choices is not None:
            return

        all_choices = []
        for choice in itertools.product(*[[0, 1] for _ in range(len(self.possible_choices))]):
            once_choice = np.array(choice)

            nb = once_choice.sum()
            if nb < self.min_number:
                continue

            if self.max_number is not None and nb > self.max_number:
                continue

            probas = [(1 - p) + (2 * p - 1) * c for p, c in zip(self.proba_choice, choice)]
            probas = np.product(probas)

            all_choices.append((choice, probas))

        choices, probas = zip(*all_choices)
        probas = np.array(probas)
        probas /= probas.sum()

        self._precomputed_choices = choices
        self._precomputed_probas = probas

    def get_size(self):
        if self._use_precomputed:
            self._precompute()
            return len(self._precomputed_choices)
        else:
            return 2 ** (len(self.possible_choices))  # upper bound: because there are impossible cases

    def get_rand(self):
        if self._use_precomputed:
            self._precompute()
            return self._get_rand_precomputed()
        else:
            return self._get_rand()

    def _get_rand_precomputed(self):
        """ generate using precomputed values """
        choice = self._precomputed_choices[
            self.random_state.choice(len(self._precomputed_choices), p=self._precomputed_probas)
        ]
        return tuple([self.possible_choices[i] for i, c in enumerate(choice) if c == 1])

    def _get_rand(self):
        """ generic using reject """
        # There is probably a more efficient way to draw from that distributions
        MAX_ITER = 1000  # noqa
        iter_ = 0
        should_continue = True
        while should_continue:
            if iter_ >= MAX_ITER + 1:
                break

            iter_ += 1
            should_continue = False

            to_take = self.random_state.uniform(0, 1, len(self.possible_choices)) <= self.proba_choice
            ii = np.arange(len(self.possible_choices))
            nb = ii.sum()

            if self.min_number is not None and nb < self.min_number:
                continue
            if self.max_number is not None and nb > self.max_number:
                continue

            return tuple([self.possible_choices[i] for i in ii[to_take]])

        # to_take = np.random.choice(len(self.possible_choices), replace=False, size=self.min_number)
        to_take = self.random_state.choice(len(self.possible_choices), replace=False, size=self.min_number)
        return tuple([self.possible_choices[i] for i in to_take])


class HyperRangeInt(_AbstractHyper):
    """ Integers between start and end """

    def __init__(self, start, end, step=1, random_state=None):
        super().__init__(random_state=random_state)
        if end <= start:
            raise ValueError("end can't be lower than start")
        self.values = [x for x in range(start, end + step, step)]  # end included


class HyperRangeFloat(_AbstractHyper):
    """ Float between start and end """

    def __init__(self, start, end, n=100, step=None, random_state=None):
        super().__init__(random_state=random_state)

        if end <= start:
            raise ValueError(f"end ({end}) can't be lower than start ({start})")

        if step is not None:
            n = int(np.floor((end - start) / step) + 1)
            self.values = [start + i * step for i in range(n)]
        else:
            self.values = [start + (end - start) * i / n for i in range(n + 1)]  # +1 to include start and end


class HyperRangeBetaFloat(_AbstractHyper):
    """ Float between start and end but with a Beta Law distribution """

    def __init__(self, start=0, end=1, alpha=3, beta=1, random_state=None):
        if end <= start:
            raise ValueError("end can't be lower than start")

        if alpha <= 0:
            raise ValueError("alpha can't be less than 0")

        if beta <= 0:
            raise ValueError("beta can't be less than 0")

        self.start = start
        self.end = end
        self.alpha = alpha
        self.beta = beta
        self._beta_dist = scipy.stats.beta(a=self.alpha, b=self.beta)
        super().__init__(random_state=random_state)

    def _set_random_state(self):
        self._beta_dist.random_state = self._random_state
        return self

    def get_rand(self):
        return self._beta_dist.rvs() * (self.end - self.start) + self.start

    def get_size(self):
        beta_std = np.sqrt(self.alpha * self.beta / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1)))
        return (
            100 * beta_std * np.sqrt(12)
        )  # So that for a uniform law I assume '100' different values (and I scale down based on standard deviation)


class HyperRangeBetaInt(HyperRangeBetaFloat):
    __doc__ = HyperRangeBetaFloat.__doc__
    __doc__ += "\nForce integer result " ""

    def __init__(self, start=0, end=1, alpha=3, beta=1, random_state=None):
        super().__init__(start, end, alpha, beta, random_state)

    def get_rand(self):
        return int(super(HyperRangeBetaInt, self).get_rand() + 0.5)


class HyperLogRangeFloat(_AbstractHyper):
    """ Float log Uniform between start and end """

    def __init__(self, start, end, n=100, random_state=None):
        super().__init__(random_state=random_state)

        if start <= 0:
            raise ValueError("start can't be negative or null")

        if end <= 0:
            raise ValueError("end can't be negative or null")

        if end <= start:
            raise ValueError("end can't be lower than start")

        values = [
            np.log10(start) + (np.log10(end) - np.log10(start)) * i / n for i in range(n + 1)
        ]  # +1 to include start and end
        self.values = [np.exp(np.log(10) * v) for v in values]


class HyperListOfDifferentSizes(_AbstractHyper):
    """ to draw list of different sizes

    Examples
    --------
    >>> hp = HyperListOfDifferentSizes(HyperRangeInt(1, 5), HyperRangeInt(50, 150))
    >>> hp.get_rand()
    """

    def __init__(self, nb_dist, value_dist, random_state=None):
        self.nb_dist = nb_dist
        self.value_dist = value_dist
        super().__init__(random_state=random_state)

    def _set_random_state(self):
        _try_set_random_state(self.nb_dist, random_state=self._random_state)
        _try_set_random_state(self.value_dist, random_state=self._random_state)
        return self

    def get_rand(self):
        list_len = _get_rand(self.nb_dist, self.random_state)
        return [_get_rand(self.value_dist, self.random_state) for _ in range(list_len)]

    def get_size(self):
        return _get_size(self.nb_dist) * _get_size(self.value_dist)


class HyperComposition(_AbstractHyper):
    """ Composition of Distributions : randomly choice among several distributions

    the size of the values can be :
    * if size 1 : list of HyperParameters
    * if size 2 : list of weight * HyperParameters

    Parameters
    ----------
    dict_vals : list or tuple of size 2 or 1

    Examples
    --------
    >>> hp = HyperComposition([HyperRangeInt(0,100), HyperRangeInt(100, 1000)])
    >>> hp.get_rand()

    >>> hp = HyperComposition([(0.9,HyperRangeInt(0,100)), (0.1, HyperRangeInt(100, 1000))])
    >>> hp.get_rand()

    >>> hp = HyperComposition([(0.9,"choice_a"), (0.1, "choice_b")])
    >>> hp.get_rand()
    """

    def __init__(self, dict_vals, random_state=None):
        if isinstance(dict_vals, (list, tuple)):
            if isinstance(dict_vals[0], tuple):
                if len(dict_vals[0]) != 2:
                    raise ValueError("I need tuple of size 2")
                self.weights, self.hypers = zip(*dict_vals)
            else:
                nb = len(dict_vals)
                self.weights = [1 / nb] * nb
                self.hypers = dict_vals
        else:
            raise TypeError("'dict_vals' parameter should be a list or a tuple.")
        super().__init__(random_state=random_state)

    def get_rand(self):
        choice = self._random_state.choice(len(self.hypers), p=self.weights)
        sub_hyper = self.hypers[choice]
        return _get_rand(sub_hyper, random_state=self.random_state)

    def _set_random_state(self):
        for hyper in self.hypers:
            _try_set_random_state(hyper, self.random_state)

    def get_size(self):
        return sum([_get_size(sub_hyper) for sub_hyper in self.hypers])

    def __add__(self, other):
        """ test to help creation of nested hyperparameter with fewer lines of code. """
        if other is None:
            return self

        if isinstance(other, (HyperCrossProduct, dict)):
            weights = self.weights
            hypers = [h + other for h in self.hypers]
            return HyperComposition(list(zip(weights, hypers)), random_state=self.random_state)
        else:
            raise NotImplementedError("Can't sum those 2 hyperparameter")


class HyperCrossProduct(_AbstractHyper):
    """ Cartesian Product of Distribution

    The list of hyperameters can be :
        * dict or OrderedDict : in that case the object will draw dictionnary with them key and value
                                from the hyperparameter
        * list or tuple       : in that case the object will draw list (resp. tuple) from each element
                                within the list (resp. tuple)

    Parameters
    ----------
    list_of_hyperparameter: list of hyperameters like object

    Examples
    --------
    >>> hp = HyperCrossProduct([HyperRangeInt(0, 10), HyperRangeFloat(0, 1)])
    >>> hp.get_rand()

    >>> hp = HyperCrossProduct({"int_value": HyperRangeInt(0, 10), "float_value": HyperRangeFloat(0, 1)})
    >>> hp.get_rand()

    >>> hp = HyperCrossProduct({
    >>>     "int_value": scipy.stats.randint(0,10),
    >>>     "float_value": HyperRangeFloat(0,1),
    >>>     "choice": ("a","b","c"),"constant":10
    >>> })
    >>> hp.get_rand()
    """

    def __init__(self, list_of_hyperparameter, random_state=None):
        if not isinstance(list_of_hyperparameter, (list, tuple, dict)):
            raise TypeError("I don't know how to deal with that type of list of parameters")
        self.list_of_hyperparameter = list_of_hyperparameter
        super().__init__(random_state=random_state)

    def get_rand(self):
        if isinstance(self.list_of_hyperparameter, dict):
            res = self.list_of_hyperparameter.__class__()
            for k, hyper in self.list_of_hyperparameter.items():
                res[k] = _get_rand(hyper, random_state=self.random_state)
            return res
        elif isinstance(self.list_of_hyperparameter, (list, tuple)):
            res = [_get_rand(hyper, random_state=self.random_state) for hyper in self.list_of_hyperparameter]
            return self.list_of_hyperparameter.__class__(res)
        else:
            raise TypeError("I don't know how to deal with that type of list of parameters")

    def get_size(self):
        if isinstance(self.list_of_hyperparameter, dict):
            return np.product([_get_size(hyper) for key, hyper in self.list_of_hyperparameter.items()])

        elif isinstance(self.list_of_hyperparameter, (list, tuple)):
            return np.product([_get_size(hyper) for hyper in self.list_of_hyperparameter])

        else:
            raise TypeError("I don't know how to deal with that type of list of parameters")

    def _set_random_state(self):
        if isinstance(self.list_of_hyperparameter, dict):
            for k, hyper in self.list_of_hyperparameter.items():
                _try_set_random_state(hyper, self.random_state)

        elif isinstance(self.list_of_hyperparameter, (list, tuple)):
            for hyper in self.list_of_hyperparameter:
                _try_set_random_state(hyper, self.random_state)

        else:
            raise TypeError("I don't know how to deal with that type of list of parameters")

    def __add__(self, other):
        """ test to help creation of nested hyperparameter with fewer lines of code """
        if other is None:
            return self

        res = {}
        for k, v in self.list_of_hyperparameter.items():
            res[k] = v

        if isinstance(other, HyperCrossProduct):
            for k, v in other.list_of_hyperparameter.items():
                res[k] = v

        elif isinstance(other, HyperComposition):
            return other.__add__(self)

        elif isinstance(other, dict):
            for k, v in other.items():
                res[k] = v
        else:
            raise TypeError(f"I don't know how to add this type {type(other)}")

        return HyperCrossProduct(res, random_state=self.random_state)
