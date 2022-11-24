import itertools
import logging
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from sklearn.utils import check_random_state

from ._registry import MODEL_REGISTRY, allow_conditional
from ._steps import get_needed_steps, modify_var_type_none_to_default, \
    create_var_type_from_steps, modify_var_type_all_default_to_none
from .hyper_parameters import HyperCrossProduct, HyperComposition, HyperMultipleChoice
from ..util import get_all_var_type, get_var_type_columns_dict
from ..util.dict import filter_dict_on_key_value_pairs
from .. import graph as mg

_logger = logging.getLogger(__name__)


def _create_all_combinations(all_blocks_to_use, max_number_of_blocks_to_test, max_number_of_blocks_to_remove):
    """ create all the combinations of 'all_blocks_to_use' that have either
    * at most 'max_number_of_blocks_to_test' elements    OR
    * at most 'max_number_of_blocks_to_remove' elements removed

    Parameters
    ----------
    max_number_of_blocks_to_test : int, default=1
        will include ALL the models with at most 'max_number_of_blocks_to_test' (all combinations)

    max_number_of_blocks_to_remove : int, default=1
        will include ALL the models with at most 'max_number_of_blocks_to_remove' blocks REMOVED (al combinations)
    """

    if max_number_of_blocks_to_test < 1:
        raise ValueError("'max_number_of_blocks_to_test' must be >= 1")

    if max_number_of_blocks_to_remove < 1:
        raise ValueError("'max_number_of_blocks_to_remove' must be >= 1")

    if len(set(all_blocks_to_use)) != len(all_blocks_to_use):
        raise ValueError("'all_blocks_to_use' must not contain duplicate")

    if max_number_of_blocks_to_test >= len(all_blocks_to_use) - 1:
        max_number_of_blocks_to_test = len(all_blocks_to_use) - 1

    if max_number_of_blocks_to_remove >= len(all_blocks_to_use) - 1:
        max_number_of_blocks_to_remove = len(all_blocks_to_use) - 1

    # Create full list of blocks_to_use to be tried
    set_blocks_to_use = set()
    for r in range(1, max_number_of_blocks_to_test + 1):
        for blocks_to_use in itertools.combinations(all_blocks_to_use, r=r):
            if blocks_to_use not in set_blocks_to_use:
                set_blocks_to_use.add(blocks_to_use)

    for r in range(1, max_number_of_blocks_to_remove + 1):
        for blocks_to_remove in itertools.combinations(all_blocks_to_use, r=r):
            blocks_to_use = tuple([b for b in all_blocks_to_use if b not in blocks_to_remove])
            if blocks_to_use not in set_blocks_to_use:
                set_blocks_to_use.add(blocks_to_use)

    return list(set_blocks_to_use)


def _random_list_generator(elements, probas=None, random_state=None):
    """ helper to create a generator in a random order.

    Parameters
    ----------
    elements : list or iterable
        the list of elements

    probas : None or list of proba
        if not None, the 'un-normalized' proba to draw each element

    random_state : None, int, or generator
        the random generator

    Yield
    -----
    element in random order
    """

    elements = list(elements)

    random_state = check_random_state(random_state)

    if probas is not None:
        probas = list(probas)
        if len(elements) != len(probas):
            raise ValueError("'elements' and 'probas' must have the same length")

        if len(elements) > 0 and np.array(probas).min() <= 0:
            raise ValueError("'probas' should be > 0")

    all_indexes = list(range(len(elements)))
    remaining_indexes = all_indexes
    while len(remaining_indexes) > 0:
        if probas is None:
            ind = random_state.choice(remaining_indexes)
        else:
            p = np.array([probas[i] for i in remaining_indexes])
            p = p / p.sum()
            ind = random_state.choice(remaining_indexes, p=p)

        remaining_indexes = [r for r in remaining_indexes if r != ind]
        yield elements[ind]


class RandomModelGenerator(object):
    """ Random model generator """

    def __init__(self, automl_config, allow_block_selection=True, block_probas=0.9, random_state=None):
        self.hyper_parameters = {}
        self.random_probas = None
        self._hyper_block = None

        self.automl_config = automl_config
        self.allow_block_selection = allow_block_selection
        self.block_probas = block_probas
        self.random_state = random_state

        self.prepare_hyper_parameters()

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, new_random_state):
        self._random_state = check_random_state(new_random_state)
        for k, hyper in self.hyper_parameters.items():
            hyper.random_state = self._random_state

        if self._hyper_block is not None:
            self._hyper_block.random_state = self._random_state

    def prepare_hyper_parameters(self):
        """ create the custom hyperparameters """
        self.hyper_parameters = {}

        for model in self.automl_config.models_to_keep:
            hyper = MODEL_REGISTRY.hyper_parameters[model]
            hyper.random_state = self.random_state
            assert isinstance(hyper, (HyperCrossProduct, HyperComposition))

            other_hyper = self.automl_config.specific_hyper.get(model, None)
            if other_hyper is None:
                self.hyper_parameters[model] = hyper
            else:
                self.hyper_parameters[model] = hyper + other_hyper  # random_state here ?

        return self

    def default_models_iterator(self):
        """ iterator that generate the list of default models to test """
        all_choices_by_steps = []
        for step in self.automl_config.needed_steps:
            if step["optional"]:
                all_choices = [(None, None)]
            else:
                all_choices = [n for n in self.automl_config.models_to_keep if n[0] == step["step"]]
                # Maybe we'd like to remove some choices here for some steps
            all_choices_by_steps.append([(step["step"], c) for c in all_choices])

        all_models_steps = [OrderedDict(m) for m in itertools.product(*all_choices_by_steps)]

        for models_by_steps in all_models_steps:
            # Blocks
            blocks_to_use = tuple(self.automl_config.columns_block.keys())  # keep all blocks

            # Hyper
            hyper_parameters_by_step = {}
            for step_name, model_name in models_by_steps.items():
                if model_name[0] is not None:
                    # If default_parameters are present in registry use them,
                    # otherwise use {} (and so will go back to default parameters of the model)
                    default_parameters = MODEL_REGISTRY.default_hyper_parameters.get(model_name, {})
                    hyper_parameters_by_step[(step_name, model_name)] = default_parameters

            # Call the 'draw_random_graph' method with preset params
            simplified_graph, all_models_params, blocks_to_use = self.draw_random_graph(
                blocks_to_use=blocks_to_use,
                models_by_steps=models_by_steps,
                hyper_parameters_by_step=hyper_parameters_by_step)

            yield simplified_graph, all_models_params, blocks_to_use

    def block_search_iterator(self,
                              max_number_of_blocks_to_test=1,
                              max_number_of_blocks_to_remove=1,
                              random_order=True):
        """ Iterator that generates the list of models to test when we are searching for blocks.
        The iteration order can be random, in that case the order is drawn by a law depending
        on the size of the blocks to use.

        Parameters
        ----------
        max_number_of_blocks_to_test : int, default=1
            will include ALL the models with at most 'max_number_of_blocks_to_test' (all combinations)

        max_number_of_blocks_to_remove : int, default=1
            will include ALL the models with at most 'max_number_of_blocks_to_remove' blocks REMOVED (al combinations)

        random_order: boolean, default=True
            if True will iterate in a random order

        Yields
        ------
        simplified_graph, all_models_params, blocks_to_use
        """
        if random_order:
            block_search_models = list(
                self._block_search_models_iterator(
                    max_number_of_blocks_to_test=max_number_of_blocks_to_test,
                    max_number_of_blocks_to_remove=max_number_of_blocks_to_remove)
            )

            n = len(self.automl_config.columns_block.keys())

            probas = []
            for _, _, blocks_to_use in block_search_models:
                p = 1 / min(len(blocks_to_use), n - len(blocks_to_use))
                probas.append(p)

            yield from _random_list_generator(block_search_models, probas=probas, random_state=self.random_state)

        else:
            yield from self._block_search_models_iterator(
                max_number_of_blocks_to_test=max_number_of_blocks_to_test,
                max_number_of_blocks_to_remove=max_number_of_blocks_to_remove)

    def _block_search_models_iterator(self, max_number_of_blocks_to_test=1, max_number_of_blocks_to_remove=1):
        """ Iterator that generate the list of models to test when we are searching for blocks.

        Parameters
        ----------
        max_number_of_blocks_to_test : int, default=1
            will include ALL the models with at most 'max_number_of_blocks_to_test' (all combinations)

        max_number_of_blocks_to_remove : int, default=1
            will include ALL the models with at most 'max_number_of_blocks_to_remove' blocks REMOVED (al combinations)

        Yields
        ------
        simplified_graph, all_models_params, blocks_to_use
        """
        all_blocks_to_use = tuple(self.automl_config.columns_block.keys())  # keep all blocks
        if len(all_blocks_to_use) <= 1:
            return  # No models

        list_blocks_to_use = _create_all_combinations(
            all_blocks_to_use,
            max_number_of_blocks_to_test=max_number_of_blocks_to_test,
            max_number_of_blocks_to_remove=max_number_of_blocks_to_remove,
        )

        for blocks_to_use in list_blocks_to_use:
            needed_steps_filtered, columns_informations_filtered, all_columns_keep =\
                self._filter_based_on_blocks(blocks_to_use)

            all_choices_by_steps = []
            for step in needed_steps_filtered:
                if step["optional"]:
                    all_choices = [(None, None)]
                else:
                    # Maybe we can remove choices for some particular steps
                    all_choices = [n for n in self.automl_config.models_to_keep_block_search if n[0] == step["step"]]

                all_choices_by_steps.append([(step["step"], c) for c in all_choices])

            all_models_steps = [OrderedDict(m) for m in itertools.product(*all_choices_by_steps)]

            for models_by_steps in all_models_steps:
                hyper_parameters_by_step = {}
                for step_name, model_name in models_by_steps.items():
                    if model_name[0] is not None:
                        # If default_parameters present in register use it,
                        # otherwise use {} (and so will go back to default parameter of the model)
                        default_parameters = MODEL_REGISTRY.default_hyper_parameters.get(model_name, {})
                        hyper_parameters_by_step[(step_name, model_name)] = default_parameters

                simplified_graph, all_models_params, blocks_to_use = self.draw_random_graph(
                    blocks_to_use=blocks_to_use,
                    models_by_steps=models_by_steps,
                    hyper_parameters_by_step=hyper_parameters_by_step)

                yield simplified_graph, all_models_params, blocks_to_use

    def _filter_based_on_blocks(self, blocks_to_use):
        """ helper function to filter what is needed when knowing which blocks are used """

        # Filter columns
        temp_kept_columns = []
        for b in blocks_to_use:
            temp_kept_columns += self.automl_config.columns_block[b]

        columns_informations_filtered = filter_dict_on_key_value_pairs(
            self.automl_config.columns_informations, lambda k, v: (k in temp_kept_columns and v["ToKeep"]))

        all_columns_keep = sorted(columns_informations_filtered.keys()) == sorted(
            self.automl_config.columns_informations)

        # Filter steps
        needed_steps_filtered_temp = get_needed_steps(
            columns_informations_filtered, self.automl_config.problem_type)
        # If user remove some steps => need to take the intersection
        steps_in_config = [s2["step"] for s2 in self.automl_config.needed_steps]
        needed_steps_filtered = [s for s in needed_steps_filtered_temp if s["step"] in steps_in_config]

        return needed_steps_filtered, columns_informations_filtered, all_columns_keep

    def draw_random_graph(self, blocks_to_use=None, models_by_steps=None, hyper_parameters_by_step=None):
        """ draw a random model graph and its parameters """

        # Draw blocks
        if blocks_to_use is None:
            if self.allow_block_selection:
                blocks_to_use = self._draw_blocks()
            else:
                blocks_to_use = tuple(sorted(self.automl_config.columns_block.keys()))

        needed_steps_filtered, columns_informations_filtered, all_columns_keep =\
            self._filter_based_on_blocks(blocks_to_use)

        # Draw models
        if models_by_steps is None:
            models_by_steps = self._draw_random_model_by_step(
                needed_steps=needed_steps_filtered,
                models_to_keep=self.automl_config.models_to_keep,
                random_probas=self.random_probas,
            )

        # Draw hyperparameters
        if hyper_parameters_by_step is None:
            hyper_parameters_by_step = self._draw_hyperparameters(models_by_steps)

        # Variable types
        # TODO: put this in a dedicated method
        # All var type still present
        remaining_var_type = get_all_var_type(columns_informations_filtered)
        # Retrieve 'saved' var type from registry
        var_type_by_steps = create_var_type_from_steps(models_by_steps)
        # Modify None to remaining_var_type + intersect other with remaining var_type
        var_type_by_steps = modify_var_type_none_to_default(var_type_by_steps, remaining_var_type)
        if all_columns_keep:
            var_type_by_steps = modify_var_type_all_default_to_none(var_type_by_steps, remaining_var_type)
        # 'columns for each type of variable
        var_type_columns_dico = get_var_type_columns_dict(columns_informations_filtered)

        # Create graph and model params
        _, simplified_graph, all_models_params = self._create_graph(
            var_type_by_steps=var_type_by_steps,
            hyper_parameters_by_step=hyper_parameters_by_step,
            var_type_columns_dico=var_type_columns_dico,
        )
        return simplified_graph, all_models_params, blocks_to_use

    def _create_graph(self, var_type_by_steps, hyper_parameters_by_step, var_type_columns_dico):
        """ create the complete graph """

        # Create complete graph (including None transformers)
        graph, new_steps = mg.create_graphical_representation(var_type_by_steps)
        # mg.graphviz_modelgraph(Graph)

        if len(new_steps) > 0:
            hyper_parameters_by_step = deepcopy(hyper_parameters_by_step)
            for step, vtype in new_steps.items():
                var_type_by_steps[step] = vtype
                hyper_parameters_by_step[step] = {}

        # Simplify graph (exclude None transformers)
        simplified_graph = mg.simplify_none_node(graph)

        # Add selectors
        simplified_graph, all_models_params = mg.add_columns_selector(
            graph=simplified_graph,
            var_type_node_dico=var_type_by_steps,
            var_type_columns_dico=var_type_columns_dico,
            all_models_params=hyper_parameters_by_step)

        # mg.graphviz_modelgraph(simplified_Graph)
        return graph, simplified_graph, all_models_params

    # Steps
    def _draw_random_model_by_step(self, needed_steps, models_to_keep, log_uniform=True, random_probas=None):
        """
        for each step draw a random model among the possible models,

        Parameters
        ----------
        * needed_steps : list of dictionaries representing each step, each one with a 'step' key and an 'optional' key

        * models_to_keep : list of models that we want to draw within

        * log_uniform : bool, default = True
            if True (and random_probas is not None), will draw a model with probability proportional
            to 'log(1 + hyperparameter.size)'

        * random_probas : dictionary of proba array for each step or None
            if None will use uniform (or log_uniform) otherwise will draw according to that probability
        """
        needed_steps_reordered =\
            sorted(needed_steps, key=lambda x: MODEL_REGISTRY._drawing_order.get(x["step"], -1))

        # TODO : specify a random_state + save random state ?
        # TODO : allow conditional probas to draw steps  => create a Graphical Proba Model to handle implication
        models_by_steps_reordered = OrderedDict()
        for step in needed_steps_reordered:

            all_choices = [n for n in models_to_keep
                           if n[0] == step["step"]
                           and allow_conditional(model=n, models_by_steps=models_by_steps_reordered)]

            if step["optional"]:
                # TODO: put that into a Constant
                all_choices.append((None, None))

            if len(all_choices) == 0:
                _logger.info(f"Skip this step: {step}")
                continue

            if random_probas is None:
                p = None
                if log_uniform:
                    # TODO: 'nothing' elements of size 1 will never be drawn
                    all_hypers = [MODEL_REGISTRY.hyper_parameters.get(model_name, None) for model_name in all_choices]
                    all_sizes = np.array([10 if h is None else h.size for h in all_hypers])
                    # Warning: when hyperparameter is custom, size is not good
                    p = np.log1p(all_sizes)
                    p /= p.sum()
                # TODO: choice should follow a 'log uniform' on possible combinations number
                # TODO: proxy with a method 'nb' on hyperparameters ?

            else:
                default_p = 1 / len(all_choices)
                p = np.array([random_probas.get((step["step"], c), default_p) for c in all_choices])
                p /= np.sum(p)

            ii = np.arange(len(all_choices))
            chosen_class = all_choices[ii[self.random_state.choice(ii, size=1, p=p)[0]]]
            models_by_steps_reordered[step["step"]] = chosen_class

        # Put everything into the corrected order
        models_by_steps = OrderedDict()
        for step in needed_steps:
            models_by_steps[step["step"]] = models_by_steps_reordered[step["step"]]

        return models_by_steps

    def _draw_hyperparameters(self, all_steps):
        """ draw the random hyperparameters """
        # TODO: specify a random state

        all_hyper = {}
        for step_name, model_name in all_steps.items():
            if model_name[0] is not None:
                hyper = self.hyper_parameters.get(
                    model_name, None
                )  # draw from self.hyper_parameters (which might have been updated by the 'specific_hyper' )
                all_hyper[(step_name, model_name)] = hyper

        def custom_draw(_hyper):
            if _hyper:
                return _hyper.get_rand()
            else:
                return {}

        all_models_params = {n: custom_draw(all_hyper[n]) for n in all_hyper.keys()}
        return all_models_params

    def _draw_blocks(self):
        """ draw the columns block to keep """
        if len(self.automl_config.columns_block.keys()) == 1:
            return tuple(sorted(self.automl_config.columns_block.keys()))

        if self._hyper_block is None:
            self._hyper_block = HyperMultipleChoice(
                tuple(sorted(self.automl_config.columns_block.keys())),
                min_number=1,
                proba_choice=self.block_probas,
                random_state=self.random_state)

        return self._hyper_block.get_rand()
