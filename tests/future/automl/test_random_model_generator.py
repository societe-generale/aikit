from copy import deepcopy

import numpy as np
import pytest
from sklearn.utils import check_random_state

from aikit.future.automl.random_model_generator import RandomModelGenerator
from aikit.future.enums import StepCategory, VariableType
from aikit.future.graph import convert_graph_to_code
from aikit.future.util.graph import get_terminal_nodes
from aikit.future.util.serialization import sklearn_model_from_param
from aikit.ml_machine.ml_machine import random_list_generator, _create_all_combinations


@pytest.mark.parametrize("type_of_iterator", ["default", "block_search", "block_search_random"])
def test_random_model_generator_iterator(type_of_iterator, dataset_and_automl_config):
    df, y, _, automl_config = dataset_and_automl_config

    random_model_generator = RandomModelGenerator(automl_config=automl_config, random_state=123)

    if type_of_iterator == "default":
        iterator = random_model_generator.default_models_iterator()
    elif type_of_iterator == "block_search":
        iterator = random_model_generator.block_search_iterator(random_order=False)
    elif type_of_iterator == "block_search_random":
        iterator = random_model_generator.block_search_iterator(random_order=True)
    else:
        raise NotImplementedError(f"Unknown iterator type: {type_of_iterator}")

    assert hasattr(iterator, "__iter__")

    # verif iterator
    for model in iterator:

        assert isinstance(model, tuple)
        assert len(model) == 3
        graph, all_models_params, block_to_use = model

        terminal_nodes = get_terminal_nodes(graph)
        assert len(terminal_nodes) == 1
        assert terminal_nodes[0][0] == StepCategory.Model

        assert hasattr(graph, "edges")
        assert hasattr(graph, "nodes")

        assert isinstance(all_models_params, dict)
        for node in graph.nodes:
            assert node in all_models_params

        assert isinstance(block_to_use, (tuple, list))
        for b in block_to_use:
            assert b in VariableType.alls

        result = convert_graph_to_code(graph, all_models_params, return_mapping=True)
        assert isinstance(result, dict)
        assert "name_mapping" in result
        assert "json_code" in result

        sk_model = sklearn_model_from_param(result["json_code"])
        assert hasattr(sk_model, "fit")

        if type_of_iterator == "default" and ('Model', ('Model', 'RandomForestClassifier')) in graph.nodes:
            # in that case I'll actually do the fitting here
            # I'll simplify the model to have 2 estimators (faster)

            all_models_params[('Model', ('Model', 'RandomForestClassifier'))]["n_estimators"] = 2
            result = convert_graph_to_code(graph, all_models_params, return_mapping=True)
            sk_model = sklearn_model_from_param(result["json_code"])

            sub_index = np.concatenate((np.where(y == 0)[0][0:10], np.where(y == 1)[0][0:10]), axis=0)
            # Needs at least 20 observations to make sure all transformers works
            sk_model.fit(df.iloc[sub_index, :], y[sub_index])

            yhat = sk_model.predict(df.head(2))
            assert yhat.shape == (2,)


def test_random_list_generator():
    elements = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

    for i in range(2):
        if i == 0:
            probas = [1 / min(i + 1, 10 + 1 - i) for i in range(len(elements))]
        else:
            probas = None

        gen = random_list_generator(elements, probas, random_state=123)

        assert hasattr(gen, "__iter__")

        elements_random_order = list(gen)
        assert len(elements_random_order) == len(elements)
        assert set(elements_random_order) == set(elements)

        elements_random_order2 = list(random_list_generator(elements, probas=probas, random_state=123))
        elements_random_order3 = list(random_list_generator(elements, probas=probas, random_state=456))
        elements_random_order4 =\
            list(random_list_generator(elements, probas=probas, random_state=check_random_state(123)))

        assert len(elements_random_order2) == len(elements)
        assert set(elements_random_order2) == set(elements)

        assert len(elements_random_order3) == len(elements)
        assert set(elements_random_order3) == set(elements)

        assert elements_random_order2 == elements_random_order
        assert elements_random_order3 != elements_random_order
        assert elements_random_order4 == elements_random_order

    with pytest.raises(ValueError):
        list(random_list_generator(elements, probas=[0.1], random_state=123))  # error : probas not the right length

    with pytest.raises(ValueError):
        list(random_list_generator(elements, probas=[0] * len(elements), random_state=123))  # error : probas 0


def test_random_list_generator_empty():
    elements_random_order = list(random_list_generator([], [], random_state=123))
    assert elements_random_order == []


def _all_same(all_gen):
    """ helper function to test if things are all the same """
    if len(all_gen) == 1:
        return True
    for gen in all_gen[1:]:
        if gen != all_gen[0]:
            return False
    # I don't want to use 'set' because thing might not be hashable
    return True


@pytest.mark.parametrize(
    "specific_hyper, only_random_forest",
    [(False, False), (True, False), (False, True), (False, False)],
)
def test_random_model_generator_random(specific_hyper, only_random_forest, dataset_and_automl_config):
    # num_only, specific_hyper, only_random_forest = False, True, True
    df, y, dataset_type, automl_config = dataset_and_automl_config

    if specific_hyper:
        automl_config.specific_hyper = {("Model", "RandomForestClassifier"): {"n_estimators": [10, 20]}}

    if only_random_forest:
        automl_config.filter_models(Model="RandomForestClassifier")

    random_model_generator = RandomModelGenerator(automl_config=automl_config, random_state=123)

    all_gen = []
    rf_key = None
    for _ in range(10):
        model = random_model_generator.draw_random_graph()
        all_gen.append(model)

        assert isinstance(model, tuple)
        assert len(model) == 3

        graph, all_models_params, block_to_use = model

        assert hasattr(graph, "edges")
        assert hasattr(graph, "nodes")

        assert isinstance(all_models_params, dict)
        for node in graph.nodes:
            assert node in all_models_params

        assert isinstance(block_to_use, (tuple, list))
        for b in block_to_use:
            assert b in VariableType.alls

        result = convert_graph_to_code(graph, all_models_params, return_mapping=True)
        assert isinstance(result, dict)
        assert "name_mapping" in result
        assert "json_code" in result

        sk_model = sklearn_model_from_param(result["json_code"])
        assert hasattr(sk_model, "fit")

        rf_key = ("Model", ("Model", "RandomForestClassifier"))
        if only_random_forest:
            assert rf_key in all_models_params

        if specific_hyper:
            if rf_key in all_models_params:
                assert all_models_params[rf_key]["n_estimators"] in (10, 20)

        if ('Model', ('Model', 'RandomForestClassifier')) in graph.nodes:
            # in that case I'll actually do the fitting here
            # I'll simplify the model to have 2 estimators (faster)
            all_models_params_copy = deepcopy(all_models_params)
            all_models_params_copy[('Model', ('Model', 'RandomForestClassifier'))]["n_estimators"] = 2
            result = convert_graph_to_code(graph, all_models_params_copy, return_mapping=True)
            sk_model = sklearn_model_from_param(result["json_code"])

            sub_index = np.concatenate((np.where(y == 0)[0][0:100], np.where(y == 1)[0][0:100]), axis=0)
            # Needs at least 20 observations to make sure all transformers works
            if hasattr(sk_model, "verbose"):
                sk_model.verbose = True
            sk_model.fit(df.iloc[sub_index, :], y[sub_index])

            yhat = sk_model.predict(df.head(2))
            assert yhat.shape == (2,)

    if not only_random_forest:
        assert any([rf_key not in m[1] for m in all_gen])  # Check that RandomForest wasn't drawn every time

    # re-draw them thing with other seed
    random_model_generator = RandomModelGenerator(automl_config=automl_config, random_state=123)
    all_gen2 = [random_model_generator.draw_random_graph() for _ in range(10)]

    all_graphs1, all_params1, all_blocks1 = zip(*all_gen)
    all_graphs2, all_params2, all_blocks2 = zip(*all_gen2)

    assert not _all_same(all_params1)
    assert not _all_same(all_graphs1)
    if dataset_type != "numeric":
        assert not _all_same(all_blocks1)  # only one block

    all_graphs1_node_edges = [(g.nodes, g.edges) for g in all_graphs1]
    all_graphs2_node_edges = [(g.nodes, g.edges) for g in all_graphs2]
    # I need to test equality of nodes and edges ... directly == on networkx graph doesn't work

    # separate test to isolate exactly what changes
    assert all_graphs1_node_edges == all_graphs2_node_edges
    assert all_params1 == all_params2
    assert all_blocks1 == all_blocks2

    # re-draw by resetting generator
    random_model_generator.random_state = 123
    all_gen3 = [random_model_generator.draw_random_graph() for _ in range(10)]

    all_graphs3, all_params3, all_blocks3 = zip(*all_gen3)
    all_graphs3_node_edges = [(g.nodes, g.edges) for g in all_graphs3]
    # I need to test equality of nodes and edgs ... directly == on networkx graph doesn't work

    # separate test to isolate exactly what changes
    assert all_graphs1_node_edges == all_graphs3_node_edges
    assert all_params1 == all_params3
    assert all_blocks1 == all_blocks3

    # Re-draw by passing a random sate
    random_state = check_random_state(123)
    random_model_generator = RandomModelGenerator(automl_config=automl_config, random_state=random_state)
    all_gen4 = [random_model_generator.draw_random_graph() for _ in range(10)]

    all_graphs4, all_params4, all_blocks4 = zip(*all_gen4)
    all_graphs4_node_edges = [(g.nodes, g.edges) for g in all_graphs4]
    # I need to test equality of nodes and edgs ... directly == on networkx graph doesn't work

    # separate test to isolate exactly what changes
    assert all_graphs1_node_edges == all_graphs4_node_edges
    assert all_params1 == all_params4
    assert all_blocks1 == all_blocks4


@pytest.mark.longtest
@pytest.mark.parametrize(
    "specific_hyper, only_random_forest",
    [(True, True)],
)
def test_random_model_generator_random_fail_seed(titanic_dataset_automl_config, specific_hyper, only_random_forest):
    test_random_model_generator_random(specific_hyper, only_random_forest, titanic_dataset_automl_config)


def test__create_all_combinations():
    def _check_all_list_of_blocks(_all_list_of_blocks, _all_blocks_to_use):
        assert isinstance(_all_list_of_blocks, list)
        for blocks_to_use in _all_list_of_blocks:
            assert isinstance(blocks_to_use, tuple)
            assert 1 <= len(blocks_to_use) <= len(_all_blocks_to_use)
            for b in blocks_to_use:
                assert b in _all_blocks_to_use
            assert len(set(blocks_to_use)) == len(blocks_to_use)
        assert len(set(_all_list_of_blocks)) == len(_all_list_of_blocks)  # no duplicate

    all_blocks_to_use = ("CAT", "NUM", "TEXT")
    all_list_of_blocks = _create_all_combinations(all_blocks_to_use, 1, 1)

    _check_all_list_of_blocks(all_list_of_blocks, all_blocks_to_use)

    all_blocks_to_use = ("a", "b", "c", "d")
    all_list_of_blocks = _create_all_combinations(all_blocks_to_use, 2, 2)
    _check_all_list_of_blocks(all_list_of_blocks, all_blocks_to_use)

    with pytest.raises(ValueError):
        _create_all_combinations(all_blocks_to_use, 0, 2)  # 0 : not possible

    with pytest.raises(ValueError):
        _create_all_combinations(all_blocks_to_use, 2, 0)  # 0 : not possible

    with pytest.raises(ValueError):
        _create_all_combinations(["a", "a"], 2, 2)  # duplicate entry

    assert _create_all_combinations(("a",), 1, 1) == []
    assert set(_create_all_combinations(("a", "b"), 1, 1)) == {("a",), ("b",)}
    assert set(_create_all_combinations(("a", "b", "c"), 1, 1)) \
           == {("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c")}
