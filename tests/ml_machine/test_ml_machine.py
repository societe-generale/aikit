# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:20:02 2019

@author: lmassoul032513
"""

from sklearn.utils import check_random_state

from aikit.datasets.datasets import load_dataset, DatasetEnum
from aikit.enums import TypeOfProblem, TypeOfVariables
from aikit.ml_machine.ml_machine import (
    AutoMlConfig,
    JobConfig,
    RandomModelGenerator,
    AutoMlResultReader,
    MlJobManager,
    MlJobRunner,
)
from aikit.ml_machine.model_graph import convert_graph_to_code
from aikit.model_definition import sklearn_model_from_param

from aikit.ml_machine.ml_machine_guider import AutoMlModelGuider
from aikit.ml_machine.data_persister import FolderDataPersister


def loader():
    """ modify this function to load the data
    
    Returns
    -------
    dfX, y 
    
    Or
    dfX, y, groups
    
    """
    dfX, y, _, _, _ = load_dataset(DatasetEnum.titanic)
    return dfX, y


def get_automl_config():
    dfX, y = loader()
    auto_ml_config = AutoMlConfig(dfX, y)
    auto_ml_config.guess_everything()

    return dfX, y, auto_ml_config


def test_AutoMlConfig():

    dfX, y, auto_ml_config = get_automl_config()

    assert auto_ml_config.type_of_problem == TypeOfProblem.CLASSIFICATION
    assert auto_ml_config.columns_informations is not None


def test_JobConfig():

    dfX, y, auto_ml_config = get_automl_config()

    job_config = JobConfig()
    job_config.guess_cv(auto_ml_config)

    assert job_config.cv is not None

    job_config.guess_scoring(auto_ml_config)
    assert isinstance(job_config.scoring, list)


def test_RandomModelGenerator_default():

    dfX, y, auto_ml_config = get_automl_config()

    random_model_generator = RandomModelGenerator(auto_ml_config=auto_ml_config, random_state=123)

    # verif iterator
    for model in random_model_generator.iterator_default_models():

        assert isinstance(model, tuple)
        assert len(model) == 3
        Graph, all_models_params, block_to_use = model

        assert hasattr(Graph, "edges")
        assert hasattr(Graph, "nodes")

        assert isinstance(all_models_params, dict)
        for node in Graph.node:
            assert node in all_models_params

        assert isinstance(block_to_use, (tuple, list))
        for b in block_to_use:
            assert b in TypeOfVariables.alls

        result = convert_graph_to_code(Graph, all_models_params, also_returns_mapping=True)
        assert isinstance(result, dict)
        assert "name_mapping" in result
        assert "json_code" in result

        model = sklearn_model_from_param(result["json_code"])
        assert hasattr(model, "fit")


def _all_same(all_gen):
    """ helper function to test if things are all the same """
    if len(all_gen) == 1:
        return True
    for gen in all_gen[1:]:
        if gen != all_gen[0]:
            return False
    # I don't want to use 'set' because thing might not be hashable

    return True


def test_RandomModelGenerator_random():

    dfX, y, auto_ml_config = get_automl_config()

    random_model_generator = RandomModelGenerator(auto_ml_config=auto_ml_config, random_state=123)

    all_gen = []
    for _ in range(10):
        model = random_model_generator.draw_random_graph()
        all_gen.append(model)

        assert isinstance(model, tuple)
        assert len(model) == 3

        Graph, all_models_params, block_to_use = model

        assert hasattr(Graph, "edges")
        assert hasattr(Graph, "nodes")

        assert isinstance(all_models_params, dict)
        for node in Graph.node:
            assert node in all_models_params

        assert isinstance(block_to_use, (tuple, list))
        for b in block_to_use:
            assert b in TypeOfVariables.alls

        result = convert_graph_to_code(Graph, all_models_params, also_returns_mapping=True)
        assert isinstance(result, dict)
        assert "name_mapping" in result
        assert "json_code" in result

        model = sklearn_model_from_param(result["json_code"])
        assert hasattr(model, "fit")

    ### re-draw them thing with other seed ###
    random_model_generator = RandomModelGenerator(auto_ml_config=auto_ml_config, random_state=123)
    all_gen2 = [random_model_generator.draw_random_graph() for _ in range(10)]

    all_graphs1, all_params1, all_blocks1 = zip(*all_gen)
    all_graphs2, all_params2, all_blocks2 = zip(*all_gen2)

    assert not _all_same(all_params1)
    assert not _all_same(all_graphs1)
    assert not _all_same(all_blocks1)

    all_graphs1_node_edges = [(g.nodes, g.edges) for g in all_graphs1]
    all_graphs2_node_edges = [(g.nodes, g.edges) for g in all_graphs2]
    # I need to test equality of nodes and edgs ... directly == on networkx graph doesn't work

    # separate test to isolate exactly what changes
    assert all_graphs1_node_edges == all_graphs2_node_edges
    assert all_params1 == all_params2
    assert all_blocks1 == all_blocks2

    ### re-draw by resetting generator ###
    random_model_generator.random_state = 123
    all_gen3 = [random_model_generator.draw_random_graph() for _ in range(10)]

    all_graphs3, all_params3, all_blocks3 = zip(*all_gen3)
    all_graphs3_node_edges = [(g.nodes, g.edges) for g in all_graphs3]
    # I need to test equality of nodes and edgs ... directly == on networkx graph doesn't work

    # separate test to isolate exactly what changes
    assert all_graphs1_node_edges == all_graphs3_node_edges
    assert all_params1 == all_params3
    assert all_blocks1 == all_blocks3

    ### Re-draw by passing a random sate
    random_state = check_random_state(123)
    random_model_generator = RandomModelGenerator(auto_ml_config=auto_ml_config, random_state=random_state)
    all_gen4 = [random_model_generator.draw_random_graph() for _ in range(10)]

    all_graphs4, all_params4, all_blocks4 = zip(*all_gen4)
    all_graphs4_node_edges = [(g.nodes, g.edges) for g in all_graphs4]
    # I need to test equality of nodes and edgs ... directly == on networkx graph doesn't work

    # separate test to isolate exactly what changes
    assert all_graphs1_node_edges == all_graphs4_node_edges
    assert all_params1 == all_params4
    assert all_blocks1 == all_blocks4


# In[] :
def test_create_everything_sequentially(tmpdir):

    # DataPersister
    data_persister = FolderDataPersister(base_folder=tmpdir)

    # Data
    dfX, y = loader()

    # Auto Ml Config
    auto_ml_config = AutoMlConfig(dfX, y)
    auto_ml_config.guess_everything()
    assert auto_ml_config

    # Job Config
    job_config = JobConfig()
    job_config.guess_scoring(auto_ml_config)
    job_config.guess_cv(auto_ml_config)
    assert job_config

    # Result Reader
    result_reader = AutoMlResultReader(data_persister)
    assert result_reader  # just verify the object was created

    # Auto ml guider
    auto_ml_guider = AutoMlModelGuider(
        result_reader=result_reader, job_config=job_config, metric_transformation="default", avg_metric=True
    )
    assert auto_ml_guider

    # Job Controller
    job_controller = MlJobManager(
        auto_ml_config=auto_ml_config,
        job_config=job_config,
        auto_ml_guider=auto_ml_guider,
        data_persister=data_persister,
        seed=None,
    )

    assert job_controller

    # Job Runner
    job_runner = MlJobRunner(
        dfX=dfX,
        y=y,
        groups=None,
        auto_ml_config=auto_ml_config,
        job_config=job_config,
        data_persister=data_persister,
        seed=None,
    )

    assert job_runner

    ### Do one iteration of the job_controller

    for i, (temp_job_id, temp_job_param) in enumerate(job_controller.iterate()):

        if i > 0:
            break  # I need to complete a loop, so I need the break to be AFTER second yield

        job_id = temp_job_id
        job_param = temp_job_param
        assert isinstance(job_id, str)
        assert isinstance(job_param, dict)

    ### retriveve job by worker
    for worker_job_id, worker_job_param in job_runner.iterate():
        break

    assert isinstance(worker_job_id, str)
    assert isinstance(worker_job_param, dict)

    assert worker_job_id == job_id
    assert worker_job_param == job_param
