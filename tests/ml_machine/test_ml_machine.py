# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:20:02 2019

@author: lmassoul032513
"""
import pytest

import pandas as pd
import numpy as np

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
    
    _create_all_combinations,
    random_list_generator
)
from aikit.ml_machine.model_graph import convert_graph_to_code
from aikit.model_definition import sklearn_model_from_param

from aikit.ml_machine.ml_machine_guider import AutoMlModelGuider
from aikit.ml_machine.data_persister import FolderDataPersister


def loader(num_only=False):
    if num_only:
        np.random.seed(123)
        dfX = pd.DataFrame(np.random.randn(100,10), columns=["COL_%d" % d for d in range(10)])
        y = 1*(np.random.randn(100)>0)
        return dfX, y
    else:
        dfX, y, _, _, _ = load_dataset(DatasetEnum.titanic)
        return dfX, y

def get_automl_config(num_only):
    dfX, y = loader(num_only)
    auto_ml_config = AutoMlConfig(dfX, y)
    auto_ml_config.guess_everything()

    return dfX, y, auto_ml_config

def test_AutoMlConfig_raise_if_wrong_nb_oberservations():
    dfX = pd.DataFrame({"a":[0,1,2,3,4,5],"b":[0,10,20,30,40,50]})
    y   = np.array([0,0,0,1,1,1])

    auto_ml_config = AutoMlConfig(dfX, y[0:3])
    with pytest.raises(ValueError):
        auto_ml_config.guess_everything() #raise because y doesn't have the correct number of observations


def test_AutoMlConfig_raise_multioutput():
    dfX = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [0, 10, 20, 30, 40, 50]})
    y = np.array([0, 0, 0, 1, 1, 1])
    y2d = np.concatenate((y[:, np.newaxis], y[:, np.newaxis]), axis=1)

    auto_ml_config = AutoMlConfig(dfX, y2d)
    with pytest.raises(ValueError):
        auto_ml_config.guess_everything()  # raise because y has 2 dimensions


@pytest.mark.parametrize('num_only', [True,False])
def test_AutoMlConfig(num_only):

    dfX, y, auto_ml_config = get_automl_config(num_only)

    assert auto_ml_config.type_of_problem == TypeOfProblem.CLASSIFICATION
    assert auto_ml_config.columns_informations is not None
    
    ###############################
    ###  Tests on needed steps  ###
    ###############################
    def _check_steps(auto_ml_config):
        assert hasattr(auto_ml_config, "needed_steps")
        assert isinstance(auto_ml_config.needed_steps, list)
        for step in auto_ml_config.needed_steps:
            assert isinstance(step, dict)
            assert set(step.keys()) == {"optional","step"}
            assert isinstance(step["optional"], bool)
            assert isinstance(step["step"], str)

    _check_steps(auto_ml_config)
    assert "Model" in [step["step"] for step in auto_ml_config.needed_steps]
    assert "Scaling" in [step["step"] for step in auto_ml_config.needed_steps]
    
    # Try assigning to needed steps
    auto_ml_config.needed_steps = [s for s in auto_ml_config.needed_steps if s["step"] != "Scaling"]
    
    _check_steps(auto_ml_config)
    assert "Model" in [step["step"] for step in auto_ml_config.needed_steps]
    assert "Scaling" not in [step["step"] for step in auto_ml_config.needed_steps]
    
    with pytest.raises(TypeError):
        auto_ml_config.needed_steps = "this shouldn't be accepted has steps"
        
    _check_steps(auto_ml_config)


    #################################
    ###  Tests on models to keep  ###
    #################################
    def _check_models(auto_ml_config):
        assert hasattr(auto_ml_config, "models_to_keep")
        assert isinstance(auto_ml_config.models_to_keep, list)
        for model in auto_ml_config.models_to_keep:
            assert isinstance(model, tuple)
            assert len(model) == 2
            assert isinstance(model[0], str)
            assert isinstance(model[1], str)
            
    _check_models(auto_ml_config)
    
    assert ('Model', 'LogisticRegression') in auto_ml_config.models_to_keep
    assert ('Model', 'RandomForestClassifier') in auto_ml_config.models_to_keep
    assert ('Model', 'ExtraTreesClassifier') in auto_ml_config.models_to_keep
    # try assignation
    auto_ml_config.models_to_keep = [m for m in auto_ml_config.models_to_keep if m[1] != "LogisticRegression"]
    
    with pytest.raises(TypeError):
        auto_ml_config.models_to_keep = "this shouldn't be accepted has models_to_keep"
        
    
    _check_models(auto_ml_config)
    assert ('Model', 'LogisticRegression') not in auto_ml_config.models_to_keep
    assert ('Model', 'RandomForestClassifier') in auto_ml_config.models_to_keep
    assert ('Model', 'ExtraTreesClassifier') in auto_ml_config.models_to_keep
 
    auto_ml_config.filter_models(Model="ExtraTreesClassifier")
    
    _check_models(auto_ml_config)
    assert ('Model', 'LogisticRegression') not in auto_ml_config.models_to_keep
    assert ('Model', 'RandomForestClassifier') not in auto_ml_config.models_to_keep
    assert ('Model', 'ExtraTreesClassifier') in auto_ml_config.models_to_keep
    
@pytest.mark.parametrize('num_only', [True,False])
def test_JobConfig(num_only):

    dfX, y, auto_ml_config = get_automl_config(num_only)

    job_config = JobConfig()
    job_config.guess_cv(auto_ml_config)

    assert job_config.cv is not None
    assert hasattr(job_config.cv, "split")
    

    job_config.guess_scoring(auto_ml_config)
    assert isinstance(job_config.scoring, list)
    
    assert hasattr(job_config, "allow_approx_cv")
    assert hasattr(job_config, "start_with_default")
    assert hasattr(job_config, "do_blocks_search")
    
    assert isinstance(job_config.allow_approx_cv, bool)
    assert isinstance(job_config.start_with_default, bool)
    assert isinstance(job_config.do_blocks_search, bool)   
    
    with pytest.raises(ValueError):
        job_config.cv = "this is not a cv"
   
def test_JobConfig_additional_scoring_function():
    job_config = JobConfig()

    assert job_config.additional_scoring_function is None

    def f(x):
        return x+1

    job_config.additional_scoring_function = f
    assert job_config.additional_scoring_function is not None
    assert job_config.additional_scoring_function(1) == 2
    
    with pytest.raises(TypeError):
        job_config.additional_scoring_function = 10 # no a function
        
    
    
@pytest.mark.parametrize('num_only', [True,False])
@pytest.mark.parametrize("type_of_iterator", ["default", "block_search","block_search_random"])
def test_RandomModelGenerator_iterator(type_of_iterator, num_only):

    dfX, y, auto_ml_config = get_automl_config(num_only)

    random_model_generator = RandomModelGenerator(auto_ml_config=auto_ml_config, random_state=123)
    
    if type_of_iterator == "default":
        iterator = random_model_generator.iterator_default_models()

    elif type_of_iterator == "block_search":
        iterator = random_model_generator.iterate_block_search(random_order=False)
        
    elif type_of_iterator == "block_search_random":
        iterator = random_model_generator.iterate_block_search(random_order=True)
        
    assert hasattr(iterator,"__iter__")

    # verif iterator
    for model in iterator:

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

#def test_RandomModelGenerator_block_search():
#    dfX, y, auto_ml_config = get_automl_config()
#
#    random_model_generator = RandomModelGenerator(auto_ml_config=auto_ml_config, random_state=123)
#
#    # verif iterator
#    for model in random_model_generator.iterate_block_search_models():
#
#        assert isinstance(model, tuple)
#        assert len(model) == 3
#        Graph, all_models_params, block_to_use = model
#
#        assert hasattr(Graph, "edges")
#        assert hasattr(Graph, "nodes")
#
#        assert isinstance(all_models_params, dict)
#        for node in Graph.node:
#            assert node in all_models_params
#
#        assert isinstance(block_to_use, (tuple, list))
#        for b in block_to_use:
#            assert b in TypeOfVariables.alls
#
#        result = convert_graph_to_code(Graph, all_models_params, also_returns_mapping=True)
#        assert isinstance(result, dict)
#        assert "name_mapping" in result
#        assert "json_code" in result
#
#        model = sklearn_model_from_param(result["json_code"])
#        assert hasattr(model, "fit")


def test_random_list_generator():
    elements = ["a","b","c","d","e","f","g","h","i","j"]
    
    for i in range(2):
        if i == 0:
            probas = [1/min(i+1,10+1-i) for i in range(len(elements))]
        else:
            probas = None
            
        gen = random_list_generator(elements,probas, random_state=123)
    
        assert hasattr(gen,"__iter__")
    
        elements_random_order = list(gen)
        assert len(elements_random_order) == len(elements)
        assert set(elements_random_order) == set(elements)
        
        elements_random_order2 = list(random_list_generator(elements,probas=probas, random_state=123))
        elements_random_order3 = list(random_list_generator(elements,probas=probas, random_state=456))
        elements_random_order4 = list(random_list_generator(elements,probas=probas, random_state=check_random_state(123)))

        assert len(elements_random_order2) == len(elements)
        assert set(elements_random_order2) == set(elements)
        
        assert len(elements_random_order3) == len(elements)
        assert set(elements_random_order3) == set(elements)
        
        assert elements_random_order2 == elements_random_order
        assert elements_random_order3 != elements_random_order
        assert elements_random_order4 == elements_random_order
    
    with pytest.raises(ValueError):
        list(random_list_generator(elements,probas=[0.1], random_state=123)) # error : probas not the right length
        
    with pytest.raises(ValueError):
        list(random_list_generator(elements,probas=[0] * len(elements), random_state=123)) # error : probas 0
        
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

@pytest.mark.parametrize('num_only', [True,False])
def test_RandomModelGenerator_random(num_only):

    dfX, y, auto_ml_config = get_automl_config(num_only)

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
    if not num_only:
        assert not _all_same(all_blocks1) # only one block

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


def test__create_all_combinations():
    
    def _check_all_list_of_blocks(all_list_of_blocks,all_blocks_to_use):
        assert isinstance(all_list_of_blocks, list)
        for blocks_to_use in all_list_of_blocks:
            assert isinstance(blocks_to_use, tuple)
            assert 1 <= len(blocks_to_use) <= len(all_blocks_to_use)
            for b in blocks_to_use:
                assert b in all_blocks_to_use
                
            assert len(set(blocks_to_use)) == len(blocks_to_use)
        assert len(set(all_list_of_blocks)) == len(all_list_of_blocks) # no duplicate
    
    all_blocks_to_use = ("CAT","NUM","TEXT")    
    all_list_of_blocks = _create_all_combinations(all_blocks_to_use, 1,1)    
    
    _check_all_list_of_blocks(all_list_of_blocks, all_blocks_to_use)
    
    
    all_blocks_to_use = ("a","b","c","d")
    all_list_of_blocks = _create_all_combinations(all_blocks_to_use, 2,2)  
    _check_all_list_of_blocks(all_list_of_blocks, all_blocks_to_use)


    with pytest.raises(ValueError):
        all_list_of_blocks = _create_all_combinations(all_blocks_to_use, 0,2)   # 0 : not possible

    with pytest.raises(ValueError):
        all_list_of_blocks = _create_all_combinations(all_blocks_to_use, 2,0)   # 0 : not possible
        
    with pytest.raises(ValueError):
        all_list_of_blocks = _create_all_combinations(["a","a"], 2,2)          # duplicate entry
        
    
    assert _create_all_combinations(("a",), 1,1) == []
    assert set(_create_all_combinations(("a","b"),1,1)) == set([("a",),("b",)])
    assert set(_create_all_combinations(("a","b","c"),1,1)) == set([("a",),("b",),("c",),("a","b"),("a","c"),("b","c")])
    

# In[] :
@pytest.mark.parametrize('num_only', [True,False])
def test_create_everything_sequentially(num_only, tmpdir):

    # DataPersister
    data_persister = FolderDataPersister(base_folder=tmpdir)

    # Data
    dfX, y = loader(num_only)

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
