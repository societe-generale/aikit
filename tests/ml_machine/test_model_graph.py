# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:41:26 2018

@author: Lionel Massoulard
"""
import pytest

from collections import OrderedDict
import networkx as nx

from aikit.ml_machine.ml_machine_registration import MODEL_REGISTER

from aikit.ml_machine.model_graph import (
    is_composition_model,
    _must_include_selector,
    create_graphical_representation,
    convert_graph_to_code,
    assert_model_graph_structure,
)
from aikit.ml_machine.model_graph import _create_name_mapping, _find_first_composition_node

from aikit.enums import TypeOfVariables


def test_is_composition_model():
    assert is_composition_model(("TargetTransformer", "BoxCoxTargetTransformer"))
    assert not is_composition_model(("CategoryEncoder", "NumericalEncoder"))

    for model in MODEL_REGISTER.informations.keys():
        if model[0] == "TargetTransformer":
            assert is_composition_model(model)
        else:
            assert not is_composition_model(model)


def test__must_include_selector():
    assert not _must_include_selector(("CategoryEncoder", "NumericalEncoder"))
    assert _must_include_selector(("TextPreprocessing", "TextDigitAnonymizer"))


def test_create_graphical_representation():

    steps = OrderedDict(
        [
            (("TextPreprocessing", ("TextPreprocessing", "CountVectorizerWrapper")), TypeOfVariables.TEXT),
            (("DimensionReduction", ("DimensionReduction", "TruncatedSVDWrapper")), TypeOfVariables.TEXT),
            (("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")), TypeOfVariables.CAT),
            (("CategoryImputer", ("CategoryImputer", "CatImputer")), TypeOfVariables.CAT),
            (("MissingValueImputer", ("MissingValueImputer", "NumImputer")), TypeOfVariables.NUM),
            (("FeatureExtraction", ("FeatureExtraction", "PolynomialExtractor")), TypeOfVariables.NUM),
            (("Scaling", ("Scaling", "StandardScaler")), (TypeOfVariables.CAT, TypeOfVariables.NUM)),
            (
                ("FeatureSelection", ("FeatureSelection", "FeaturesSelectorClassifier")),
                (TypeOfVariables.CAT, TypeOfVariables.NUM, TypeOfVariables.TEXT),
            ),
            (
                ("Model", ("Model", "LightGBMClassifier")),
                (TypeOfVariables.CAT, TypeOfVariables.NUM, TypeOfVariables.TEXT),
            ),
        ]
    )

    #    columns = {"TEXT":["txt1","txt2"],
    #               "CAT":["cat1","cat2","cat3"],
    #               "NUM":["num1","num2"]}

    #    params = {n:("param_%s" % n) for n,t in steps.items()}

    G, new_steps = create_graphical_representation(steps)

    assert isinstance(G, nx.DiGraph)
    assert len(new_steps) == 0

    # graphviz_modelgraph(G)

    expected_edges = [
        (
            ("TextPreprocessing", ("TextPreprocessing", "CountVectorizerWrapper")),
            ("DimensionReduction", ("DimensionReduction", "TruncatedSVDWrapper")),
        ),
        (
            ("DimensionReduction", ("DimensionReduction", "TruncatedSVDWrapper")),
            ("FeatureSelection", ("FeatureSelection", "FeaturesSelectorClassifier")),
        ),
        (
            ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")),
            ("CategoryImputer", ("CategoryImputer", "CatImputer")),
        ),
        (("CategoryImputer", ("CategoryImputer", "CatImputer")), ("Scaling", ("Scaling", "StandardScaler"))),
        (
            ("MissingValueImputer", ("MissingValueImputer", "NumImputer")),
            ("FeatureExtraction", ("FeatureExtraction", "PolynomialExtractor")),
        ),
        (
            ("FeatureExtraction", ("FeatureExtraction", "PolynomialExtractor")),
            ("Scaling", ("Scaling", "StandardScaler")),
        ),
        (
            ("Scaling", ("Scaling", "StandardScaler")),
            ("FeatureSelection", ("FeatureSelection", "FeaturesSelectorClassifier")),
        ),
        (
            ("FeatureSelection", ("FeatureSelection", "FeaturesSelectorClassifier")),
            ("Model", ("Model", "LightGBMClassifier")),
        ),
    ]

    expected_nodes = [
        ("TextPreprocessing", ("TextPreprocessing", "CountVectorizerWrapper")),
        ("DimensionReduction", ("DimensionReduction", "TruncatedSVDWrapper")),
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")),
        ("CategoryImputer", ("CategoryImputer", "CatImputer")),
        ("MissingValueImputer", ("MissingValueImputer", "NumImputer")),
        ("FeatureExtraction", ("FeatureExtraction", "PolynomialExtractor")),
        ("Scaling", ("Scaling", "StandardScaler")),
        ("FeatureSelection", ("FeatureSelection", "FeaturesSelectorClassifier")),
        ("Model", ("Model", "LightGBMClassifier")),
    ]

    assert set(expected_edges) == set(G.edges)
    assert set(expected_nodes) == set(G.nodes)

    params = {}
    for n, _ in steps.items():
        params[n] = {"__%s_%s__" % n[1]: "param"}

    res1 = convert_graph_to_code(G, params)
    assert isinstance(res1, tuple)
    assert res1[0] == "GraphPipeline"
    assert isinstance(res1[1], dict)
    assert "edges" in res1[1]
    assert "models" in res1[1]


def test__create_name_mapping():

    ## Case 1 : no duplicate model ##
    nodes = [
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")),
        ("TargetTransformer", ("TargetTransformer", "BoxCoxTargetTransformer")),
        ("DimensionReduction", ("DimensionReduction", "KMeansTransformer")),
        ("Model", ("Model", "RandomForestClassifier")),
    ]

    mapping = _create_name_mapping(nodes)
    assert isinstance(mapping, dict)
    assert len(mapping.values()) == len(set(mapping.values()))

    for node in nodes:
        assert node in mapping
        assert mapping[node] == node[1][1]

    ## Case 2 : one duplicate model ##
    nodes = [
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")),
        ("TargetTransformer", ("TargetTransformer", "BoxCoxTargetTransformer")),
        ("DimensionReductionText", ("DimensionReductionText", "TruncatedSVDWrapper")),
        ("DimensionReduction", ("DimensionReduction", "TruncatedSVDWrapper")),
        ("Model", ("Model", "RandomForestClassifier")),
    ]

    mapping = _create_name_mapping(nodes)

    assert isinstance(mapping, dict)
    assert len(mapping.values()) == len(set(mapping.values()))

    for node in nodes:
        assert node in mapping
        if node[1][1] == "TruncatedSVDWrapper":
            assert mapping[node] == node[1][0] + "_" + node[1][1]
        else:
            assert mapping[node] == node[1][1]

    nodes = [
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")),
        ("Stacking", ("Stacking1", "OutSampler")),
        ("Stacking", ("Stacking2", "OutSampler")),
        ("Model", ("Model", "RandomForestClassifier")),
        ("Model", ("Model", "LogisticRegression")),
        ("Blender", ("Blender", "LogisticRegression")),
    ]

    mapping = _create_name_mapping(nodes)
    assert isinstance(mapping, dict)
    assert len(mapping.values()) == len(set(mapping.values()))

    nodes = [
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")),
        ("Stacking", ("Stacking", "OutSampler")),
        ("Stacking", ("Stacking", "OutSampler")),
        ("Model", ("Model", "RandomForestClassifier")),
        ("Model", ("Model", "LogisticRegression")),
        ("Blender", ("Blender", "LogisticRegression")),
    ]

    with pytest.raises(ValueError):
        mapping = _create_name_mapping(nodes)
        # can't work : duplicate nodes


def test_convert_graph_to_code():

    ###################################
    ### ** Only one Simple Model ** ###
    ###################################

    Graph = nx.DiGraph()
    Graph.add_node(("Model", ("Model", "LogisticRegression")))

    assert _find_first_composition_node(Graph) is None

    ## a) no params
    all_models_params = {("Model", ("Model", "LogisticRegression")): {}}
    model_json_code = convert_graph_to_code(Graph, all_models_params)

    assert model_json_code == ("LogisticRegression", {})

    ## b) params
    all_models_params = {("Model", ("Model", "LogisticRegression")): {"C": 10}}
    model_json_code = convert_graph_to_code(Graph, all_models_params)

    assert model_json_code == ("LogisticRegression", {"C": 10})

    #####################
    ### ** 2 steps ** ###
    #####################
    Graph = nx.DiGraph()
    Graph.add_edge(
        ("DimensionReduction", ("DimensionReduction", "KMeansTransformer")),
        ("Model", ("Model", "RandomForestClassifier")),
    )

    assert _find_first_composition_node(Graph) is None

    ## a) no params
    all_models_params = {
        ("DimensionReduction", ("DimensionReduction", "KMeansTransformer")): {},
        ("Model", ("Model", "RandomForestClassifier")): {},
    }

    model_json_code = convert_graph_to_code(Graph, all_models_params)

    assert model_json_code == (
        "GraphPipeline",
        {
            "edges": [("KMeansTransformer", "RandomForestClassifier")],
            "models": {
                "KMeansTransformer": ("KMeansTransformer", {}),
                "RandomForestClassifier": ("RandomForestClassifier", {}),
            },
        },
    )

    ## b) no params
    all_models_params = {
        ("DimensionReduction", ("DimensionReduction", "KMeansTransformer")): {"n_clusters": 5},
        ("Model", ("Model", "RandomForestClassifier")): {"n_estimators": 100},
    }

    model_json_code = convert_graph_to_code(Graph, all_models_params)

    assert model_json_code == (
        "GraphPipeline",
        {
            "edges": [("KMeansTransformer", "RandomForestClassifier")],
            "models": {
                "KMeansTransformer": ("KMeansTransformer", {"n_clusters": 5}),
                "RandomForestClassifier": ("RandomForestClassifier", {"n_estimators": 100}),
            },
        },
    )

    ################################
    ### ** 1 composition step ** ###
    ################################

    ## a) no params
    Graph = nx.DiGraph()
    Graph.add_edge(
        ("TargetTransformer", ("TargetTransformer", "BoxCoxTargetTransformer")),
        ("Model", ("Model", "RandomForestClassifier")),
    )

    assert _find_first_composition_node(Graph) == (
        "TargetTransformer",
        ("TargetTransformer", "BoxCoxTargetTransformer"),
    )

    all_models_params = {
        ("TargetTransformer", ("TargetTransformer", "BoxCoxTargetTransformer")): {},
        ("Model", ("Model", "RandomForestClassifier")): {},
    }

    model_json_code = convert_graph_to_code(Graph, all_models_params)
    expected_json_code = ("BoxCoxTargetTransformer", ("RandomForestClassifier", {}), {})

    assert model_json_code == expected_json_code

    ## b) params

    Graph = nx.DiGraph()
    Graph.add_edge(
        ("TargetTransformer", ("TargetTransformer", "BoxCoxTargetTransformer")),
        ("Model", ("Model", "RandomForestClassifier")),
    )

    assert _find_first_composition_node(Graph) == (
        "TargetTransformer",
        ("TargetTransformer", "BoxCoxTargetTransformer"),
    )

    all_models_params = {
        ("TargetTransformer", ("TargetTransformer", "BoxCoxTargetTransformer")): {"ll": 10},
        ("Model", ("Model", "RandomForestClassifier")): {"n_estimators": 10},
    }

    model_json_code = convert_graph_to_code(Graph, all_models_params)
    expected_json_code = ("BoxCoxTargetTransformer", ("RandomForestClassifier", {"n_estimators": 10}), {"ll": 10})

    assert model_json_code == expected_json_code

    ##########################################
    ## ** 1 composition above a pipeline ** ##
    ##########################################

    ## a) no param
    Graph = nx.DiGraph()
    Graph.add_edge(
        ("TargetTransformer", ("TargetTransformer", "BoxCoxTargetTransformer")),
        ("DimensionReduction", ("DimensionReduction", "KMeansTransformer")),
    )

    Graph.add_edge(
        ("DimensionReduction", ("DimensionReduction", "KMeansTransformer")),
        ("Model", ("Model", "RandomForestClassifier")),
    )

    assert _find_first_composition_node(Graph) == (
        "TargetTransformer",
        ("TargetTransformer", "BoxCoxTargetTransformer"),
    )

    all_models_params = {
        ("TargetTransformer", ("TargetTransformer", "BoxCoxTargetTransformer")): {},
        ("DimensionReduction", ("DimensionReduction", "KMeansTransformer")): {},
        ("Model", ("Model", "RandomForestClassifier")): {},
    }

    model_json_code = convert_graph_to_code(Graph, all_models_params)

    expected_json_code = (
        "BoxCoxTargetTransformer",
        (
            "GraphPipeline",
            {
                "edges": [("KMeansTransformer", "RandomForestClassifier")],
                "models": {
                    "KMeansTransformer": ("KMeansTransformer", {}),
                    "RandomForestClassifier": ("RandomForestClassifier", {}),
                },
            },
        ),
        {},
    )
    assert model_json_code == expected_json_code

    ## b) params
    Graph = nx.DiGraph()
    Graph.add_edge(
        ("TargetTransformer", ("TargetTransformer", "BoxCoxTargetTransformer")),
        ("DimensionReduction", ("DimensionReduction", "KMeansTransformer")),
    )

    Graph.add_edge(
        ("DimensionReduction", ("DimensionReduction", "KMeansTransformer")),
        ("Model", ("Model", "RandomForestClassifier")),
    )

    all_models_params = {
        ("TargetTransformer", ("TargetTransformer", "BoxCoxTargetTransformer")): {"ll": 10},
        ("DimensionReduction", ("DimensionReduction", "KMeansTransformer")): {"n_clusters": 10},
        ("Model", ("Model", "RandomForestClassifier")): {"n_estimators": 10},
    }

    model_json_code = convert_graph_to_code(Graph, all_models_params)

    expected_json_code = (
        "BoxCoxTargetTransformer",
        (
            "GraphPipeline",
            {
                "edges": [("KMeansTransformer", "RandomForestClassifier")],
                "models": {
                    "KMeansTransformer": ("KMeansTransformer", {"n_clusters": 10}),
                    "RandomForestClassifier": ("RandomForestClassifier", {"n_estimators": 10}),
                },
            },
        ),
        {"ll": 10},
    )

    assert model_json_code == expected_json_code

    #########################################################
    ## ** 1 composition node in the middle of the Graph ** ##
    #########################################################
    Graph = nx.DiGraph()

    Graph.add_edge(
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")),
        ("TargetTransformer", ("TargetTransformer", "BoxCoxTargetTransformer")),
    )
    Graph.add_edge(
        ("TargetTransformer", ("TargetTransformer", "BoxCoxTargetTransformer")),
        ("DimensionReduction", ("DimensionReduction", "KMeansTransformer")),
    )

    Graph.add_edge(
        ("DimensionReduction", ("DimensionReduction", "KMeansTransformer")),
        ("Model", ("Model", "RandomForestClassifier")),
    )

    all_models_params = {
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")): {},
        ("TargetTransformer", ("TargetTransformer", "BoxCoxTargetTransformer")): {"ll": 10},
        ("DimensionReduction", ("DimensionReduction", "KMeansTransformer")): {"n_clusters": 10},
        ("Model", ("Model", "RandomForestClassifier")): {"n_estimators": 10},
    }

    model_json_code = convert_graph_to_code(Graph, all_models_params)

    expected_json_code = (
        "GraphPipeline",
        {
            "edges": [("NumericalEncoder", "BoxCoxTargetTransformer")],
            "models": {
                "BoxCoxTargetTransformer": (
                    "BoxCoxTargetTransformer",
                    (
                        "GraphPipeline",
                        {
                            "edges": [("KMeansTransformer", "RandomForestClassifier")],
                            "models": {
                                "KMeansTransformer": ("KMeansTransformer", {"n_clusters": 10}),
                                "RandomForestClassifier": ("RandomForestClassifier", {"n_estimators": 10}),
                            },
                        },
                    ),
                    {"ll": 10},
                ),
                "NumericalEncoder": ("NumericalEncoder", {}),
            },
        },
    )

    assert model_json_code == expected_json_code

    ###################################################
    ## ** 1 composition with several nodes bellow ** ##
    ###################################################
    Graph = nx.DiGraph()
    # TODO : essayer de faire un stacking avec plusieurs trucs en dessous
    Graph.add_edge(
        ("Stacking", ("Stacking", "StackingClassifierRegressor")), ("Model", ("Model", "RandomForestClassifier"))
    )
    Graph.add_edge(
        ("Stacking", ("Stacking", "StackingClassifierRegressor")), ("Model", ("Model", "LogisticRegression"))
    )

    assert _find_first_composition_node(Graph) == ("Stacking", ("Stacking", "StackingClassifierRegressor"))

    # Rmk : Blending specification is missing, BUT it is enough to test the function
    all_models_params = {
        ("Stacking", ("Stacking", "StackingClassifierRegressor")): {"cv": 10},
        ("Model", ("Model", "RandomForestClassifier")): {"n_estimators": 100},
        ("Model", ("Model", "LogisticRegression")): {"C": 10},
    }

    with pytest.raises(ValueError):
        model_json_code = convert_graph_to_code(Graph, all_models_params)
        # Unsuported for now : more than one terminal node

    model_json_code = convert_graph_to_code(Graph, all_models_params, _check_structure=False)

    expected_json_code1 = (
        "StackingClassifierRegressor",
        [("RandomForestClassifier", {"n_estimators": 100}), ("LogisticRegression", {"C": 10})],
        {"cv": 10},
    )
    #    expected_json_code2 = (
    #        "StackingClassifierRegressor",
    #        [("LogisticRegression", {"C": 10}), ("RandomForestClassifier", {"n_estimators": 100})],
    #        {"cv": 10},
    #    )
    assert expected_json_code1 == model_json_code  # or (expected_json_code2 == model_json_code)

    #######################################
    ## ** 2 nested compositions steps ** ##
    #######################################

    Graph = nx.DiGraph()
    Graph.add_edge(
        ("TargetTransformer", ("TargetTransformer", "BoxCoxTargetTransformer")),
        ("UnderOverSampler", ("UnderOverSampler", "TargetUnderSampler")),
    )

    Graph.add_edge(
        ("UnderOverSampler", ("UnderOverSampler", "TargetUnderSampler")), ("Model", ("Model", "RandomForestClassifier"))
    )

    all_models_params = {}
    all_models_params[("TargetTransformer", ("TargetTransformer", "BoxCoxTargetTransformer"))] = {"ll": 10}
    all_models_params[("Model", ("Model", "RandomForestClassifier"))] = {"n_estimators": 100}
    all_models_params[("UnderOverSampler", ("UnderOverSampler", "TargetUnderSampler"))] = {"target_ratio": "balanced"}

    assert _find_first_composition_node(Graph) == (
        "TargetTransformer",
        ("TargetTransformer", "BoxCoxTargetTransformer"),
    )

    assert _find_first_composition_node(
        Graph, composition_already_done={("TargetTransformer", ("TargetTransformer", "BoxCoxTargetTransformer"))}
    ) == ("UnderOverSampler", ("UnderOverSampler", "TargetUnderSampler"))

    model_json_code = convert_graph_to_code(Graph, all_models_params)

    expected_json_code = (
        "BoxCoxTargetTransformer",
        ("TargetUnderSampler", ("RandomForestClassifier", {"n_estimators": 100}), {"target_ratio": "balanced"}),
        {"ll": 10},
    )

    assert model_json_code == expected_json_code

    ###################################################
    ## ** 1 composition with several nodes bellow ** ##
    ###################################################
    Graph = nx.DiGraph()
    # TODO : essayer de faire un stacking avec plusieurs trucs en dessous

    ## 1) with one node above

    Graph.add_edge(
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")),
        ("Stacking", ("Stacking", "StackingClassifierRegressor")),
    )
    Graph.add_edge(
        ("Stacking", ("Stacking", "StackingClassifierRegressor")), ("Model", ("Model", "RandomForestClassifier"))
    )
    Graph.add_edge(
        ("Stacking", ("Stacking", "StackingClassifierRegressor")), ("Model", ("Model", "LogisticRegression"))
    )

    all_models_params = {
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")): {},
        ("Stacking", ("Stacking", "StackingClassifierRegressor")): {"cv": 10},
        ("Model", ("Model", "RandomForestClassifier")): {"n_estimators": 100},
        ("Model", ("Model", "LogisticRegression")): {"C": 10},
    }

    model_json_code = convert_graph_to_code(Graph, all_models_params, _check_structure=False)
    expected_json_code = (
        "GraphPipeline",
        {
            "edges": [("NumericalEncoder", "StackingClassifierRegressor")],
            "models": {
                "NumericalEncoder": ("NumericalEncoder", {}),
                "StackingClassifierRegressor": (
                    "StackingClassifierRegressor",
                    [("RandomForestClassifier", {"n_estimators": 100}), ("LogisticRegression", {"C": 10})],
                    {"cv": 10},
                ),
            },
        },
    )

    # Rmk : the Stacker is missing the blender, that I can't enter into the graph..
    assert expected_json_code == model_json_code

    ### With a node above, and a blender bellow
    Graph = nx.DiGraph()
    Graph.add_edge(
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")), ("Stacking", ("Stacking", "OutSampler"))
    )
    Graph.add_edge(("Stacking", ("Stacking", "OutSampler")), ("Model", ("Model", "RandomForestClassifier")))
    Graph.add_edge(("Stacking", ("Stacking", "OutSampler")), ("Model", ("Model", "LogisticRegression")))

    Graph.add_edge(("Model", ("Model", "LogisticRegression")), ("Blender", ("Blender", "LogisticRegression")))

    Graph.add_edge(("Model", ("Model", "RandomForestClassifier")), ("Blender", ("Blender", "LogisticRegression")))

    all_models_params = {
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")): {},
        ("Stacking", ("Stacking", "OutSampler")): {"cv": 10},
        ("Model", ("Model", "RandomForestClassifier")): {"n_estimators": 100},
        ("Model", ("Model", "LogisticRegression")): {"C": 10},
        ("Blender", ("Blender", "LogisticRegression")): {"C": 100},
    }

    model_json_code = convert_graph_to_code(Graph, all_models_params)

    expected_json = (
        "GraphPipeline",
        {
            "edges": [("NumericalEncoder", "OutSampler", "Blender_LogisticRegression")],
            "models": {
                "Blender_LogisticRegression": ("LogisticRegression", {"C": 100}),
                "NumericalEncoder": ("NumericalEncoder", {}),
                "OutSampler": (
                    "OutSampler",
                    [("RandomForestClassifier", {"n_estimators": 100}), ("LogisticRegression", {"C": 10})],
                    {"cv": 10},
                ),
            },
        },
    )

    assert expected_json == model_json_code

    ### With encoder feature going back into the Blender
    Graph = nx.DiGraph()
    Graph.add_edge(
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")), ("Stacking", ("Stacking", "OutSampler"))
    )
    Graph.add_edge(("Stacking", ("Stacking", "OutSampler")), ("Model", ("Model", "RandomForestClassifier")))
    Graph.add_edge(("Stacking", ("Stacking", "OutSampler")), ("Model", ("Model", "LogisticRegression")))

    Graph.add_edge(("Model", ("Model", "LogisticRegression")), ("Blender", ("Blender", "LogisticRegression")))

    Graph.add_edge(("Model", ("Model", "RandomForestClassifier")), ("Blender", ("Blender", "LogisticRegression")))

    Graph.add_edge(
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")), ("Blender", ("Blender", "LogisticRegression"))
    )

    all_models_params = {
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")): {},
        ("Stacking", ("Stacking", "OutSampler")): {"cv": 10},
        ("Model", ("Model", "RandomForestClassifier")): {"n_estimators": 100},
        ("Model", ("Model", "LogisticRegression")): {"C": 10},
        ("Blender", ("Blender", "LogisticRegression")): {"C": 100},
    }

    model_json_code = convert_graph_to_code(Graph, all_models_params)

    expected_json = (
        "GraphPipeline",
        {
            "edges": [
                ("NumericalEncoder", "Blender_LogisticRegression"),
                ("NumericalEncoder", "OutSampler", "Blender_LogisticRegression"),
            ],
            "models": {
                "Blender_LogisticRegression": ("LogisticRegression", {"C": 100}),
                "NumericalEncoder": ("NumericalEncoder", {}),
                "OutSampler": (
                    "OutSampler",
                    [("RandomForestClassifier", {"n_estimators": 100}), ("LogisticRegression", {"C": 10})],
                    {"cv": 10},
                ),
            },
        },
    )

    assert expected_json == model_json_code

    # Same thing but with 2 OutSampler (one per model)
    Graph = nx.DiGraph()
    Graph.add_edge(
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")), ("Stacking", ("Stacking1", "OutSampler"))
    )
    Graph.add_edge(
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")), ("Stacking", ("Stacking2", "OutSampler"))
    )
    Graph.add_edge(("Stacking", ("Stacking1", "OutSampler")), ("Model", ("Model", "RandomForestClassifier")))
    Graph.add_edge(("Stacking", ("Stacking2", "OutSampler")), ("Model", ("Model", "LogisticRegression")))

    Graph.add_edge(("Model", ("Model", "LogisticRegression")), ("Blender", ("Blender", "LogisticRegression")))

    Graph.add_edge(("Model", ("Model", "RandomForestClassifier")), ("Blender", ("Blender", "LogisticRegression")))

    Graph.add_edge(
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")), ("Blender", ("Blender", "LogisticRegression"))
    )

    all_models_params = {
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")): {},
        ("Stacking", ("Stacking1", "OutSampler")): {"cv": 10},
        ("Stacking", ("Stacking2", "OutSampler")): {"cv": 10},
        ("Model", ("Model", "RandomForestClassifier")): {"n_estimators": 100},
        ("Model", ("Model", "LogisticRegression")): {"C": 10},
        ("Blender", ("Blender", "LogisticRegression")): {"C": 100},
    }

    model_json_code = convert_graph_to_code(Graph, all_models_params)

    expected_json = (
        "GraphPipeline",
        {
            "edges": [
                ("NumericalEncoder", "Blender_LogisticRegression"),
                ("NumericalEncoder", "Stacking1_OutSampler", "Blender_LogisticRegression"),
                ("NumericalEncoder", "Stacking2_OutSampler", "Blender_LogisticRegression"),
            ],
            "models": {
                "Blender_LogisticRegression": ("LogisticRegression", {"C": 100}),
                "NumericalEncoder": ("NumericalEncoder", {}),
                "Stacking1_OutSampler": ("OutSampler", ("RandomForestClassifier", {"n_estimators": 100}), {"cv": 10}),
                "Stacking2_OutSampler": ("OutSampler", ("LogisticRegression", {"C": 10}), {"cv": 10}),
            },
        },
    )

    assert expected_json == model_json_code

    ### Multi output ###
    Graph = nx.DiGraph()
    Graph.add_node(("Model", ("Model", "LogisticRegression")))
    Graph.add_node(("Model", ("Model", "RandomForestClassifier")))

    all_models_params = {
        ("Model", ("Model", "LogisticRegression")): {"C": 10},
        ("Model", ("Model", "RandomForestClassifier")): {"n_estimators": 100},
    }

    assert _find_first_composition_node(Graph) is None

    model_json_code = convert_graph_to_code(Graph, all_models_params, _check_structure=False)
    expected_json = (
        "GraphPipeline",
        {
            "edges": [("LogisticRegression",), ("RandomForestClassifier",)],
            "models": {
                "LogisticRegression": ("LogisticRegression", {"C": 10}),
                "RandomForestClassifier": ("RandomForestClassifier", {"n_estimators": 100}),
            },
        },
    )

    assert expected_json == model_json_code

    ### Impossible graph ###
    Graph = nx.DiGraph()
    Graph.add_edge(
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")), ("Stacking", ("Stacking", "OutSampler"))
    )

    Graph.add_edge(("Stacking", ("Stacking", "OutSampler")), ("Model", ("Model", "RandomForestClassifier")))

    Graph.add_edge(("Stacking", ("Stacking", "OutSampler")), ("Model", ("Model", "LogisticRegression")))

    Graph.add_edge(("Stacking", ("Stacking", "OutSampler")), ("Model", ("Model", "ExtraTreesClassifier")))
    # This edge make it impossible : it comes from the composition node ...
    # but doesn't have the same child as the other

    Graph.add_edge(("Model", ("Model", "LogisticRegression")), ("Blender", ("Blender", "LogisticRegression")))

    Graph.add_edge(("Model", ("Model", "RandomForestClassifier")), ("Blender", ("Blender", "LogisticRegression")))

    #    graphviz_modelgraph(Graph)

    all_models_params = {
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")): {},
        ("Stacking", ("Stacking", "OutSampler")): {"cv": 10},
        ("Model", ("Model", "RandomForestClassifier")): {"n_estimators": 100},
        ("Model", ("Model", "ExtraTreesClassifier")): {"n_estimators": 200},
        ("Model", ("Model", "LogisticRegression")): {"C": 10},
        ("Blender", ("Blender", "LogisticRegression")): {"C": 100},
    }

    with pytest.raises(ValueError):
        model_json_code = convert_graph_to_code(Graph, all_models_params, _check_structure=False)


def test_assert_model_graph_structure():
    ## Impossible Graph :
    Graph = nx.DiGraph()
    Graph.add_edge(
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")), ("Stacking", ("Stacking", "OutSampler"))
    )

    Graph.add_edge(("Stacking", ("Stacking", "OutSampler")), ("Model", ("Model", "RandomForestClassifier")))

    Graph.add_edge(
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")), ("Model", ("Model", "RandomForestClassifier"))
    )
    # This edge makes it impossible : RF can't have two parents, if one is a compostion

    with pytest.raises(ValueError):
        assert_model_graph_structure(Graph)

    ## Composition without a child
    Graph = nx.DiGraph()
    Graph.add_edge(
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")), ("Stacking", ("Stacking", "OutSampler"))
    )

    with pytest.raises(ValueError):
        assert_model_graph_structure(Graph)

    # Cycle
    Graph = nx.DiGraph()
    Graph.add_edge(
        ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder")), ("Model", ("Model", "RandomForestClassifier"))
    )
    Graph.add_edge(
        ("Model", ("Model", "RandomForestClassifier")), ("CategoryEncoder", ("CategoryEncoder", "NumericalEncoder"))
    )
    with pytest.raises(ValueError):
        assert_model_graph_structure(Graph)
