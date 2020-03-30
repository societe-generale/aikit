# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 10:25:16 2018

@author: Lionel Massoulard
"""


import pytest

import graphviz

import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.base import is_classifier, is_regressor
from sklearn.base import clone

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from tests.helpers.testing_help import get_sample_df, get_random_strings

from aikit.tools.helper_functions import diff
from aikit.transformers.base import PassThrough, ColumnsSelector
from aikit.transformers.model_wrapper import DebugPassThrough

from aikit.pipeline import GraphPipeline, make_pipeline
from aikit.enums import DataTypes

from aikit.transformers.text import CountVectorizerWrapper

from aikit.transformers.block_selector import BlockSelector, BlockManager, TransformToBlockManager

from aikit.cross_validation import is_clusterer


# In[]
dfX = pd.DataFrame(
    {
        "text1": ["aa bb", "bb bb cc", "dd aa cc", "ee"],
        "text2": ["AAA ZZZ", "BBB EEE", "DDD TTT", "AAA BBB CCC"],
        "num1": [0, 1, 2, 3],
        "num2": [1.1, 1.5, -2, -3.5],
        "num3": [-1, 1, 25, 4],
        "cat1": ["A", "B", "A", "D"],
        "cat2": ["toto", "tata", "truc", "toto"],
    }
)

X = dfX.loc[:, ["num1", "num2", "num3"]]
y = np.array([10, 20, -10, -20])
yc = 1 * (y > 0)


# In[]
def test_make_pipeline():
    s = StandardScaler()
    d = DecisionTreeClassifier()
    gpipeline = make_pipeline(s, d)
    assert isinstance(gpipeline, GraphPipeline)

    assert set(gpipeline.models.keys()) == set(["standardscaler", "decisiontreeclassifier"])
    assert gpipeline.edges == [("standardscaler", "decisiontreeclassifier")]

    assert gpipeline.models["standardscaler"] is s
    assert gpipeline.models["decisiontreeclassifier"] is d


def test_gpipeline_regression():
    gpipeline = GraphPipeline({"PT": PassThrough(), "Ridge": Ridge()}, [("PT", "Ridge")])

    X = dfX.loc[:, ["num1", "num2", "num3"]]

    gpipeline.fit(X, y)
    yhat = gpipeline.predict(X)
    yhat2 = gpipeline.models["Ridge"].predict(X)

    assert yhat.shape == y.shape
    assert (yhat == yhat2).all()

    with pytest.raises(AttributeError):
        gpipeline.predict_proba(X)

    with pytest.raises(AttributeError):
        gpipeline.predict_log_proba(X)

    assert gpipeline.get_feature_names_at_node("PT") == list(X.columns)
    assert gpipeline.get_input_features_at_node("PT") == list(X.columns)
    assert gpipeline.get_input_features_at_node("Ridge") == list(X.columns)

    with pytest.raises(ValueError):
        assert gpipeline.get_feature_names_at_node("DONTEXIST")


def test_gpipeline_raise_not_fitted():
    gpipeline = GraphPipeline({"PT": PassThrough(), "Ridge": Ridge()}, [("PT", "Ridge")])

    with pytest.raises(NotFittedError):
        gpipeline.predict(X)


def test_gpipeline_clone():
    gpipeline = GraphPipeline({"PT": PassThrough(), "Ridge": Ridge()}, [("PT", "Ridge")])
    gpipeline.fit(X, y)

    cloned_gpipeline = clone(gpipeline)

    with pytest.raises(NotFittedError):
        cloned_gpipeline.predict(X)

    for m in gpipeline.models.keys():
        assert m in cloned_gpipeline.models
        assert id(gpipeline.models[m]) != id(cloned_gpipeline.models[m])


def test_gpipeline_classification():

    gpipeline = GraphPipeline({"PT": PassThrough(), "Logit": LogisticRegression()}, [("PT", "Logit")])
    gpipeline.fit(X, yc)

    yhat_proba = gpipeline.predict_proba(X)
    yhat_proba2 = gpipeline.models["Logit"].predict_proba(X)

    assert yhat_proba.shape == (X.shape[0], 2)
    assert (yhat_proba == yhat_proba2).all()
    assert list(gpipeline.classes_) == [0, 1]


def test_gpipeline_clustering():

    gpipeline = GraphPipeline({"PT": PassThrough(), "kmeans": KMeans(n_clusters=2)}, [("PT", "kmeans")])
    gpipeline.fit(X)

    yhat = gpipeline.predict(X)
    yhat2 = gpipeline.models["kmeans"].predict(X)

    assert (yhat == yhat2).all()


def test_gpipeline_graphviz():

    gpipeline = GraphPipeline(
        {
            "ColNum": ColumnsSelector(columns_to_use=["num1", "num2", "num3"]),
            "ColCat": ColumnsSelector(columns_to_use=["cat1", "cat2"]),
            "Pt": PassThrough(),
        },
        edges=[("ColNum", "Pt"), ("ColCat", "Pt")],
    )

    gpipeline.fit(dfX, y)
    assert isinstance(gpipeline.graphviz, graphviz.dot.Digraph)

    gpipeline = GraphPipeline(
        {
            "ColNum": ColumnsSelector(columns_to_use=["num1", "num2", "num3"]),
            "ColCat": ColumnsSelector(columns_to_use=["cat1", "cat2"]),
            "Pt": PassThrough(),
        },
        edges=[("ColCat", "Pt"), ("ColNum", "Pt")],
    )

    assert isinstance(gpipeline.graphviz, graphviz.dot.Digraph)  # graphviz even before fit is called


def test_graphpipeline_merging_node():

    gpipeline = GraphPipeline(
        {
            "ColNum": ColumnsSelector(columns_to_use=["num1", "num2", "num3"]),
            "ColCat": ColumnsSelector(columns_to_use=["cat1", "cat2"]),
            "Pt": DebugPassThrough(debug=True),
        },
        edges=[("ColNum", "Pt"), ("ColCat", "Pt")],
    )

    gpipeline.fit(dfX, y)

    pt = gpipeline.models["Pt"]
    assert pt._expected_columns == ["num1", "num2", "num3", "cat1", "cat2"]
    assert pt._expected_type == DataTypes.DataFrame
    assert pt._expected_nbcols == 5

    dfX_transformed = gpipeline.transform(dfX)
    assert (dfX_transformed == dfX.loc[:, ["num1", "num2", "num3", "cat1", "cat2"]]).all().all()

    assert gpipeline.get_feature_names() == ["num1", "num2", "num3", "cat1", "cat2"]
    assert gpipeline.get_feature_names_at_node("Pt") == ["num1", "num2", "num3", "cat1", "cat2"]
    assert gpipeline.get_feature_names_at_node("ColNum") == ["num1", "num2", "num3"]
    assert gpipeline.get_feature_names_at_node("ColCat") == ["cat1", "cat2"]

    assert gpipeline.get_input_features_at_node("ColNum") == list(dfX.columns)
    assert gpipeline.get_input_features_at_node("ColCat") == list(dfX.columns)
    assert gpipeline.get_input_features_at_node("Pt") == ["num1", "num2", "num3", "cat1", "cat2"]

    # concatenation in the other oreder
    gpipeline = GraphPipeline(
        {
            "ColNum": ColumnsSelector(columns_to_use=["num1", "num2", "num3"]),
            "ColCat": ColumnsSelector(columns_to_use=["cat1", "cat2"]),
            "Pt": DebugPassThrough(debug=True),
        },
        edges=[("ColCat", "Pt"), ("ColNum", "Pt")],
    )

    gpipeline.fit(dfX, y)

    pt = gpipeline.models["Pt"]
    assert pt._expected_columns == ["cat1", "cat2", "num1", "num2", "num3"]  # Concanteation in the order of the edges
    assert pt._expected_type == DataTypes.DataFrame
    assert pt._expected_nbcols == 5

    assert gpipeline.get_feature_names() == ["cat1", "cat2", "num1", "num2", "num3"]
    assert gpipeline.get_feature_names_at_node("Pt") == ["cat1", "cat2", "num1", "num2", "num3"]
    assert gpipeline.get_feature_names_at_node("ColNum") == ["num1", "num2", "num3"]
    assert gpipeline.get_feature_names_at_node("ColCat") == ["cat1", "cat2"]

    assert gpipeline.get_input_features_at_node("ColNum") == list(dfX.columns)
    assert gpipeline.get_input_features_at_node("ColCat") == list(dfX.columns)
    assert gpipeline.get_input_features_at_node("Pt") == ["cat1", "cat2", "num1", "num2", "num3"]

    dfX_transformed = gpipeline.transform(dfX)
    assert (dfX_transformed == dfX.loc[:, ["cat1", "cat2", "num1", "num2", "num3"]]).all().all()


### Test errors : those tests make sure the GraphPipeline generate an error with something is misspecified
def test_graphpipeline_more_than_one_terminal_node():
    gpipeline = GraphPipeline(
        {
            "ColNum": ColumnsSelector(columns_to_use=["num1", "num2", "num3"]),
            "ColCat": ColumnsSelector(columns_to_use=["cat1", "cat2"]),
            "PtNum": PassThrough(),
            "PtCat": PassThrough(),
        },
        edges=[("ColNum", "PtNum"), ("ColCat", "PtCat")],
    )

    with pytest.raises(ValueError):
        gpipeline.fit(dfX, y)  # ValueError the graph should have only one terminal node, instead i got 2


def test_graphpipeline_edge_not_in_models():
    gpipeline = GraphPipeline(
        {
            "ColNum": ColumnsSelector(columns_to_use=["num1", "num2", "num3"]),
            "ColCat": ColumnsSelector(columns_to_use=["cat1", "cat2"]),
            "PtNum": PassThrough(),
            "PtCat": PassThrough(),
        },
        edges=[("ColNum", "PtNummm"), ("ColCat", "PtCat")],
    )

    with pytest.raises(ValueError):
        gpipeline.fit(dfX, y)  # ValueError "the node 'PtNummm' isn't in the dictionnary of models"


def test_graphpipeline_no_terminal_node():
    gpipeline = GraphPipeline(
        {"A": PassThrough(), "B": PassThrough(), "C": PassThrough()}, edges=[("A", "B", "C"), ("C", "A")]
    )
    with pytest.raises(ValueError):
        gpipeline.fit(X, y)  # ValueError: the graph should have only one terminal node, instead i got 0


def test_graphpipeline_cycle():
    gpipeline = GraphPipeline(
        {"A": PassThrough(), "B": PassThrough(), "C": PassThrough(), "D": PassThrough()},
        edges=[("A", "B", "C"), ("C", "A"), ("C", "D")],
    )

    with pytest.raises(ValueError):
        gpipeline.fit(X, y)  # ValueError: The graph shouldn't have any cycle


# In[]


def test_graphpipeline_fit_params():

    gpipeline = GraphPipeline(
        {"A": DebugPassThrough(debug=True), "B": DebugPassThrough(debug=True), "C": DebugPassThrough(debug=True)},
        edges=[("A", "B", "C")],
    )

    gpipeline.fit(X, y)
    assert gpipeline.models["A"].fit_params == {}
    assert gpipeline.models["B"].fit_params == {}
    assert gpipeline.models["C"].fit_params == {}

    gpipeline.fit(X, y, A__fitparam_A="paramA")
    assert gpipeline.models["A"].fit_params == {"fitparam_A": "paramA"}
    assert gpipeline.models["B"].fit_params == {}
    assert gpipeline.models["C"].fit_params == {}


class TransformerFailNoGroups(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y, groups=None):
        if groups is None:
            raise ValueError("I need a groups")

        assert X.shape[0] == groups.shape[0]
        return self

    def fit_transform(self, X, y, groups=None):
        if groups is None:
            raise ValueError("I need a groups")

        assert X.shape[0] == groups.shape[0]

        return X

    def transform(self, X):
        return X


def test_graphpipeline_passing_of_groups():
    gpipeline = GraphPipeline({"A": TransformerFailNoGroups(), "B": DebugPassThrough(debug=True)}, edges=[("A", "B")])

    with pytest.raises(ValueError):
        gpipeline.fit(X, y)

    groups = np.zeros(len(y))

    gpipeline.fit(X, y, groups)  # check that it didn't failed


def test_graphpipeline_set_params():

    gpipeline = GraphPipeline(
        {"A": PassThrough(), "B": PassThrough(), "C": DebugPassThrough(debug=True)}, edges=[("A", "B", "C")]
    )

    assert gpipeline.models["C"].debug is True
    gpipeline.set_params(C__debug=False)
    assert gpipeline.models["C"].debug is False


def test_graphpipeline_other_input_syntaxes():

    # regular syntax
    gpipeline = GraphPipeline({"A": PassThrough(), "B": PassThrough(), "C": PassThrough()}, edges=[("A", "B", "C")])
    gpipeline._complete_init()

    expected_nodes = {"A", "B", "C"}
    expected_edges = {("A", "B"), ("B", "C")}

    assert set(gpipeline.complete_graph.nodes) == expected_nodes
    assert set(gpipeline.complete_graph.edges) == expected_edges

    # pipeline syntax
    gpipeline = GraphPipeline([("A", PassThrough()), ("B", PassThrough()), ("C", PassThrough())])

    gpipeline._complete_init()
    assert set(gpipeline.complete_graph.nodes) == expected_nodes
    assert set(gpipeline.complete_graph.edges) == expected_edges

    ## with a merge
    expected_nodes = {"A", "B", "C", "D"}
    expected_edges = {("A", "B"), ("B", "D"), ("C", "D")}

    gpipeline = GraphPipeline(
        {"A": PassThrough(), "B": PassThrough(), "C": PassThrough(), "D": PassThrough()},
        edges=[("A", "B", "D"), ("C", "D")],
    )

    gpipeline._complete_init()
    assert set(gpipeline.complete_graph.nodes) == expected_nodes
    assert set(gpipeline.complete_graph.edges) == expected_edges

    gpipeline = GraphPipeline(
        {"A": PassThrough(), "B": PassThrough(), "C": PassThrough(), "D": PassThrough()},
        edges=[("A", "B"), ("B", "D"), ("C", "D")],
    )
    gpipeline._complete_init()
    assert set(gpipeline.complete_graph.nodes) == expected_nodes
    assert set(gpipeline.complete_graph.edges) == expected_edges

    gpipeline = GraphPipeline(
        {"A": PassThrough(), "B": PassThrough(), "C": PassThrough(), "D": PassThrough()}, edges="A - B - D ; C - D"
    )
    gpipeline._complete_init()
    assert set(gpipeline.complete_graph.nodes) == expected_nodes
    assert set(gpipeline.complete_graph.edges) == expected_edges


def test_estimator_type_GraphPipeline():

    pipe_c = GraphPipeline({"scale": StandardScaler(), "rf": RandomForestClassifier()}, edges=[("scale", "rf")])

    assert is_classifier(pipe_c)
    assert not is_regressor(pipe_c)
    assert not is_clusterer(pipe_c)

    pipe_r = GraphPipeline({"scale": StandardScaler(), "rf": RandomForestRegressor()}, edges=[("scale", "rf")])
    assert not is_classifier(pipe_r)
    assert not is_clusterer(pipe_r)
    assert is_regressor(pipe_r)

    pipe_t = GraphPipeline({"scale": StandardScaler(), "rf": StandardScaler()}, edges=[("scale", "rf")])
    assert not is_classifier(pipe_t)
    assert not is_clusterer(pipe_t)
    assert not is_regressor(pipe_t)

    pipe_cluster = GraphPipeline({"scale": StandardScaler(), "kmeans": KMeans()}, edges=[("scale", "kmeans")])
    assert is_clusterer(pipe_cluster)
    assert not is_regressor(pipe_cluster)
    assert not is_classifier(pipe_cluster)


def test_graphpipeline_concat_names():

    df = get_sample_df(size=100, seed=123)
    gpipeline = GraphPipeline(
        models={
            "sel": ColumnsSelector(columns_to_use=["float_col", "int_col"]),
            "vec": CountVectorizerWrapper(columns_to_use=["text_col"]),
            "pt": PassThrough(),
        },
        edges=[("sel", "pt"), ("vec", "pt")],
    )

    gpipeline.fit(df)
    df_res = gpipeline.transform(df)

    assert list(df_res.columns) == [
        "float_col",
        "int_col",
        "text_col__BAG__aaa",
        "text_col__BAG__bbb",
        "text_col__BAG__ccc",
        "text_col__BAG__ddd",
        "text_col__BAG__eee",
        "text_col__BAG__fff",
        "text_col__BAG__jjj",
    ]

    assert gpipeline.get_feature_names() == list(df_res.columns)


class PassThroughWithCallback(TransformerMixin, BaseEstimator):
    """ testing transformer that counts the number of time its methods are called """

    nb_fit = 0
    nb_transform = 0
    nb_fit_transform = 0

    @classmethod
    def reset_counters(cls):
        cls.nb_fit = 0
        cls.nb_transform = 0
        cls.nb_fit_transform = 0

    def fit(self, X, y=None):
        type(self).nb_fit += 1
        return self

    def transform(self, X):
        type(self).nb_transform += 1
        return X

    def fit_transform(self, X, y=None):
        type(self).nb_fit_transform += 1
        return X


class PassThroughWithCallback2(PassThroughWithCallback):
    pass


class PassThroughWithCallback_cant_cv_transform(PassThroughWithCallback):
    def can_cv_transform(self):
        return False


def test_approx_cross_validation_graphpipeline():

    X, y = make_classification(n_samples=100)
    X = pd.DataFrame(X, columns=["col_%d" % i for i in range(X.shape[1])])

    ## Fit ##
    PassThroughWithCallback.reset_counters()
    PassThroughWithCallback2.reset_counters()

    gpipeline = GraphPipeline(
        models={"A": PassThroughWithCallback(), "B": PassThroughWithCallback2(), "C": LogisticRegression()},
        edges=[("A", "B", "C")],
    )

    gpipeline.fit(X, y)

    assert PassThroughWithCallback.nb_fit_transform == 1
    assert PassThroughWithCallback2.nb_fit_transform == 1

    assert PassThroughWithCallback.nb_fit == 0
    assert PassThroughWithCallback2.nb_fit == 0

    assert PassThroughWithCallback.nb_transform == 0
    assert PassThroughWithCallback2.nb_transform == 0

    ## approx cv ##
    PassThroughWithCallback.reset_counters()
    PassThroughWithCallback2.reset_counters()

    gpipeline = GraphPipeline(
        models={"A": PassThroughWithCallback(), "B": PassThroughWithCallback2(), "C": LogisticRegression()},
        edges=[("A", "B", "C")],
    )

    cv_res = gpipeline.approx_cross_validation(X, y, scoring=["neg_mean_squared_error"], cv=10, verbose=False)

    assert PassThroughWithCallback.nb_fit_transform == 10
    assert PassThroughWithCallback2.nb_fit_transform == 10

    assert PassThroughWithCallback.nb_fit == 0
    assert PassThroughWithCallback2.nb_fit == 0

    assert PassThroughWithCallback.nb_transform == 20  # 10 fold x 2 (for score in train and test)
    assert PassThroughWithCallback2.nb_transform == 20

    ## approx cv but skip nodes ##
    PassThroughWithCallback.reset_counters()
    PassThroughWithCallback2.reset_counters()

    gpipeline = GraphPipeline(
        models={"A": PassThroughWithCallback(), "B": PassThroughWithCallback2(), "C": LogisticRegression()},
        edges=[("A", "B", "C")],
    )

    #    cv_res = gpipeline.approx_cross_validation(X, y, scoring=["neg_mean_squared_error"],cv = 10, verbose = False, nodes_not_to_crossvalidate = ("A",))
    cv_res = gpipeline.approx_cross_validation(
        X, y, scoring=["neg_mean_squared_error"], cv=10, verbose=1, nodes_not_to_crossvalidate=("A",)
    )

    assert cv_res is not None
    assert cv_res.shape[0] == 10

    assert PassThroughWithCallback.nb_fit_transform == 1
    assert PassThroughWithCallback2.nb_fit_transform == 10

    assert PassThroughWithCallback.nb_fit == 0
    assert PassThroughWithCallback2.nb_fit == 0

    assert PassThroughWithCallback.nb_transform == 0
    assert PassThroughWithCallback2.nb_transform == 20

    PassThroughWithCallback.reset_counters()
    PassThroughWithCallback2.reset_counters()

    gpipeline = GraphPipeline(
        models={"A": PassThroughWithCallback(), "B": PassThroughWithCallback2(), "C": LogisticRegression()},
        edges=[("A", "B", "C")],
    )

    #    cv_res = gpipeline.approx_cross_validation(X, y, scoring=["neg_mean_squared_error"],cv = 10, verbose = False, nodes_not_to_crossvalidate = ("A",))
    cv_res = gpipeline.approx_cross_validation(
        X, y, scoring=["neg_mean_squared_error"], cv=10, verbose=1, nodes_not_to_crossvalidate=("A", "B")
    )

    assert cv_res is not None
    assert cv_res.shape[0] == 10

    assert PassThroughWithCallback.nb_fit_transform == 1
    assert PassThroughWithCallback2.nb_fit_transform == 1

    assert PassThroughWithCallback.nb_fit == 0
    assert PassThroughWithCallback2.nb_fit == 0

    assert PassThroughWithCallback.nb_transform == 0
    assert PassThroughWithCallback2.nb_transform == 0

    PassThroughWithCallback_cant_cv_transform.reset_counters()
    PassThroughWithCallback2.reset_counters()

    gpipeline = GraphPipeline(
        models={
            "A": PassThroughWithCallback_cant_cv_transform(),
            "B": PassThroughWithCallback2(),
            "C": LogisticRegression(),
        },
        edges=[("A", "B", "C")],
    )

    #    cv_res = gpipeline.approx_cross_validation(X, y, scoring=["neg_mean_squared_error"],cv = 10, verbose = False, nodes_not_to_crossvalidate = ("A",))
    cv_res = gpipeline.approx_cross_validation(
        X, y, scoring=["neg_mean_squared_error"], cv=10, verbose=1, nodes_not_to_crossvalidate={"B"}
    )

    assert cv_res is not None
    assert cv_res.shape[0] == 10

    assert PassThroughWithCallback_cant_cv_transform.nb_fit_transform == 10
    assert PassThroughWithCallback2.nb_fit_transform == 10

    assert PassThroughWithCallback_cant_cv_transform.nb_fit == 0
    assert PassThroughWithCallback2.nb_fit == 0

    assert PassThroughWithCallback_cant_cv_transform.nb_transform == 20
    assert PassThroughWithCallback2.nb_transform == 20


def test_graphpipeline_blockselector():

    Xnum, y = make_classification(n_samples=100)

    dfX_text = pd.DataFrame({"text1": get_random_strings(100), "text2": get_random_strings(100)})

    X = {"text": dfX_text, "num": Xnum}

    graphpipeline = GraphPipeline(
        models={
            "BS_text": BlockSelector("text"),
            "CV": CountVectorizerWrapper(analyzer="char"),
            "BS_num": BlockSelector("num"),
            "RF": DecisionTreeClassifier(),
        },
        edges=[("BS_text", "CV", "RF"), ("BS_num", "RF")],
    )

    graphpipeline.fit(X, y)
    yhat = graphpipeline.predict(X)

    assert yhat.ndim == 1
    assert yhat.shape[0] == y.shape[0]

    ### X = dico ###
    X = {"text": dfX_text, "num": Xnum}

    graphpipeline = GraphPipeline(
        models={"BS_text": BlockSelector("text"), "BS_num": BlockSelector("num"), "PT": DebugPassThrough()},
        edges=[("BS_text", "PT"), ("BS_num", "PT")],
    )

    Xhat = graphpipeline.fit_transform(X)

    assert Xhat.shape[0] == dfX_text.shape[0]
    assert Xhat.shape[1] == dfX_text.shape[1] + Xnum.shape[1]

    assert "text1" in Xhat.columns
    assert "text2" in Xhat.columns
    assert (Xhat.loc[:, ["text1", "text2"]] == dfX_text).all().all()

    cols = diff(list(Xhat.columns), ["text1", "text2"])
    assert (Xhat.loc[:, cols].values == Xnum).all()

    ### X = list
    X = [dfX_text, Xnum]

    graphpipeline = GraphPipeline(
        models={"BS_text": BlockSelector(0), "BS_num": BlockSelector(1), "PT": DebugPassThrough()},
        edges=[("BS_text", "PT"), ("BS_num", "PT")],
    )

    Xhat = graphpipeline.fit_transform(X)

    assert Xhat.shape[0] == dfX_text.shape[0]
    assert Xhat.shape[1] == dfX_text.shape[1] + Xnum.shape[1]

    assert "text1" in Xhat.columns
    assert "text2" in Xhat.columns
    assert (Xhat.loc[:, ["text1", "text2"]] == dfX_text).all().all()

    cols = diff(list(Xhat.columns), ["text1", "text2"])
    assert (Xhat.loc[:, cols].values == Xnum).all()

    ### X = DataManager
    X = BlockManager({"text": dfX_text, "num": Xnum})

    graphpipeline = GraphPipeline(
        models={"BS_text": BlockSelector("text"), "BS_num": BlockSelector("num"), "PT": DebugPassThrough()},
        edges=[("BS_text", "PT"), ("BS_num", "PT")],
    )

    Xhat = graphpipeline.fit_transform(X)

    assert Xhat.shape[0] == dfX_text.shape[0]
    assert Xhat.shape[1] == dfX_text.shape[1] + Xnum.shape[1]

    assert "text1" in Xhat.columns
    assert "text2" in Xhat.columns
    assert (Xhat.loc[:, ["text1", "text2"]] == dfX_text).all().all()

    cols = diff(list(Xhat.columns), ["text1", "text2"])
    assert (Xhat.loc[:, cols].values == Xnum).all()


def test_graphpipeline_blockselector_cv():

    Xnum, y = make_classification(n_samples=100)

    dfX_text = pd.DataFrame({"text1": get_random_strings(100), "text2": get_random_strings(100)})

    ### X = dico
    X = {"text": dfX_text, "num": Xnum}

    graphpipeline = GraphPipeline(
        models={
            "BS_text": BlockSelector("text"),
            "CV": CountVectorizerWrapper(analyzer="char"),
            "BS_num": BlockSelector("num"),
            "RF": DecisionTreeClassifier(),
        },
        edges=[("BS_text", "CV", "RF"), ("BS_num", "RF")],
    )

    from sklearn.model_selection import cross_val_score

    with pytest.raises(ValueError):
        cv_res = cross_val_score(graphpipeline, X, y, scoring="accuracy", cv=10)
        # doesn't work, can't subset dictionnary

    X = BlockManager({"text": dfX_text, "num": Xnum})

    graphpipeline = GraphPipeline(
        models={
            "BS_text": BlockSelector("text"),
            "CV": CountVectorizerWrapper(analyzer="char"),
            "BS_num": BlockSelector("num"),
            "RF": DecisionTreeClassifier(),
        },
        edges=[("BS_text", "CV", "RF"), ("BS_num", "RF")],
    )

    cv_res = cross_val_score(graphpipeline, X, y, scoring="accuracy", cv=10)

    assert len(cv_res) == 10


# In[] : columns names
def test_graphpipeline_get_features_names():

    dfX = pd.DataFrame(
        {
            "text1": ["aa bb", "bb bb cc", "dd aa cc", "ee"],
            "text2": ["AAA ZZZ", "BBB EEE", "DDD TTT", "AAA BBB CCC"],
            "num1": [0, 1, 2, 3],
            "num2": [1.1, 1.5, -2, -3.5],
            "num3": [-1, 1, 25, 4],
            "cat1": ["A", "B", "A", "D"],
            "cat2": ["toto", "tata", "truc", "toto"],
        }
    )

    ###  Test 1  ###
    model = GraphPipeline({"sel": ColumnsSelector(["cat1", "cat2"]), "pt": PassThrough()}, edges=[("sel", "pt")])

    model.fit(dfX)

    assert model.get_feature_names() == ["cat1", "cat2"]  # features at ending nodeC

    assert model.get_feature_names_at_node("pt") == ["cat1", "cat2"]
    assert model.get_feature_names_at_node("sel") == ["cat1", "cat2"]

    assert model.get_input_features_at_node("pt") == ["cat1", "cat2"]
    assert model.get_input_features_at_node("sel") == ["text1", "text2", "num1", "num2", "num3", "cat1", "cat2"]

    ###  Test 2  ###
    model = GraphPipeline(
        {"sel1": ColumnsSelector(["cat1", "cat2"]), "sel2": ColumnsSelector(["num1", "num2"]), "pt": PassThrough()},
        edges=[("sel1", "pt"), ("sel2", "pt")],
    )

    model.fit(dfX)

    assert model.get_feature_names() == ["cat1", "cat2", "num1", "num2"]
    assert model.get_feature_names_at_node("pt") == ["cat1", "cat2", "num1", "num2"]
    assert model.get_feature_names_at_node("sel1") == ["cat1", "cat2"]
    assert model.get_feature_names_at_node("sel2") == ["num1", "num2"]

    assert model.get_input_features_at_node("pt") == ["cat1", "cat2", "num1", "num2"]
    assert model.get_input_features_at_node("sel1") == ["text1", "text2", "num1", "num2", "num3", "cat1", "cat2"]
    assert model.get_input_features_at_node("sel2") == ["text1", "text2", "num1", "num2", "num3", "cat1", "cat2"]

    ###  Test 3  ###
    model = GraphPipeline(
        {
            "sel1": ColumnsSelector(["cat1", "cat2"]),
            "sel2": ColumnsSelector(["num1", "num2"]),
            "sel12": ColumnsSelector(["cat1", "num1"]),
            "pt": PassThrough(),
        },
        edges=[("sel1", "sel12", "pt"), ("sel2", "sel12", "pt")],
    )

    model.fit(dfX)

    assert model.get_feature_names() == ["cat1", "num1"]

    assert model.get_feature_names_at_node("pt") == ["cat1", "num1"]
    assert model.get_feature_names_at_node("sel12") == ["cat1", "num1"]
    assert model.get_feature_names_at_node("sel1") == ["cat1", "cat2"]
    assert model.get_feature_names_at_node("sel2") == ["num1", "num2"]

    assert model.get_input_features_at_node("pt") == ["cat1", "num1"]
    assert model.get_input_features_at_node("sel12") == ["cat1", "cat2", "num1", "num2"]
    assert model.get_input_features_at_node("sel1") == ["text1", "text2", "num1", "num2", "num3", "cat1", "cat2"]
    assert model.get_input_features_at_node("sel2") == ["text1", "text2", "num1", "num2", "num3", "cat1", "cat2"]


class PassThroughtWithFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, prefix):
        self.prefix = prefix

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self._Xcolumns = list(X.columns)

        return self

    def transform(self, X):
        return X

    def get_feature_names(self, input_features=None):
        if input_features is None:
            return [self.prefix + "__" + c for c in self._Xcolumns]
        else:
            return [self.prefix + "__" + c for c in input_features]


def test_graphpipeline_get_features_names_with_input_features():

    xx = np.random.randn(10, 5)
    df = pd.DataFrame(xx, columns=["COL_%d" % j for j in range(xx.shape[1])])

    model = GraphPipeline(
        {"pt1": PassThroughtWithFeatures(prefix="PT1"), "pt2": PassThroughtWithFeatures(prefix="PT2")},
        edges=[("pt1", "pt2")],
    )
    model.fit(df)

    ### Test 1 : without input_features ###
    assert model.get_feature_names() == [
        "PT2__PT1__COL_0",
        "PT2__PT1__COL_1",
        "PT2__PT1__COL_2",
        "PT2__PT1__COL_3",
        "PT2__PT1__COL_4",
    ]
    assert model.get_feature_names_at_node("pt2") == [
        "PT2__PT1__COL_0",
        "PT2__PT1__COL_1",
        "PT2__PT1__COL_2",
        "PT2__PT1__COL_3",
        "PT2__PT1__COL_4",
    ]
    assert model.get_feature_names_at_node("pt1") == [
        "PT1__COL_0",
        "PT1__COL_1",
        "PT1__COL_2",
        "PT1__COL_3",
        "PT1__COL_4",
    ]

    assert model.get_input_features_at_node("pt2") == [
        "PT1__COL_0",
        "PT1__COL_1",
        "PT1__COL_2",
        "PT1__COL_3",
        "PT1__COL_4",
    ]
    assert model.get_input_features_at_node("pt1") == ["COL_0", "COL_1", "COL_2", "COL_3", "COL_4"]

    ### Test 2 : with input feautres ###
    assert model.get_feature_names(input_features=["a", "b", "c", "d", "e"]) == [
        "PT2__PT1__a",
        "PT2__PT1__b",
        "PT2__PT1__c",
        "PT2__PT1__d",
        "PT2__PT1__e",
    ]
    assert model.get_feature_names_at_node("pt2", input_features=["a", "b", "c", "d", "e"]) == [
        "PT2__PT1__a",
        "PT2__PT1__b",
        "PT2__PT1__c",
        "PT2__PT1__d",
        "PT2__PT1__e",
    ]
    assert model.get_feature_names_at_node("pt1", input_features=["a", "b", "c", "d", "e"]) == [
        "PT1__a",
        "PT1__b",
        "PT1__c",
        "PT1__d",
        "PT1__e",
    ]

    assert model.get_input_features_at_node("pt2", input_features=["a", "b", "c", "d", "e"]) == [
        "PT1__a",
        "PT1__b",
        "PT1__c",
        "PT1__d",
        "PT1__e",
    ]
    assert model.get_input_features_at_node("pt1", input_features=["a", "b", "c", "d", "e"]) == [
        "a",
        "b",
        "c",
        "d",
        "e",
    ]

    ### Test 3 :  with numpy array ###
    model = GraphPipeline(
        {"pt1": PassThroughtWithFeatures(prefix="PT1"), "pt2": PassThroughtWithFeatures(prefix="PT2")},
        edges=[("pt1", "pt2")],
    )
    model.fit(xx)

    assert model.get_feature_names() is None
    assert model.get_feature_names_at_node("pt2") is None
    assert model.get_feature_names_at_node("pt1") is None
    assert model.get_input_features_at_node("pt2") is None
    assert model.get_input_features_at_node("pt1") is None

    assert model.get_feature_names(input_features=["a", "b", "c", "d", "e"]) == [
        "PT2__PT1__a",
        "PT2__PT1__b",
        "PT2__PT1__c",
        "PT2__PT1__d",
        "PT2__PT1__e",
    ]
    assert model.get_feature_names_at_node("pt2", input_features=["a", "b", "c", "d", "e"]) == [
        "PT2__PT1__a",
        "PT2__PT1__b",
        "PT2__PT1__c",
        "PT2__PT1__d",
        "PT2__PT1__e",
    ]
    assert model.get_feature_names_at_node("pt1", input_features=["a", "b", "c", "d", "e"]) == [
        "PT1__a",
        "PT1__b",
        "PT1__c",
        "PT1__d",
        "PT1__e",
    ]

    assert model.get_input_features_at_node("pt2", input_features=["a", "b", "c", "d", "e"]) == [
        "PT1__a",
        "PT1__b",
        "PT1__c",
        "PT1__d",
        "PT1__e",
    ]
    assert model.get_input_features_at_node("pt1", input_features=["a", "b", "c", "d", "e"]) == [
        "a",
        "b",
        "c",
        "d",
        "e",
    ]


def test_graphpipeline_no_concat():

    gpipeline = GraphPipeline(
        {"A": DebugPassThrough(debug=True), "B": DebugPassThrough(debug=True), "C": DebugPassThrough(debug=True)},
        edges=[("A", "C"), ("B", "C")],
        no_concat_nodes={"C"},
    )

    Xtransformed = gpipeline.fit_transform(X)
    assert isinstance(Xtransformed, dict)
    assert set(Xtransformed.keys()) == {"A", "B"}
    assert (Xtransformed["A"] == X).all().all()
    assert (Xtransformed["B"] == X).all().all()

    gpipeline = GraphPipeline(
        {"A": DebugPassThrough(debug=True), "B": DebugPassThrough(debug=True), "C": TransformToBlockManager()},
        edges=[("A", "C"), ("B", "C")],
        no_concat_nodes={"C"},
    )

    Xtransformed = gpipeline.fit_transform(X)
    assert isinstance(Xtransformed, BlockManager)
    assert (Xtransformed["A"] == X).all().all()
    assert (Xtransformed["B"] == X).all().all()


def test_graphpipeline_nodes_concat_order():

    cols = list(dfX.columns)

    ### 1
    pipeline = GraphPipeline(
        {
            "pt1": DebugPassThrough(column_prefix="PT1_", debug=True),
            "pt2": DebugPassThrough(column_prefix="PT2_", debug=True),
            "pt3": DebugPassThrough(column_prefix="PT3_", debug=True),
        },
        edges=[("pt1", "pt3"), ("pt2", "pt3")],
    )

    Xres = pipeline.fit_transform(dfX)
    assert list(Xres.columns) == ["PT3__PT1__" + c for c in cols] + [
        "PT3__PT2__" + c for c in cols
    ]  # PT1 on the left, PT2 on the right
    assert list(Xres.columns) == pipeline.get_feature_names()

    ### 2 : reverse order
    pipeline = GraphPipeline(
        {
            "pt1": DebugPassThrough(column_prefix="PT1_", debug=True),
            "pt2": DebugPassThrough(column_prefix="PT2_", debug=True),
            "pt3": DebugPassThrough(column_prefix="PT3_", debug=True),
        },
        edges=[("pt2", "pt3"), ("pt1", "pt3")],
    )

    Xres = pipeline.fit_transform(dfX)
    assert list(Xres.columns) == ["PT3__PT2__" + c for c in cols] + [
        "PT3__PT1__" + c for c in cols
    ]  # PT1 on the left, PT2 on the right
    assert list(Xres.columns) == pipeline.get_feature_names()

    ### 3 : with 4 nodes
    for edges in ([("pt1", "pt3", "pt4"), ("pt2", "pt3", "pt4")], [("pt1", "pt3", "pt4"), ("pt2", "pt3")]):
        pipeline = GraphPipeline(
            {
                "pt1": DebugPassThrough(column_prefix="PT1_", debug=True),
                "pt2": DebugPassThrough(column_prefix="PT2_", debug=True),
                "pt3": DebugPassThrough(column_prefix="PT3_", debug=True),
                "pt4": DebugPassThrough(column_prefix="PT4_", debug=True),
            },
            edges=edges,
        )
        Xres = pipeline.fit_transform(dfX)
        assert list(Xres.columns) == ["PT4__PT3__PT1__" + c for c in cols] + [
            "PT4__PT3__PT2__" + c for c in cols
        ]  # PT1 on the left, PT2 on the right
        assert list(Xres.columns) == pipeline.get_feature_names()

    ### 4 : reverse order
    for edges in ([("pt2", "pt3", "pt4"), ("pt1", "pt3", "pt4")], [("pt2", "pt3", "pt4"), ("pt1", "pt3")]):
        pipeline = GraphPipeline(
            {
                "pt1": DebugPassThrough(column_prefix="PT1_", debug=True),
                "pt2": DebugPassThrough(column_prefix="PT2_", debug=True),
                "pt3": DebugPassThrough(column_prefix="PT3_", debug=True),
                "pt4": DebugPassThrough(column_prefix="PT4_", debug=True),
            },
            edges=edges,
        )
        Xres = pipeline.fit_transform(dfX)
        assert list(Xres.columns) == ["PT4__PT3__PT2__" + c for c in cols] + [
            "PT4__PT3__PT1__" + c for c in cols
        ]  # PT1 on the left, PT2 on the right
        assert list(Xres.columns) == pipeline.get_feature_names()


def test_get_subpipeline():
    def get_pipeline():
        pipeline = GraphPipeline(
            {
                "pt1": DebugPassThrough(column_prefix="PT1_", debug=True),
                "pt2": DebugPassThrough(column_prefix="PT2_", debug=True),
                "pt3": DebugPassThrough(column_prefix="PT3_", debug=True),
                "pt4": DebugPassThrough(column_prefix="PT4_", debug=True),
            },
            edges=[("pt1", "pt3", "pt4"), ("pt2", "pt3", "pt4")],
        )
        return pipeline

    pipeline = get_pipeline()

    pipeline.fit(dfX, y)

    cols = list(dfX.columns)

    # Test on an already fitted models
    pipeline.fit(dfX)
    sub_pipeline = pipeline.get_subpipeline(end_node="pt3")
    assert isinstance(sub_pipeline, GraphPipeline)

    Xres_sub = sub_pipeline.transform(dfX)
    assert list(Xres_sub.columns) == ["PT3__PT1__" + c for c in cols] + ["PT3__PT2__" + c for c in cols]
    assert list(Xres_sub.columns) == sub_pipeline.get_feature_names()
    assert list(Xres_sub.columns) == pipeline.get_feature_names_at_node("pt3")

    # Test on a not fitted model
    pipeline = get_pipeline()

    sub_pipeline = pipeline.get_subpipeline(end_node="pt3")
    assert isinstance(sub_pipeline, GraphPipeline)
    with pytest.raises(NotFittedError):
        sub_pipeline.transform(dfX)

    Xres_sub = sub_pipeline.fit_transform(dfX)
    assert list(Xres_sub.columns) == ["PT3__PT1__" + c for c in cols] + ["PT3__PT2__" + c for c in cols]
    assert list(Xres_sub.columns) == sub_pipeline.get_feature_names()

    # Test if end_node not in pipeline
    with pytest.raises(ValueError):
        pipeline.get_subpipeline("not_in_pipeline")

    # Test if end_node is root node
    pipeline = get_pipeline()
    sub_pipeline = pipeline.get_subpipeline("pt1")
    assert isinstance(sub_pipeline, type(pipeline._models["pt1"]))



def test_GraphPipeline_from_sklearn():
    
    np.random.seed(123)
    X = np.random.randn(100,10)
    y = 1*(np.random.randn(100)>0)
    
    sk_pipeline = Pipeline(steps=[("pt", PassThrough()),
                                  ("dt", DecisionTreeClassifier(random_state=123))
                                  ])


    # Case 1 
    # from a non fitted sklearn Pipeline

    gpipeline = GraphPipeline.from_sklearn(sk_pipeline)
    
    assert isinstance(gpipeline, GraphPipeline)
    with pytest.raises(NotFittedError):
        check_is_fitted(gpipeline)
        
    gpipeline.fit(X, y)
    yhat = gpipeline.predict(X)
    yhat_proba = gpipeline.predict_proba(X)
    
    
    yhat2 = sk_pipeline.fit(X, y).predict(X)
    yhat_proba2 = sk_pipeline.predict_proba(X)

    
    assert (yhat == yhat2).all()
    assert (yhat_proba == yhat_proba2).all()

    # Case 2
    # from an already fitted pipeline
    gpipeline = GraphPipeline.from_sklearn(sk_pipeline)
    yhat = gpipeline.predict(X)
    yhat_proba = gpipeline.predict_proba(X)
    
    
    yhat2 = sk_pipeline.predict(X)
    yhat_proba2 = sk_pipeline.predict_proba(X)
    
    assert (yhat == yhat2).all()
    assert (yhat_proba == yhat_proba2).all()
    
