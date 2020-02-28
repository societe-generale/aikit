# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 11:19:58 2018

@author: Lionel Massoulard
"""

from collections import OrderedDict
from copy import deepcopy

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.utils.metaestimators import if_delegate_has_method

from sklearn.exceptions import NotFittedError
import sklearn.model_selection

import networkx as nx

from aikit.tools.graph_helper import (
    edges_from_edges_string,
    graph_from_edges,
    edges_from_graph,
    get_terminal_nodes,
    iter_graph,
    graphviz_graph,
    get_all_predecessors,
    get_two_by_two_edges,
)


from aikit.transformers.model_wrapper import try_to_find_features_names
from aikit.tools.data_structure_helper import generic_hstack, guess_output_type
from aikit.tools.helper_functions import unlist, dico_key_filter, function_has_named_argument


from aikit.transformers.block_selector import BlockSelector, BlockManager

from aikit.cross_validation import cross_validation, create_cv


def make_pipeline(*steps):
    """ Construct of linear GraphPipeline from a list of steps """
    models = {}
    edges = []
    for model in steps:
        if not hasattr(model, "fit"):
            raise TypeError("the argument should be model with a 'fit' method")
        name = model.__class__.__name__.lower()
        if name in models:
            i = 1
            while (name + "_" + str(i)) in models:
                i += 1
            name = name + "_" + str(i)

        models[name] = model
        edges.append(name)

    return GraphPipeline(models=models, edges=[tuple(edges)])


# In[]
class GraphPipeline(TransformerMixin, BaseEstimator):
    """ sklearn Transformer that act like a pipeline but on a more generic graph structure
    
    Parameter
    ---------
    models : dict
        dictionnary of models, keys = name of models, values = the models
        
        
    edges : list of tuple
        in each tuple each consecutives elements is an edge
        
    verbose : boolean, default = False
        level of verbosity
        
    no_concat_nodes : list or None, default = None
        if not None contains, the node on that list will be called with a dictionnary of Data : key = parent node and values = data at precedding node

    """

    def __init__(self, models, edges=None, verbose=False, no_concat_nodes=None):

        self.models = models
        self.edges = edges

        self._models = None
        self._edges = None
        self.complete_graph = None

        self._preparation_done = False

        # Hidden
        self.verbose = verbose

        self._already_fitted = False

        self.no_concat_nodes = no_concat_nodes

    def _prepare_arguments(self):
        """ preparation of the arguments """
        # Rmk : models = thing entered by the user, _models object used by the code
        # Rmk : edges  = thing entered by the user, _edges object used by the code
        self._models = self.models
        self._edges = self.edges

        if self._edges is None:
            # no edge is given...

            # 1) models is a list of 2-uple, (name,model) as in classical pipeline ...
            if isinstance(self._models, list):
                self._models = OrderedDict(self._models)
                # ... I'll convert it into an OrderedDict

            # 2) models is an NON OrderedDict ...
            elif isinstance(self._models, dict) and not isinstance(self._models, OrderedDict):
                raise TypeError("Please use an OrderedDict or a list of 2-uple to specify a given order")
                # ... I can't (or won't) guess the order

            # 3) model is an OrderedDict ...
            if isinstance(self._models, OrderedDict):
                self._edges = [tuple(self._models.keys())]
                # .. I'll use the order of the dict

        elif isinstance(self._edges, (list, tuple)):
            if len(self._edges) > 0 and not isinstance(self._edges[0], (list, tuple)):
                self._edges = [self._edges]

        elif isinstance(self._edges, str):

            self._edges = edges_from_edges_string(self._edges)

        elif isinstance(self._edges, nx.DiGraph):
            # Already a graph
            self._edges = [tuple(e) for e in self._edges.edges]

        #  Doing that I'll be able to allow more flexible way to enter the graph, while keeping a uniform framework

    def _verif(self):
        """ verification to do on the arguments passed """
        # models should be a dictionnary
        if not isinstance(self._models, dict):
            raise TypeError("_models should be a dictionnary type")

        for key in self._models.keys():
            if not isinstance(key, (int, str)):
                raise TypeError(
                    "models should be a dictionnary with integer or string keys, instead I got %s" % type(key)
                )

        # _edges should be a list or a tuple
        if not isinstance(self._edges, (list, tuple)):
            raise TypeError("_edges should be a list or tuple, instead it is '%s'" % type(self._edges))

        # all element of _edges should be list or tuple
        for sub_edges in self._edges:
            if not isinstance(sub_edges, (list, tuple)):
                raise TypeError("_edges should be an ensemble of lists or tuples, instead it is '%s'" % type(sub_edges))

            if len(sub_edges) <= 1:
                raise ValueError("all elements of _edges should be of length at least 2")

            for n in sub_edges:
                if not isinstance(n, (str, int)):
                    raise TypeError("edge should be string or integer, instead I got %s" % type(n))

        # all elements in the edges should be in models
        all_nodes_in_edges = set()
        for sub_edges in self._edges:
            for node in sub_edges:
                all_nodes_in_edges.add(node)

        for node in all_nodes_in_edges:
            if node not in self._models:
                raise ValueError("the node '%s' isn't in the dictionnary of models" % node)

        # all the element of models should be used in at least one edge
        for model_name in self._models.keys():
            if model_name not in all_nodes_in_edges:
                raise ValueError("the model '%s' doesn't appears within any edges" % model_name)

    def create_graph(self):
        """ create the graphical structure """
        self.complete_graph = graph_from_edges(*self._edges)
        self._verif_graph_structure()
        self._terminal_node = get_terminal_nodes(self.complete_graph)[0]
        self._nodes_order = list(iter_graph(self.complete_graph))

    def _verif_graph_structure(self):
        """ verification on the structure of the graph """
        # Only one terminal node
        terminal_nodes = get_terminal_nodes(self.complete_graph)
        if len(terminal_nodes) != 1:
            raise ValueError("the graph should have only one terminal node, instead i got %d" % len(terminal_nodes))

        # Connexe
        if not nx.is_connected(self.complete_graph.to_undirected()):
            raise ValueError("the graph should be connected")

        # No Cycle
        has_error = False
        try:
            nx.find_cycle(self.complete_graph)
        except nx.NetworkXNoCycle:
            has_error = True

        if not has_error:
            raise ValueError("The graph shouldn't have any cycle")

        # Verif that I have model everywhere
        for name, model in self._models.items():

            # Terminal state
            if name in terminal_nodes:
                if not hasattr(model, "fit"):
                    raise TypeError("The terminal step (%s) should have a fit method" % name)
            else:
                if not hasattr(model, "fit") or not hasattr(model, "fit_transform") or not hasattr(model, "transform"):
                    raise TypeError(
                        "Intermediary step (%s) should have a 'fit','fit_transform' and 'transform' method" % name
                    )

    def _complete_init(self):
        """ complete initialisation of the model,
        creation of graph, verification of arguments, ...
        """

        if self._preparation_done:
            return

        self._prepare_arguments()
        self._verif()
        self.create_graph()
        self._verif_graph_structure()

        self._all_concat_type = {}
        self._all_concat_order = {}

        self._preparation_done = True

    def _fit_transform(self, X, y=None, groups=None, method=None, fit_params=None):
        """ main method of GraphPipeline, handles the fit and predict of object """
        do_fit = method in ("fit", "fit_transform", "fit_predict")

        if not self._already_fitted and not do_fit:
            raise NotFittedError("Please fit the model before")

        # Split fit_params into a 'step-by-step' dictionnary
        fit_params_step = {name: {} for name in self.complete_graph.nodes}
        if fit_params is not None:
            for key, value in fit_params.items():
                step, param = key.split("__", 1)
                fit_params_step[step][param] = value

        data_dico = {}  # Will contain transformed blocks at each node
        feature_dico = {}  # Will contain the get_feature_names() of each node

        if do_fit:
            input_features = getattr(X, "columns", None)
            if input_features is not None:
                input_features = list(input_features)

            self._Xinput_features = input_features

        else:
            input_features = self._Xinput_features

        nodes_done = set()
        for node in self._nodes_order:

            nodes_done.add(node)

            if self.verbose:
                print("start processing node %s ..." % node)

            ### Debugging Help ###
            if (
                getattr(self, "_return_before_node", None) is not None
                and getattr(self, "_return_before_node", None) == node
            ):
                return data_dico

            model = self._models[node]

            predecessors = list(self.complete_graph.predecessors(node))
            # Carefull : here it is not necessary always in the same order

            #### I'll use the order in which the edges were given

            # Concatenation : alphabetical order
            concat_at_this_node = self.no_concat_nodes is None or node not in self.no_concat_nodes

            if len(predecessors) == 0:
                #########################
                ###  No predecessors  ###
                #########################
                if concat_at_this_node:
                    lastX = X

                else:
                    lastX = {"_data": X}
                # ==> Apply on original data

                last_features = input_features

            elif len(predecessors) == 1:
                ########################
                ###  One predecessor ###
                ########################

                # ==> Apply on data coming out of last node
                if concat_at_this_node:
                    lastX = data_dico[predecessors[0]]
                else:
                    lastX = {predecessor: data_dico[predecessor] for predecessor in predecessors}

                last_features = feature_dico[predecessors[0]]

            elif len(predecessors) > 1:
                #######################
                ###  More than one  ###
                #######################
                # ==> concat all the predecessors node and apply it

                ### Fix concatenation order ###
                if do_fit:
                    edges_number = self._get_edges_number(predecessors, node)
                    predecessors = sorted(predecessors, key=lambda p: (edges_number.get(p, -1), p))
                    self._all_concat_order[node] = predecessors
                else:
                    predecessors = self._all_concat_order[node]

                all_lastX = [data_dico[predecessor] for predecessor in predecessors]
                all_last_features = [feature_dico[predecessor] for predecessor in predecessors]

                if all_last_features is None or None in all_last_features:
                    last_features = None
                else:
                    last_features = unlist(all_last_features)

                # all_columns_names = [try_to_find_features_names( self._models[predecessor], input_features = input_features)
                #        for predecessor, input_features in zip(predecessors, all_last_features)]

                # for predecessor, input_features in zip(predecessors,all_last_features):
                #    try_to_find_features_names( self._models[predecessor], input_features = input_features)

                if self.verbose:
                    print("start aggregation...")

                if do_fit:
                    output_type = guess_output_type(all_lastX)
                    self._all_concat_type[node] = output_type
                else:
                    output_type = self._all_concat_type[node]

                if concat_at_this_node:
                    lastX = generic_hstack(all_lastX, output_type=output_type, all_columns_names=all_last_features)
                else:
                    lastX = {predecessor: data_dico[predecessor] for predecessor in predecessors}

            if node != self._terminal_node:
                # This is not the end of the graph
                if do_fit:
                    if groups is not None and function_has_named_argument(model.fit_transform, "groups"):
                        data_dico[node] = model.fit_transform(lastX, y, groups=groups, **fit_params_step[node])
                    else:
                        data_dico[node] = model.fit_transform(lastX, y, **fit_params_step[node])

                    # ICI : on pourrait sauté le fit pour certains models dans le fit params
                    # Quelque-chose comme :

                    # if node in preffited_models:
                    #
                    # self._model[node] = preffited_models[node]
                    # model = preffited_models[node]
                    # + copy model into pipeline

                    #    data_dico[node] = model.transform(lastX, y)
                    # else:
                    #    data_dico[node] = model.fit_transform(lastX, y, **fit_params_step[node] )

                else:
                    data_dico[node] = model.transform(lastX)

                feature_dico[node] = try_to_find_features_names(model, input_features=last_features)

            else:
                # This is the last node of the Graph
                if method == "fit":
                    if groups is not None and function_has_named_argument(model.fit, "groups"):
                        model.fit(lastX, y, groups, **fit_params_step[node])
                    else:
                        model.fit(lastX, y, **fit_params_step[node])
                    result = self

                elif method == "fit_predict":
                    if groups is not None and function_has_named_argument(model.fit_predict, "groups"):
                        result = model.fit_predict(lastX, y, groups, **fit_params_step[node])
                    else:
                        result = model.fit_predict(lastX, y, **fit_params_step[node])

                elif method == "fit_transform":
                    if groups is not None and function_has_named_argument(model.fit_transform, "groups"):
                        result = model.fit_transform(lastX, y, groups, **fit_params_step[node])
                    else:
                        result = model.fit_transform(lastX, y, **fit_params_step[node])

                elif method == "transform":
                    result = model.transform(lastX)

                elif method == "predict":
                    result = model.predict(lastX)

                elif method == "predict_proba":
                    result = model.predict_proba(lastX)

                elif method == "predict_log_proba":
                    result = model.predict_log_proba(lastX)

                elif method == "decision_function":
                    result = model.decision_function(lastX)

                elif method == "score":
                    result = model.score(lastX, y)

                else:
                    raise ValueError("I don't know that kind of method '%s' " % method)

                feature_dico[node] = try_to_find_features_names(model, input_features=last_features)
                return result

            #######################
            #### Dico cleaning ####
            #######################
            # I'll do a step of cleaning to remove useless blocks in memory
            # I need to remove data in nodes that wont be accessed anymore
            still_usefull = set()
            for n in self.complete_graph.nodes:
                if n in nodes_done:
                    continue

                p = list(self.complete_graph.predecessors(n))
                still_usefull.update(p)

            for n in data_dico.keys():
                if data_dico[n] is None:
                    continue
                if n not in still_usefull:
                    if self.verbose:
                        print("deleting useless node %s" % n)
                    data_dico[n] = None
                    # Carefull, don't use del as it will change the dictionnary key

    ### Exposed methods of objects ###
    def fit(self, X, y=None, groups=None, **fit_params):
        self._complete_init()
        self._fit_transform(X=X, y=y, groups=groups, method="fit", fit_params=fit_params)
        self._already_fitted = True
        return self

    @if_delegate_has_method(delegate="_final_estimator")
    def fit_transform(self, X, y=None, groups=None, **fit_params):
        self._complete_init()
        Xres = self._fit_transform(X=X, y=y, groups=groups, method="fit_transform", fit_params=fit_params)
        self._already_fitted = True
        return Xres

    @if_delegate_has_method(delegate="_final_estimator")
    def transform(self, X):
        Xres = self._fit_transform(X=X, y=None, groups=None, method="transform", fit_params=None)
        return Xres

    @if_delegate_has_method(delegate="_final_estimator")
    def predict(self, X):
        Xres = self._fit_transform(X=X, y=None, groups=None, method="predict", fit_params=None)
        return Xres

    @if_delegate_has_method(delegate="_final_estimator")
    def fit_predict(self, X, y=None, groups=None, **fit_params):
        self._complete_init()
        Xres = self._fit_transform(X=X, y=y, groups=groups, method="fit_predict", fit_params=fit_params)
        self._already_fitted = True
        return Xres

    @if_delegate_has_method(delegate="_final_estimator")
    def predict_proba(self, X):
        Xres = self._fit_transform(X=X, y=None, groups=None, method="predict_proba", fit_params=None)
        return Xres

    @if_delegate_has_method(delegate="_final_estimator")
    def predict_log_proba(self, X):
        Xres = self._fit_transform(X=X, y=None, groups=None, method="predict_log_proba", fit_params=None)
        return Xres

    @if_delegate_has_method(delegate="_final_estimator")
    def decision_function(self, X):
        Xres = self._fit_transform(X=X, y=None, groups=None, method="decision_function", fit_params=None)
        return Xres

    @if_delegate_has_method(delegate="_final_estimator")
    def score(self, X, y=None):
        Xres = self._fit_transform(X=X, y=y, groups=None, method="score", fit_params=None)
        return Xres

    # @if_delegate_has_method(delegate='_final_estimator')
    def get_feature_names(self, input_features=None):
        """ retrieve the features name at the last node """
        if not self._already_fitted:
            raise NotFittedError("Please fit the model before")

        return self.get_feature_names_at_node(self._terminal_node, input_features=input_features)

    def get_input_features_at_node(self, node, input_features=None):
        """ retrieve the names of the feature at the ENTRY of a given node 
        
        Parameter
        ---------
        node : string or ...
            name of the node
            
        input_features : None or list
            if not None, the list of feature (at the input of the graphpipeline)
        
        Returns
        -------
            list of features for the given node, or None
        
        """
        return self._get_feature_names_at_node(node=node, input_features=input_features, entry=True)

    def get_feature_names_at_node(self, node, input_features=None):
        """ retrieve the names of the feature a given node 
        
        Parameter
        ---------
        node : string or ...
            name of the node
            
        input_features : None or list
            if not None, the list of feature (at the input of the graphpipeline)
        
        Returns
        -------
            list of features for the given node, or None
        
        """
        return self._get_feature_names_at_node(node=node, input_features=input_features, entry=False)

    def _get_feature_names_at_node(self, node, input_features=None, entry=False):
        """ main function to make the feature go down the graphpipleine and retrieve the features at a given node 
        
        Parameter
        ---------
        node : string or ..
            name of the node
            
        input_features : None or list
            if not None, the list of feature (at the input of the graphpipeline)
            
        entry : boolean, default = False
            if True will retrieve the feature at the ENTRY of a given model, otherwise the feature at the EXIT of a given model
        
        Returns
        -------
            list of features for the given node, or None
        
        """

        if not self._already_fitted:
            raise NotFittedError("Please fit the model before")

        if input_features is None:
            input_features = self._Xinput_features

        feature_dico = {}
        for n in self._nodes_order:

            predecessors = list(self.complete_graph.predecessors(n))

            if len(predecessors) == 0:
                last_features = input_features

            elif len(predecessors) == 1:
                last_features = feature_dico[predecessors[0]]

            else:
                predecessors = self._all_concat_order[n]
                all_last_features = [feature_dico[predecessor] for predecessor in predecessors]

                if all_last_features is None or None in all_last_features:
                    last_features = None
                else:
                    last_features = unlist(all_last_features)

            model = self._models[n]

            if last_features is None or None in last_features:
                last_features = None

            if n != node:
                feature_dico[n] = try_to_find_features_names(model, input_features=last_features)
                if feature_dico[n] is not None:
                    feature_dico[n] = list(feature_dico[n])

            else:

                if entry:
                    # Entry, I'll return the features at the entry of the node
                    return last_features
                else:
                    # Otherwise I'll return the features at the exit of the node
                    feature_dico[n] = try_to_find_features_names(model, input_features=last_features)
                    if feature_dico[n] is not None:
                        feature_dico[n] = list(feature_dico[n])

                    return feature_dico[n]

        raise ValueError("node %s isn't in the graph" % node)

    # def get_feature_names_at_node(self,)

    # TODO : get_features_names_by_node(self,node)

    ### property to be able to easily retrieve a few things from last estimator
    # TODO : peut etre aussi rajouter features names
    @property
    def _final_estimator(self):
        self._complete_init()
        return self._models[self._terminal_node]

    @property
    def classes_(self):
        return self._final_estimator.classes_

    @property
    def n_outputs_(self):
        return self._final_estimator.n_outputs_

    @property
    def _estimator_type(self):
        return self._final_estimator._estimator_type

    # TODO : faire un truc pour avoir les features_names en sortie

    #    ### Params
    def get_params(self, deep=False):
        out = super(GraphPipeline, self).get_params(deep=False)

        if deep:

            self._complete_init()
            out.update(self._models)

            for name, model in self._models.items():
                for key, value in model.get_params(deep=True).items():
                    out["%s__%s" % (name, key)] = value

        return out

    def set_params(self, **params):

        # 1) edges
        if "edges" in params:
            setattr(self, "edges", params.pop("edges"))
            self._complete_init()

        # 2) models
        if "models" in params:
            setattr(self, "models", params.pop("models"))
            self._complete_init()

        # 3) model
        self._complete_init()
        for name in params.keys():
            if "__" not in name and name in self._models:
                ### It means I modifying a model
                self._models[name] = params.pop(name)

                if isinstance(self.models, dict):
                    self.models[name] = self._models[name]
                elif isinstance(self.models, list):
                    for i, (n, m) in enumerate(self.models):
                        if n == name:
                            self.models[i] = (name, self._models[name])
                else:
                    raise ValueError("I can't modify that model %s" % name)

        # 4)
        super(GraphPipeline, self).set_params(**params)

        return self

    ### Draw graph
    @property
    def graphviz(self):
        """ return a graphviz version of the graph,
        in IPython like environnement it will make a nice plot
        """

        self._complete_init()
        return graphviz_graph(self.complete_graph)

    def draw(self, ax=None):
        """ matplotlib draw """
        self._complete_init()

        import matplotlib.pylab as plt

        if ax is None:
            ax = plt.cla()

        pos = nx.spring_layout(self.complete_graph)  # positions for all nodes
        nx.draw(self.complete_graph, pos=pos, ax=ax)
        nx.draw_networkx_labels(self.complete_graph, pos=pos, ax=ax)

        return ax

    ### Helper
    def _get_edges_number(self, predecessors, node):
        """ helper function to retrieve the number of the edge in which all predecessors - node appears
        This is used to specify the order of concatenation
        """
        edges_number = {}
        for p in predecessors:
            for en, edge in enumerate(self._edges):
                for e1, e2 in zip(edge[:-1], edge[1:]):
                    if (e1, e2) == (p, node):
                        edges_number[p] = en

        return edges_number

    def _approx_cross_validation_pre_calculation(
        self,
        X,
        y,
        groups,
        scoring,
        cv,
        verbose,
        fit_params_step,
        return_predict,
        method,
        no_scoring,
        stopping_round,
        stopping_threshold,
        nodes_not_to_crossvalidate,
        nodes_cant_cv_transform,
        kwargs_step,
    ):
        """ sub-method to loop through the nodes of the pipeline and pre-compute everything that can be pre-computed """

        data_dico = {}  # Will contain transformed blocks at each node

        nodes_done = set()
        for node in self._nodes_order:

            concat_at_this_node = self.no_concat_nodes is None or node not in self.no_concat_nodes
            if not concat_at_this_node:
                raise NotImplementedError(
                    "Approx cross-validation does't work if no concatenation (node %s)" % str(node)
                )

            nodes_done.add(node)

            if self.verbose:
                print("start processing node %s ..." % node)

            ### Debugging Help ###
            # if getattr(self,"_return_before_node",None) is not None and getattr(self,"_return_before_node",None) == node:
            #    return data_dico

            model = self._models[node]

            predecessors = list(self.complete_graph.predecessors(node))
            # Carefull : here it is not necessary always in the same order

            #### I'll use the order in which the edges were given

            # Concatenation : alphabetical order

            if len(predecessors) == 0:
                #########################
                ###  No predecessors  ###
                #########################

                # ==> Apply on original data
                lastX = X

            elif len(predecessors) == 1:
                ########################
                ###  One predecessor ###
                ########################

                # ==> Apply on data coming out of last node
                lastX = data_dico[predecessors[0]]
                # data_dico[node] = model.fit_transform(lastX, y, **fit_params_step[node] )

            elif len(predecessors) > 1:
                #######################
                ###  More than one  ###
                #######################
                # ==> concat all the predecessors node and apply it

                ### Fix concatenation order ###
                edges_number = self._get_edges_number(predecessors, node)
                predecessors = sorted(predecessors, key=lambda p: (edges_number.get(p, -1), p))
                self._all_concat_order[node] = predecessors

                all_lastX = [data_dico[predecessor] for predecessor in predecessors]

                if self.verbose:
                    print("start aggregation...")

                # if do_fit:
                output_type = guess_output_type(all_lastX)
                self._all_concat_type[node] = output_type
                # else:
                #    output_type = self._all_concat_type[node]
                has_none = False
                for x in all_lastX:
                    if x is None:
                        has_none = True
                        break

                # None in all_lastX

                if has_none:
                    lastX = None
                else:
                    lastX = generic_hstack(all_lastX, output_type=output_type)

            if node != self._terminal_node and lastX is not None:
                # This is not the end of the graph

                if node not in nodes_not_to_crossvalidate and node not in nodes_cant_cv_transform:
                    ### 1) Node should BE crossvalitaded  ...
                    ### 2) ... and we CAN use 'cv_transform'

                    if self.verbose:
                        print("do crossvalidation on %s" % node)

                    _, data_dico[node] = cross_validation(
                        model,
                        lastX,
                        y,
                        groups=groups,
                        cv=cv,
                        verbose=verbose,
                        fit_params=fit_params_step[node],
                        return_predict=True,
                        method="transform",
                        no_scoring=True,
                        stopping_round=None,
                        stopping_threshold=None,
                        **kwargs_step[node]
                    )

                elif node not in nodes_not_to_crossvalidate and node in nodes_cant_cv_transform:
                    ### 1) Node should BE crossvalitated ...
                    ### 2) ... but we can't use 'cv_transform'

                    if self.verbose:
                        print("can't do node %s" % node)
                    data_dico[node] = None  # Can't compute this node

                else:
                    ### Node that shouldn't be cross-validated ###

                    if self.verbose:
                        print("skip crossvalidation on %s" % node)
                    cloned_model = clone(model)
                    if groups is not None and function_has_named_argument(cloned_model.fit_transform, "groups"):
                        data_dico[node] = cloned_model.fit_transform(lastX, y, groups, **fit_params_step[node])
                    else:
                        data_dico[node] = cloned_model.fit_transform(lastX, y, **fit_params_step[node])

            elif lastX is not None:

                ### CV no matter what at the last node ###

                #                if node not in nodes_not_to_crossvalidate and node not in nodes_cant_cv_transform:
                #
                #                    # This is the last node of the Graph
                #                    result = approx_cross_validation( model, lastX, y, groups = groups, scoring = scoring, cv = cv ,
                #                                                verbose = verbose, fit_params = fit_params_step[node],
                #                                                return_predict = return_predict , method = method, no_scoring = no_scoring,
                #                                                stopping_round = stopping_round, stopping_threshold = stopping_threshold,
                #                                                **kwargs_step[node])
                #
                #                elif node not in nodes_not_to_crossvalidate and node in nodes_cant_cv_transform:
                #                    pass
                #
                #                else:

                # This is the last node of the Graph
                result = cross_validation(
                    model,
                    lastX,
                    y,
                    groups=groups,
                    scoring=scoring,
                    cv=cv,
                    verbose=verbose,
                    fit_params=fit_params_step[node],
                    return_predict=return_predict,
                    method=method,
                    no_scoring=no_scoring,
                    stopping_round=stopping_round,
                    stopping_threshold=stopping_threshold,
                    **kwargs_step[node]
                )

                # Rmk : if we do that so column regarding the time of fit are 'false' : they will only account for the time spent in the last node

                return True, data_dico, result
            #                return result

            else:
                ###
                if self.verbose:
                    print("can't compute node %s because lastX is None" % node)
                data_dico[node] = None
                # return result

        return False, data_dico, None  # None : no result yet

    def _approx_cross_validation_create_sub_graph_pipeline(self, data_dico, X):
        """ this sub-method create the new graph-pipeline that should be fully cross-validated,
        it also create the new data on which to cv 
        
        Returns
        -------
        new_graph_pipeline
        
        new_data
        """
        ### Create a new GraphPipeline with only the remaning Nodes ###

        dones_nodes = set()
        for k, v in data_dico.items():
            if v is not None:
                dones_nodes.add(k)

        newG = nx.DiGraph()
        new_models = {}
        new_datas = {}
        block_selector_nodes = set()

        for n1, n2 in self.complete_graph.edges:

            if n1 in dones_nodes and n2 in dones_nodes:
                pass

            elif n1 in dones_nodes and n2 not in dones_nodes:

                newG.add_edge("_data_%s" % n1, n2)

                new_models[n2] = self._models[n2]
                new_models["_data_%s" % n1] = BlockSelector("_data_%s" % n1)

                new_datas["_data_%s" % n1] = data_dico[n1]

                block_selector_nodes.add("_data_%s" % n1)
                # Add a BlockSelector

            elif n1 not in dones_nodes and n2 not in dones_nodes:
                newG.add_edge(n1, n2)

                new_models[n1] = self._models[n1]
                new_models[n2] = self._models[n2]

            else:
                raise ValueError("Should never go there")

        nodes = list(newG.nodes)  # copy because I'll modify the graph
        for n in nodes:
            preds = list(newG.predecessors(n))
            if len(preds) == 0 and n not in block_selector_nodes:

                newG.add_edge("_data_", n)
                new_models["_data_"] = BlockSelector("_data_")

                new_datas["_data_"] = X

        new_data_dtm = BlockManager(new_datas)

        new_graph_pipeline = GraphPipeline(models=new_models, edges=edges_from_graph(newG))

        return new_graph_pipeline, new_data_dtm

    def approx_cross_validation(
        self,
        X,
        y,
        groups=None,
        scoring=None,
        cv=None,
        verbose=1,
        fit_params=None,
        return_predict=False,
        method=None,
        no_scoring=False,
        stopping_round=None,
        stopping_threshold=None,
        nodes_not_to_crossvalidate=None,
        **kwargs
    ):

        ###################
        ### Preparation ###
        ###################
        _orig_verbose = self.verbose

        self.verbose = verbose

        self._complete_init()

        if nodes_not_to_crossvalidate is None:
            nodes_not_to_crossvalidate = set()

        #################################################################
        ### Prepare the list of nodes that can't be 'cv_transformed' ####
        #################################################################
        nodes_cant_cv_transform = set()
        for node, m in self._models.items():
            cant = True
            if hasattr(m, "can_cv_transform"):
                if m.can_cv_transform():
                    cant = False

            if cant:
                nodes_cant_cv_transform.add(node)

        # verif:
        for node in nodes_cant_cv_transform:
            if node not in self._models:
                raise ValueError("the node (within nodes_cant_cv_transform) %s isn't in the node of the model" % node)

        for node in nodes_cant_cv_transform:
            if node not in self._models:
                raise ValueError("the node (within nodes_cant_cv_transform) %s isn't in the node of the model" % node)

        cv = create_cv(
            cv, y, classifier=sklearn.model_selection._validation.is_classifier(self), shuffle=True, random_state=123
        )

        # Split fit_params into a 'step-by-step' dictionnary
        fit_params_step = {name: {} for name in self.complete_graph.nodes}
        if fit_params is not None:
            for key, value in fit_params.items():
                step, param = key.split("__", 1)
                fit_params_step[step][param] = value

        kwargs_step = {name: {} for name in self.complete_graph.nodes}
        if kwargs:
            for key, value in kwargs.items():
                step, param = key.split("__", 1)
                kwargs_step[step][param] = value

        ################################
        ### Pre-calculate everything ###
        ################################
        is_finished, data_dico, result = self._approx_cross_validation_pre_calculation(
            X=X,
            y=y,
            groups=groups,
            scoring=scoring,
            cv=cv,
            verbose=verbose,
            fit_params_step=fit_params_step,
            return_predict=return_predict,
            method=method,
            no_scoring=no_scoring,
            stopping_round=stopping_round,
            stopping_threshold=stopping_threshold,
            nodes_not_to_crossvalidate=nodes_not_to_crossvalidate,
            nodes_cant_cv_transform=nodes_cant_cv_transform,
            kwargs_step=kwargs_step,
        )

        if is_finished:
            if verbose:
                print("CV is finished")
            self.verbose = _orig_verbose
            return result

        ###########################################################
        ### Create a new graphpipeline with the remaining nodes ###
        ###########################################################
        new_graph_pipeline, new_data_dtm = self._approx_cross_validation_create_sub_graph_pipeline(data_dico, X)

        if verbose:
            print("here is a new GraphPipeline")
            print(new_graph_pipeline)

            print("")
            print("new_data_dtm")
            print(type(new_data_dtm))

        ############################################################################
        ### Now do a 'classical cross-validation' on the remaining GraphPipeline ###
        ############################################################################
        result = cross_validation(
            new_graph_pipeline,
            new_data_dtm,
            y,
            groups=groups,
            scoring=scoring,
            cv=cv,
            verbose=verbose,
            fit_params=fit_params,
            return_predict=return_predict,
            method=method,
            no_scoring=no_scoring,
            stopping_round=stopping_round,
            stopping_threshold=stopping_threshold,
            approximate_cv=False,
            **kwargs
        )

        self.verbose = _orig_verbose

        return result

    def get_subpipeline(self, end_node, deepcopy_models=False):
        """ create a New model that corresponds to the original GraphPipeline but with a new ending node
        If the original GraphPipeline was fitted, the new model will also be fitted
        
        Parameters
        ----------
        
        end_node : str
            the name of the node at which the new pipeline will stop. Must be in the Graph
            
        deepcopy_models : boolean, default=False
            if True will make a deepcopy of the models.
        
        Returns
        -------
        new GraphPipeline instance

        """
        self._complete_init()

        if end_node not in self.complete_graph:
            raise ValueError("the node '%s' isn't in the original graph" % end_node)

        # get all predecessors of nodes => to include in the graph
        predecessors = get_all_predecessors(self.complete_graph, end_node)
        nodes_to_keep = list(predecessors) + [end_node]

        # Remark : we could separated into a  submethod to create a subpipeline from a list of nodes

        if len(nodes_to_keep) == 1:
            assert end_node == nodes_to_keep[0]
            return self._models[end_node]

        # filter edges
        edges_to_keep = []
        for e1, e2 in get_two_by_two_edges(*self._edges):
            if e1 in nodes_to_keep and e2 in nodes_to_keep:
                edges_to_keep.append((e1, e2))

        # I do that insteaf of :
        # complete_graph_sub.subgraph(nodes_to_keep).edges
        # beacause that way I preseve the order of the edges, which handle the concatenation order

        # Retrieve sklearn model
        if deepcopy_models:
            models = {node: deepcopy(self._models[node]) for node in nodes_to_keep}
        else:
            models = {node: self._models[node] for node in nodes_to_keep}

        # Change 'no_concat_nodes'
        if self.no_concat_nodes is None:
            no_concat_nodes = None
        else:
            no_concat_nodes = [n for n in self.no_concat_nodes if n in nodes_to_keep]
            no_concat_nodes = type(self.no_concat_nodes)(no_concat_nodes)
            if len(no_concat_nodes) == 0:
                no_concat_nodes = None

        ###############################
        ###   Create new pipeline   ###
        ###############################
        sub_pipeline = GraphPipeline(
            models=models, edges=edges_to_keep, verbose=self.verbose, no_concat_nodes=no_concat_nodes
        )

        # Internal modification to change the state
        if self._preparation_done:
            sub_pipeline._complete_init()

        if not self._already_fitted:
            return sub_pipeline

        # here the pipeline was fitted
        sub_pipeline._already_fitted = True
        sub_pipeline._Xinput_features = deepcopy(self._Xinput_features)  # copy just to be safe
        sub_pipeline._all_concat_order = dico_key_filter(self._all_concat_order, lambda n: n in nodes_to_keep)
        sub_pipeline._all_concat_type = dico_key_filter(self._all_concat_type, lambda n: n in nodes_to_keep)

        return sub_pipeline
