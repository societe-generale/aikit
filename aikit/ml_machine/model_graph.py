# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:22:19 2018

@author: Lionel Massoulard
"""


try:
    import matplotlib.pylab as plt
except ImportError:
    plt = None

import networkx as nx
from collections import OrderedDict

try:
    import graphviz
except ModuleNotFoundError:
    graphviz = None


from aikit.tools.helper_functions import unnest_tuple, tuple_include, diff
from aikit.enums import SpecialModels, StepCategories

from aikit.tools import graph_helper as gh

from aikit.ml_machine.ml_machine_registration import MODEL_REGISTER


# In[]

# def _get_params(name):
#    """ temp function : to test """
#    return {"__param__":"_%s_" % n} # Retrieve the parameters


def _get_columns(var_type, var_type_columns_dico):
    """ function to get the columns 
    
    Parameters
    ----------
    var_type : tuple or str
        the type of variable we want (one type : string, several : tuple)
    
    var_type_columns_dico : dict
        dictionnary with keys = type of variable and values = name of columns
        
    Returns
    -------
    list of columns corresponding to that type(s)
    
    """
    if not isinstance(var_type, tuple):
        var_type = (var_type,)()

    res = []
    for v in var_type:
        res += var_type_columns_dico[v]

    return res


def _get_params(node, params):
    if params is None:
        return {}
    return params.get(node, {})


def _must_include_selector(name):
    """ function that tells if I need to include a selector before a given model 
    it is using the init_params in MODEL_REGISTER,
    by default I'll include a selector
    """
    if name is None:
        return True

    if name[1] == SpecialModels.ColumnsSelector:
        return False

    return "columns_to_use" not in MODEL_REGISTER.init_parameters.get(name, {})


# In[]
# def gplot(G):
#    plt.cla()
#    pos=nx.spring_layout(G) # positions for all nodes
#    nx.draw(G,pos=pos)
#    nx.draw_networkx_labels(G,pos = pos)

# def is_composition_step(step_name):
#    """ is it a composition step """
#    return step_name in (StepCategories.TargetTransformer,)


def is_composition_model(name):
    """ is it a composition model """
    return StepCategories.is_composition_step(name[0])


def model_graph_plot(Graph, ax=None):
    """ plot a graphical representing a model """
    if plt is None:
        raise ValueError("Please install matplotlib")

    if ax is None:
        ax = plt.gca()

    ax.cla()
    pos = nx.spring_layout(Graph)
    n1 = []
    n2 = []
    for step_name, model_name in Graph.nodes:
        if StepCategories.is_composition_step(step_name):
            n1.append((step_name, model_name))
        else:
            n2.append((step_name, model_name))

    nx.draw(Graph, pos=pos, ax=ax, node_color="y", nodelist=n1)
    nx.draw(Graph, pos=pos, ax=ax, node_color="r", nodelist=n2)
    nx.draw_networkx_labels(Graph, pos=pos, ax=ax)

    return ax


def create_graphical_representation(steps):
    """ from a an OrderedDict of steps create a Graphical reprensetation of the model we'll use """

    # Rmk : il faut a priori, mettre les numero de l'etape dans le graph
    # + mettre les labels correct
    # comme ça on pourra avoir plusieurs noeud avec le meme nom (Ex : Scaler...)

    ### 1) Split Composion Steps vs Rest
    all_composition_steps = []
    all_others = []
    for (step_name, model_name), var_type in steps.items():
        if StepCategories.is_composition_step(step_name):
            all_composition_steps.append((step_name, model_name, var_type))
        else:
            all_others.append((step_name, model_name, var_type))

    ### 2) Create Graph for non-composition step
    new_steps = OrderedDict()

    G = nx.DiGraph()
    for step_name, model_name, var_type in all_others:
        # for name,var_type in steps.items():

        unested_var_type = unnest_tuple(var_type)

        terminal_nodes = gh.get_terminal_nodes(G)  # Terminal links : I'll add the new step on one (or more) of those

        ending_node_type = {unnest_tuple(steps[node]): node for node in terminal_nodes}

        node_name = (step_name, model_name)  # 2-uple
        if node_name in G.nodes:
            raise ValueError("This node already exists '(%s,%s)'" % node_name)

        # 1) Soit je rattache le nouveau a UN noeud terminal
        # 2) Soit je cree une nouvelle branche (nouveau noeud ratacher a rien)
        # 3) Soit je rattache a PLUSIEURS noeud terminaux

        elif unested_var_type in ending_node_type:
            ### 1) I already have a branch of this type
            last_node = ending_node_type[unested_var_type]
            G = gh.add_node_after(G, node_name, last_node)

        ### I don't have a branch ###
        else:
            all_candidates = [(t, n) for t, n in ending_node_type.items() if tuple_include(t, unested_var_type)]
            # I need to look where I want to plug it #
            if len(all_candidates) == 0:
                ### 2) Je dois creer une nouvelle branche : aucun noeud ###
                G = gh.add_node_after(G, node_name)
            else:
                ### 3) Je rattache a plusieurs noeuds

                ### Ici : il faut parfois rajouter un noeud en AMONT, si on a des types qui n'ont pas ete rajouter
                types_added = unnest_tuple([t for t, n in all_candidates])
                types_not_added = diff(unested_var_type, types_added)
                if len(types_not_added) > 0:

                    name_of_cat = "Selector_%s" % unnest_tuple(types_not_added)
                    new_node = (name_of_cat, (name_of_cat, SpecialModels.ColumnsSelector))

                    G = gh.add_node_after(G, new_node)

                    new_steps[new_node] = types_not_added  # I also must dynamically add the node to the list of steps

                    all_candidates = all_candidates + [(types_not_added, new_node)]

                G = gh.add_node_after(G, node_name, *[n for t, n in all_candidates])

    ### 3) Include composition node on top
    for step_name, model_name, _ in reversed(all_composition_steps):
        starting_nodes = gh.get_starting_nodes(G)
        for n in starting_nodes:
            G.add_edge((step_name, model_name), n)

    ### 4) Verify the Graph structure

    for (step_name, model_name), _ in steps.items():
        if (step_name, model_name) not in G:
            raise ValueError("'(%s , %s)' should be in graph" % (step_name, model_name))
    # all nodes were in the steps
    for node in G.nodes():
        if node not in steps and node not in new_steps:
            raise ValueError("'(%s,%s)' shouldn't be in graph" % node)

    assert_model_graph_structure(G)

    return G, new_steps


def assert_model_graph_structure(G):
    """ verification on the structure of the graph """

    # only one terminal node
    if len(gh.get_terminal_nodes(G)) != 1:
        raise ValueError("I should have only one terminal node")

    # connex graph
    if not gh.is_connected(G):
        raise ValueError("the graph should be connected")

    # no cycle
    if gh.has_cycle(G):
        raise ValueError("The graph shouldn't have any cycle")

    for node in G.nodes:
        if is_composition_model(node):
            successors = list(G.successors(node))

            if len(successors) == 0:
                raise ValueError("Composition node %s has no successor" % str(node))

            for successor in successors:
                predecessors = list(G.predecessors(successor))
                if predecessors != [node]:
                    raise ValueError(
                        "The node %s has more than one parent, which is impossible for a child of a composition node (%s)"
                        % (str(successor), str(node))
                    )

            # successors = gh.get_all_successors(G, node)
            #            predecessors = gh.get_all_predecessors(G, node)

            # if not gh.is_it_a_partition(list(G.nodes), [successors, [node], predecessors]):
            #    raise ValueError("Incorrect split around composition node %s" % str(node))
            # _verif_split_is_everything(G,node,)


def add_columns_selector(Graph, var_type_node_dico, var_type_columns_dico, all_models_params):
    """ include columns selector where it is needed
    Either modify the graph by adding new edge with selector model
    Or modify all_models_params to include a 'columns_to_use' parameter
    
    Parameters
    ----------
    Graph : nx.DiGraph
        Graph representation the model
        
    var_type_node_dico : dict
        dictionnary indicating which node works on which type of columns.
        keys = node of Graph (ie : name of models)
        values = variable type
        
    var_type_columns_dico : dict
        dictionnary indicating which type of variable corresponds to which columns
        keys = variable type
        values = list of columns
        
    all_models_params : dict
        dictionnary indicating the parameter of each models, it will be modified to include 'columns_to_use'
        keys = node of Graph (ie: name of models)
        values = parameters of each models
        
        WILL BE MODIFIED in PLACE !
        
    Returns
    -------
    modified Graph
    modified all_params
    
    
    """

    nodes_no_composition = [n for n in Graph.nodes if not StepCategories.is_composition_step(n[0])]
    sub_Graph = Graph.subgraph(nodes_no_composition)

    starting_nodes = gh.get_starting_nodes(sub_Graph)

    for node in starting_nodes:
        vtype = var_type_node_dico[node]
        if _must_include_selector(node[1]):

            if vtype is not None:

                name_of_cat = "Selector_%s" % unnest_tuple(vtype)
                new_node = (name_of_cat, (name_of_cat, SpecialModels.ColumnsSelector))
                if new_node in Graph.nodes:
                    raise ValueError("Please check, I have duplicate names : %s" % str(new_node))

                Graph = gh.insert_node_above(Graph, node, new_node=new_node)

                all_models_params[new_node] = {"columns_to_use": _get_columns(vtype, var_type_columns_dico)}

            else:
                pass  # nothing to do : the transformer would need a selector BUT vtype is None which means I can apply to everything

        else:
            if vtype is not None:
                all_models_params[node]["columns_to_use"] = _get_columns(vtype, var_type_columns_dico)

    return Graph, all_models_params


def simplify_none_node(Graph):
    """ Remove the node where model_name is None from a Graph, those node are 'Passtrought' by convention """
    simplified_Graph = Graph.copy()
    while True:
        has_none_node = False
        for step_node, step_name in gh.iter_graph(simplified_Graph):
            if step_name[0] is None:
                has_none_node = True
                break

        if has_none_node:
            simplified_Graph = gh.remove_node_keep_connections(simplified_Graph, (step_node, step_name))
        else:
            break

    return simplified_Graph


def graphviz_modelgraph(G):
    """ create a graphviz Graph that can be plotted.
    
    Remark: graphviz can directly be displayed in an interactive environnement like IPython or Jupyter 
    """
    if graphviz is None:
        raise ValueError("You need to install graphviz")

    if isinstance(G, nx.DiGraph):
        G2 = graphviz.Digraph()
    else:
        G2 = graphviz.Graph()

    new_n = lambda x: (x[1][0], x[1][1])

    node_compo = []
    node_other = []
    for node in G.nodes:
        if StepCategories.is_composition_step(node[0]):
            node_compo.append(new_n(node))
        else:
            node_other.append(new_n(node))

    G2.attr("node", color="lightgreen")
    for node in node_compo:
        G2.node(str(node))

    G2.attr("node", color="lightblue")
    for node in node_other:
        G2.node(str(node))

    for e1, e2 in G.edges():
        if e1 in node_compo:
            G2.attr("edge", color="lightgreen", penwidth="1.0")
        else:
            G2.attr("edge", color="black", pendwidth="1.0")
        G2.edge(str(new_n(e1)), str(new_n(e2)))

    G2.node_attr.update(style="filled")

    return G2


def _create_name_mapping(all_nodes):
    """ helper function to creates the name of the nodes within a GraphPipeline model
    - if no ambiguities, name of node will be name of model
    - otherwise, name of node will be '%s_%s' % (name_of_step,name_of_model)
    
    
    Parameters
    ----------
    all_nodes : list of 2-uple
        nodes of graph : (step_name, model_name)
        
    Returns
    -------
    dictionnary with key = node, and value : string corresponding to the node
    """

    count_by_model_name = dict()
    mapping = {}
    done = set()
    for step_name, model_name in all_nodes:
        if (step_name, model_name) in done:
            raise ValueError("I have a duplicate node %s" % str((step_name, model_name)))
        done.add((step_name, model_name))

        count_by_model_name[model_name[1]] = count_by_model_name.get(model_name[1], 0) + 1

    for step_name, model_name in all_nodes:
        if count_by_model_name[model_name[1]] > 1:
            mapping[(step_name, model_name)] = "%s_%s" % (model_name[0], model_name[1])
        else:
            mapping[(step_name, model_name)] = model_name[1]

    count_by_name = dict()
    for k, v in mapping.items():
        count_by_name[k] = count_by_model_name.get(k, 0) + 1
    for k, v in count_by_name.items():
        if v > 1:
            raise ValueError("I have duplicate name for node %s" % str(k))

    return mapping


def _klass_from_node(node):
    """ retrieve the name of the class from the name of the node """
    return node[1][1]


def _find_first_composition_node(Graph, composition_already_done=None):
    """ retrieve the 'first' composition node of a Graph,
    it will ignore composition node already in 'composition_already_done'
    it no composition node, return None
    """
    if composition_already_done is None:
        composition_already_done = set()

    for node in gh.iter_graph(Graph):
        if StepCategories.is_composition_step(node[0]) and node not in composition_already_done:
            return node

    return None


def convert_graph_to_code(Graph, all_models_params, also_returns_mapping=False, _check_structure=True):
    """ convertion of a Graph representing a model into its json code 
    
    Parameter
    ---------
    
    Graph : nx.DirectGraph
        the graph of the model, each node as the form ( step, (step, klass) )
        
    all_models_params : dict
        hyperparameters of each model, keys = node of Graph, values = corresponding hyper-parameters
    
    also_returns_mapping : boolean, default = False
        if True will return a dictionnary with 'name_mapping' and 'json_code' as its key.
        So that the name in the GraphPipeline can be accessed 
        otherwise will just return the json_code
        
    Return
    ------
    
    a json-like object representing the model than can be translated into a model using 'sklearn_model_from_param'
        
    
    """

    if _check_structure:
        assert_model_graph_structure(Graph)

    models_dico = {node: (_klass_from_node(node), all_models_params[node]) for node in Graph.nodes}

    model_name_mapping = _create_name_mapping(Graph.nodes)

    rec_result = _rec_convert_graph_to_code(
        Graph=Graph, all_models_params=all_models_params, models_dico=models_dico, model_name_mapping=model_name_mapping
    )

    if not also_returns_mapping:
        return rec_result
    else:
        return {"name_mapping": model_name_mapping, "json_code": rec_result}


def _rec_convert_graph_to_code(
    Graph, all_models_params, models_dico, model_name_mapping=None, composition_already_done=None
):
    """ recursive function used to convert a Graph into a json code 
   
    See convert_graph_to_code
    """

    if composition_already_done is None:
        composition_already_done = set()

    if len(Graph.nodes) == 1:
        node = list(Graph.nodes)[0]
        return models_dico[node]

    node = _find_first_composition_node(Graph, composition_already_done)

    if node is not None:
        successors = list(Graph.successors(node))
        assert len(successors) > 0

    else:
        successors = []

    if node is None or len(successors) == 0:
        ### ** It's means I'll return a GraphPipeline ** ###
        # 2 cases :
        # * nodes is None  : meaning there is no composition node

        if len(successors) > 0:
            raise ValueError("a composition node should have at most one successor '%s'" % str(node))

        # assert len(successors) > 0

        # it shouldn't append ...
        # 1) either it an original node => composition node => no successor isn't possible
        # 2) the node was already handled => should have been in the list

        edges = gh.edges_from_graph(Graph)

        if model_name_mapping is None:
            model_name_mapping = _create_name_mapping(list(Graph.nodes))
        # each node in graph will be mapped to a name within the GraphPipeline

        models = {model_name_mapping[n]: models_dico[n] for n in Graph.nodes}

        edges = [tuple((model_name_mapping[e] for e in edge)) for edge in edges]

        return (SpecialModels.GraphPipeline, {"models": models, "edges": edges})

    composition_already_done.add(node)  # to prevent looping on the same node

    all_sub_branch_nodes = {}
    all_terminal_nodes = []
    for successor in successors:

        sub_branch_nodes = list(gh.subbranch_search(starting_node=successor, Graph=Graph, visited={node}))

        all_sub_branch_nodes[successor] = sub_branch_nodes

        assert successor in sub_branch_nodes

        sub_Graph = Graph.subgraph(sub_branch_nodes)

        all_terminal_nodes += gh.get_terminal_nodes(sub_Graph)

        models_dico[successor] = _rec_convert_graph_to_code(
            sub_Graph,
            all_models_params=all_models_params,
            models_dico=models_dico,
            model_name_mapping=model_name_mapping,
            composition_already_done=composition_already_done,
        )

    # Check
    all_s = [frozenset(Graph.successors(t_node)) for t_node in all_terminal_nodes]
    if len(set(all_s)) != 1:
        # By convention, if we look at the nodes AFTER the composition
        # (ie : the successors of the terminal nodes of the part of the graph that will be merged by the composition)
        # Those nodes should have the same list of successors. Those successors will be the successors of the merged node
        raise ValueError("The successor at the end of the composition node %s are not always the same" % str(node))

    if len(successors) == 1:

        # Only one sucessor of composition node

        models_dico[node] = (_klass_from_node(node), models_dico[successors[0]], all_models_params[node])

    elif len(successors) > 1:

        models_dico[node] = (
            _klass_from_node(node),
            [models_dico[successor] for successor in successors],
            all_models_params[node],
        )

    else:
        raise NotImplementedError("can't go there")

    # Now I need to merge 'node' with all the sub-branches
    nodes_mapping = {}
    for successor, sub_branch_nodes in all_sub_branch_nodes.items():
        for n in sub_branch_nodes:
            nodes_mapping[n] = node

    Gmerged = gh.merge_nodes(Graph, nodes_mapping=nodes_mapping)
    # All the node in successor will be 'fused' with 'node' ...
    # Recurse now, that the composition node is taken care of

    return _rec_convert_graph_to_code(
        Gmerged,
        all_models_params=all_models_params,
        models_dico=models_dico,
        model_name_mapping=model_name_mapping,
        composition_already_done=composition_already_done,
    )


# In[] : Old functions


def convert_graph_to_code_OLD(G, all_models_params):
    """ convertion of Graphical model into a json representation
    
    Parameters
    ----------
    G : nx.DiGraph
        graph representing the model, each node should be a 2-uple : (name_of_step,name_of_model)
        
    all_models_params : dict
        parameters of each models, key = node, value = dictionnary of hyper-parameters for this node
        
    Returns
    -------
    json like python object representing the model
    
    """
    all_params = {}
    for node in G.nodes:
        all_params[node] = (node[1][1], all_models_params.get(node, {}))

    assert_model_graph_structure(G)

    return _rec_convert_graph_to_code_OLD(G, all_params)


def _rec_convert_graph_to_code_OLD(G, all_params):
    """ recursive function to convert a graph into a json representation """
    if len(G.nodes) == 0:
        return {}

    ### 1) Find First composition node
    has_composition = False
    for node in gh.iter_graph(G):
        if StepCategories.is_composition_step(node[0]):
            has_composition = True
            break

    return_gpipe = not has_composition

    if has_composition:
        ### If there is a composition node, I need to split between what is above and what is bellow
        predecessors = gh.get_all_predecessors(G, node)
        successors = gh.get_all_successors(G, node)

        if not gh.is_it_a_partition(list(G.nodes), [predecessors, [node], successors]):
            raise ValueError("Incorrect graph, wrong split around node %s" % str(node))

        if len(successors) == 0:
            # If nothing bellow, I'll be able to return something
            return_gpipe = True

    if return_gpipe:

        if len(G.nodes) > 1:
            ### I'll create a GraphPipeline object

            edges = gh.edges_from_graph(G)

            model_name_mapping = _create_name_mapping(list(G.nodes))
            # each node in graph will be mapped to a name within the GraphPipeline

            models = {model_name_mapping[n]: all_params[n] for n in G.nodes}

            edges = [tuple((model_name_mapping[e] for e in edge)) for edge in edges]

            return (SpecialModels.GraphPipeline, {"models": models, "edges": edges})

        else:
            ### Otherwise it is just the model_name with its parameters
            return node[1][1], all_params[list(G.nodes)[0]]

    G_above = G.subgraph(predecessors + [node])
    G_bellow = G.subgraph(successors)

    connected_Gbellow = gh.get_connected_graphs(G_bellow)
    if len(connected_Gbellow) == 1:
        # what is bellow is a 'connected graph' : it means that the composition need should be applied to One model
        all_params[node] = _rec_convert_graph_to_code_OLD(G_bellow, all_params)

    else:
        # otherwise, the composition will be applied to a list of models
        all_params[node] = [_rec_convert_graph_to_code_OLD(g, all_params) for g in connected_Gbellow]

    return _rec_convert_graph_to_code_OLD(G_above, all_params)


def convert_graph_to_code_OLD2(Graph, all_models_params, also_returns_mapping=False):
    """ convertion of a Graph representing a model into its json code 
    
    Parameter
    ---------
    
    Graph : nx.DirectGraph
        the graph of the model, each node as the form ( step, (step, klass) )
        
    all_models_params : dict
        hyperparameters of each model, keys = node of Graph, values = corresponding hyper-parameters
    
    also_returns_mapping : boolean, default = False
        if True will return a dictionnary with 'name_mapping' and 'json_code' as its key.
        So that the name in the GraphPipeline can be accessed 
        otherwise will just return the json_code
        
    Return
    ------
    
    a json-like object representing the model than can be translated into a model using 'sklearn_model_from_param'
        
    
    """
    models_dico = {node: (_klass_from_node(node), all_models_params[node]) for node in Graph.nodes}

    model_name_mapping = _create_name_mapping(Graph.nodes)

    rec_result = _rec_convert_graph_to_code_OLD(
        Graph=Graph, all_models_params=all_models_params, models_dico=models_dico, model_name_mapping=model_name_mapping
    )

    if not also_returns_mapping:
        return rec_result
    else:
        return {"name_mapping": model_name_mapping, "json_code": rec_result}


def _rec_convert_graph_to_code_OLD2(Graph, all_models_params, models_dico, model_name_mapping=None):
    """ recursive function used to convert a Graph into a json code 
   
    See convert_graph_to_code
    """

    ### ** only one node in Graph : I'll return what was saved in models_dico ** ###
    if len(Graph.nodes) == 1:
        node = list(Graph.nodes)[0]
        return models_dico[node]

    node = _find_first_composition_node(Graph)

    if node is not None:
        predecessors = gh.get_all_predecessors(Graph, node)
        successors = gh.get_all_successors(Graph, node)

        if not gh.is_it_a_partition(list(Graph.nodes), [predecessors, [node], successors]):
            raise ValueError("Incorrect graph, wrong split around node %s" % str(node))
    else:
        predecessors = []
        successors = []

    if node is None or len(successors) == 0:
        ### ** It's means I'll return a GraphPipeline ** ###
        edges = gh.edges_from_graph(Graph)

        if model_name_mapping is None:
            model_name_mapping = _create_name_mapping(list(Graph.nodes))
        # each node in graph will be mapped to a name within the GraphPipeline

        models = {model_name_mapping[n]: models_dico[n] for n in Graph.nodes}

        edges = [tuple((model_name_mapping[e] for e in edge)) for edge in edges]

        return (SpecialModels.GraphPipeline, {"models": models, "edges": edges})

    Graph_bellow = Graph.subgraph(successors)

    connected_Gbellow = gh.get_connected_graphs(Graph_bellow)

    if len(predecessors) == 0 and len(connected_Gbellow) > 1:

        return (
            _klass_from_node(node),
            [
                _rec_convert_graph_to_code_OLD2(Gb, all_models_params, models_dico, model_name_mapping)
                for Gb in connected_Gbellow
            ],
            all_models_params[node],
        )

    elif len(predecessors) == 0 and len(connected_Gbellow) == 1:

        return (
            _klass_from_node(node),
            _rec_convert_graph_to_code_OLD2(Graph_bellow, all_models_params, models_dico, model_name_mapping),
            all_models_params[node],
        )

    else:

        G_bellow_and_node = Graph.subgraph([node] + successors)
        G_above = Graph.subgraph(predecessors + [node])

        models_dico[node] = _rec_convert_graph_to_code_OLD2(
            G_bellow_and_node, all_models_params, models_dico, model_name_mapping
        )

        return _rec_convert_graph_to_code(G_above, all_models_params, models_dico, model_name_mapping)
