from collections import OrderedDict

import networkx as nx

from ._check import assert_model_graph_structure
from ..automl import MODEL_REGISTRY
from ..enums import StepCategory, SpecialModels
from ..util import graph as gh
from ..util.list import unnest_tuple, tuple_include, diff


def create_graphical_representation(steps):
    """ from an OrderedDict of steps create a Graphical representation of the model
    TODO add more documentation (with graphical examples)
    """

    # Remark:
    # We need to put the number of step in the graph + labels
    # In that case we can have several nodes with same name (ex: Scaler)

    # 1) Split composition Steps vs Rest
    all_composition_steps = []
    all_others = []
    for (step_name, model_name), var_type in steps.items():
        if StepCategory.is_composition_step(step_name):
            all_composition_steps.append((step_name, model_name, var_type))
        else:
            all_others.append((step_name, model_name, var_type))

    # 2) Create Graph for non-composition step
    new_steps = OrderedDict()

    G = nx.DiGraph()  # noqa
    for step_name, model_name, var_type in all_others:
        unnested_var_type = unnest_tuple(var_type)
        terminal_nodes = gh.get_terminal_nodes(G)  # Terminal links : I'll add the new step on one (or more) of those
        ending_node_type = {unnest_tuple(steps[node]): node for node in terminal_nodes}
        node_name = (step_name, model_name)  # 2-tuple
        if node_name in G.nodes:
            raise ValueError(f"This node already exists '({node_name})'")

        # 1) New node is attached to a terminal node
        # 2) Create a new branch (new node attached to 'nothing')
        # 3) New node attached to multiple terminal nodes
        elif unnested_var_type in ending_node_type:
            # 1) A branch if this type already exists
            last_node = ending_node_type[unnested_var_type]
            G = gh.add_node_after(G, node_name, last_node)  # noqa

        # No branch
        else:
            # Look candidates where to plug the branch
            all_candidates = [(t, n) for t, n in ending_node_type.items() if tuple_include(t, unnested_var_type)]
            if len(all_candidates) == 0:
                # 2) Create a new branch with no nodes
                G = gh.add_node_after(G, node_name)  # noqa
            else:
                # 3) Attach node to multiple terminal nodes
                # If some types are not added, add a node
                types_added = unnest_tuple([t for t, n in all_candidates])
                types_not_added = diff(unnested_var_type, types_added)
                if len(types_not_added) > 0:
                    name_of_cat = f"Selector_{unnest_tuple(types_not_added)}"
                    new_node = (name_of_cat, (name_of_cat, SpecialModels.ColumnsSelector))
                    G = gh.add_node_after(G, new_node)  # noqa
                    new_steps[new_node] = types_not_added  # I also must dynamically add the node to the list of steps
                    all_candidates = all_candidates + [(types_not_added, new_node)]

                G = gh.add_node_after(G, node_name, *[n for t, n in all_candidates])  # noqa

    # 3) Include composition node on top
    for step_name, model_name, _ in reversed(all_composition_steps):
        starting_nodes = gh.get_starting_nodes(G)
        for n in starting_nodes:
            G.add_edge((step_name, model_name), n)

    # 4) Verify the Graph structure
    for (step_name, model_name), _ in steps.items():
        if (step_name, model_name) not in G:
            raise ValueError(f"'({step_name} , {model_name})' should be in graph")
    # all nodes were in the steps
    for node in G.nodes():
        if node not in steps and node not in new_steps:
            raise ValueError(f"'({node})' shouldn't be in graph")

    assert_model_graph_structure(G)

    return G, new_steps


def simplify_none_node(G):  # noqa
    """ Remove the node where model_name is None from a Graph, those node are 'Passthrough' by convention """
    simplified_graph = G.copy()
    while True:
        has_none_node = False
        step_node = step_name = None
        for step_node, step_name in gh.iter_graph(simplified_graph):
            if step_name[0] is None:
                has_none_node = True
                break
        if has_none_node:
            simplified_graph = gh.remove_node_keep_connections(simplified_graph, (step_node, step_name))
        else:
            break

    return simplified_graph


def _get_columns(var_type, var_type_columns_dico):
    """ function to get the columns

    Parameters
    ----------
    var_type : tuple or str
        the type of variable we want (one type : string, several : tuple)

    var_type_columns_dico : dict
        dictionary with keys = type of variable and values = name of columns

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


def _must_include_selector(name):
    """ function that tells if I need to include a selector before a given model
    it is using the init_params in MODEL_REGISTER,
    by default I'll include a selector
    """
    if name is None:
        return True

    if name[1] == SpecialModels.ColumnsSelector:
        return False

    return "columns_to_use" not in MODEL_REGISTRY.init_parameters.get(name, {})


def add_columns_selector(graph, var_type_node_dico, var_type_columns_dico, all_models_params):
    """ include columns selector where it is needed
    Either modify the graph by adding new edge with selector model
    Or modify all_models_params to include a 'columns_to_use' parameter

    Parameters
    ----------
    graph : nx.DiGraph
        Graph representation the model

    var_type_node_dico : dict
        dictionary indicating which node works on which type of columns.
        keys = node of Graph (ie : name of models)
        values = variable type

    var_type_columns_dico : dict
        dictionary indicating which type of variable corresponds to which columns
            key = variable type
            value = list of columns

    all_models_params : dict
        dictionary indicating the parameter of each model, it will be modified to include 'columns_to_use'
            key = node of Graph (ie: name of models)
            value = parameters of each models

        WILL BE MODIFIED in PLACE !

    Returns
    -------
    modified Graph
    modified all_params
    """
    nodes_no_composition = [n for n in graph.nodes if not StepCategory.is_composition_step(n[0])]
    sub_graph = graph.subgraph(nodes_no_composition)

    starting_nodes = gh.get_starting_nodes(sub_graph)

    for node in starting_nodes:
        vtype = var_type_node_dico[node]
        if _must_include_selector(node[1]):
            if vtype is not None:
                name_of_cat = f"Selector_{unnest_tuple(vtype)}"
                new_node = (name_of_cat, (name_of_cat, SpecialModels.ColumnsSelector))
                if new_node in graph.nodes:
                    raise ValueError("Please check, I have duplicate names : %s" % str(new_node))
                graph = gh.insert_node_above(graph, node, new_node=new_node)
                all_models_params[new_node] = {"columns_to_use": _get_columns(vtype, var_type_columns_dico)}
            else:
                # nothing to do: the transformer would need a selector
                # BUT vtype is None which means I can apply to everything
                pass
        else:
            if vtype is not None:
                all_models_params[node]["columns_to_use"] = _get_columns(vtype, var_type_columns_dico)

    return graph, all_models_params
