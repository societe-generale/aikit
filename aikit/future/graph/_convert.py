from ._check import assert_model_graph_structure
from ..enums import SpecialModels, StepCategory
from ..util import graph as gh


def _klass_from_node(node):
    """ retrieve the name of the class from the name of the node """
    return node[1][1]


def _find_first_composition_node(G, composition_already_done=None):  # noqa
    """ retrieve the 'first' composition node of a Graph,
    it will ignore composition node already in 'composition_already_done'
    it no composition node, return None
    """
    if composition_already_done is None:
        composition_already_done = set()

    for node in gh.iter_graph(G):
        if StepCategory.is_composition_step(node[0]) and node not in composition_already_done:
            return node

    return None


def _create_name_mapping(all_nodes):
    """ Helper function to create the name of the nodes within a GraphPipeline model.
        - if no ambiguities, name of node will be the name of the model
        - otherwise, name of node will be '%s_%s' % (name_of_step, name_of_model)

    Parameters
    ----------
    all_nodes : list of 2-tuple
        nodes of graph : (step_name, model_name)

    Returns
    -------
    dictionary with key = node, and value : string corresponding to the node
    """
    count_by_model_name = {}
    mapping = {}
    done = set()
    for step_name, model_name in all_nodes:
        if (step_name, model_name) in done:
            raise ValueError("I have a duplicate node %s" % str((step_name, model_name)))
        done.add((step_name, model_name))
        count_by_model_name[model_name[1]] = count_by_model_name.get(model_name[1], 0) + 1

    for step_name, model_name in all_nodes:
        if count_by_model_name[model_name[1]] > 1:
            mapping[(step_name, model_name)] = f"{model_name[0]}_{model_name[1]}"
        else:
            mapping[(step_name, model_name)] = model_name[1]

    count_by_name = {}
    for k, v in mapping.items():
        count_by_name[k] = count_by_model_name.get(k, 0) + 1
    for k, v in count_by_name.items():
        if v > 1:
            raise ValueError(f"Found duplicate name for node {k}")

    return mapping


def convert_graph_to_code(graph, all_models_params, return_mapping=False, check_structure=True):
    """ Conversion of a graph representing a model into its json code 

    Parameter
    ---------
    graph : nx.DiGraph
        the graph of the model, each node as the form (step, (step, klass))

    all_models_params : dict
        hyperparameters of each model, keys = node of graph, values = corresponding hyperparameters

    returns_mapping : boolean, default = False
        if True will return a dictionary with 'name_mapping' and 'json_code' as its key.
        So that the name in the graphPipeline can be accessed 
        otherwise will just return the json_code
        
    check_structure: boolean
        if True runs a check on the graph structure
        
    Return
    ------
    a json-like object representing the model than can be translated into a model using 'sklearn_model_from_param'
    """
    if check_structure:
        assert_model_graph_structure(graph)

    models_dico = {node: (_klass_from_node(node), all_models_params[node]) for node in graph.nodes}

    model_name_mapping = _create_name_mapping(graph.nodes)

    rec_result = _rec_convert_graph_to_code(graph=graph,
                                            all_models_params=all_models_params,
                                            models_dico=models_dico,
                                            model_name_mapping=model_name_mapping)

    if not return_mapping:
        return rec_result
    else:
        return {"name_mapping": model_name_mapping, "json_code": rec_result}


def _rec_convert_graph_to_code(graph,
                               all_models_params,
                               models_dico,
                               model_name_mapping=None,
                               composition_already_done=None):
    """ recursive function used to convert a graph into a json code 
    See convert_graph_to_code
    """
    if composition_already_done is None:
        composition_already_done = set()

    if len(graph.nodes) == 1:
        node = list(graph.nodes)[0]
        return models_dico[node]

    node = _find_first_composition_node(graph, composition_already_done)

    if node is not None:
        successors = list(graph.successors(node))
        assert len(successors) > 0

    else:
        successors = []

    if node is None or len(successors) == 0:
        # Return a GraphPipeline object
        # 2 cases :
        #   * nodes is None : meaning there is no composition node
        if len(successors) > 0:
            raise ValueError("a composition node should have at most one successor '%s'" % str(node))

        # It shouldn't happen
        #   1) either it an original node => composition node => no successor is not possible
        #   2) the node was already handled => should have been in the list
        edges = gh.edges_from_graph(graph)

        if model_name_mapping is None:
            # each node in graph will be mapped to a name within the GraphPipeline
            model_name_mapping = _create_name_mapping(list(graph.nodes))

        models = {model_name_mapping[n]: models_dico[n] for n in graph.nodes}
        edges = [tuple((model_name_mapping[e] for e in edge)) for edge in edges]
        return SpecialModels.GraphPipeline, {"models": models, "edges": edges}

    composition_already_done.add(node)  # to prevent looping on the same node

    all_sub_branch_nodes = {}
    all_terminal_nodes = []
    for successor in successors:
        sub_branch_nodes = list(gh.subbranch_search(starting_node=successor, G=graph, visited=[node]))
        all_sub_branch_nodes[successor] = sub_branch_nodes
        assert successor in sub_branch_nodes

        sub_graph = graph.subgraph(sub_branch_nodes)
        all_terminal_nodes += gh.get_terminal_nodes(sub_graph)

        models_dico[successor] = _rec_convert_graph_to_code(
            sub_graph,
            all_models_params=all_models_params,
            models_dico=models_dico,
            model_name_mapping=model_name_mapping,
            composition_already_done=composition_already_done)

    # Check
    all_s = [frozenset(graph.successors(t_node)) for t_node in all_terminal_nodes]
    if len(set(all_s)) != 1:
        # By convention, if we look at the nodes AFTER the composition
        # (ie : the successors of the terminal nodes of the part of the graph that will be merged by the composition)
        # Those nodes should have the same list of successors.
        # Those successors will be the successors of the merged node.
        raise ValueError(f"The successor at the end of the composition node {node} are not always the same")

    if len(successors) == 1:
        # Only one successor of composition node
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

    graph_merged = gh.merge_nodes(graph, nodes_mapping=nodes_mapping)
    # All the node in successor will be 'fused' with 'node' ...
    # Recurse now, that the composition node is taken care of

    return _rec_convert_graph_to_code(
        graph_merged,
        all_models_params=all_models_params,
        models_dico=models_dico,
        model_name_mapping=model_name_mapping,
        composition_already_done=composition_already_done)
