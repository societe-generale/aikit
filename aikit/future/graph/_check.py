from ..enums import StepCategory
from ..util import graph as gh


def is_composition_model(name):
    """ is it a composition model """
    return StepCategory.is_composition_step(name[0])


def assert_model_graph_structure(G):  # noqa
    """ verification on the structure of the graph """

    # only one terminal node
    if len(gh.get_terminal_nodes(G)) != 1:
        raise ValueError("The graph must have a terminal node")

    # connex graph
    if not gh.is_connected(G):
        raise ValueError("The graph must be connected")

    # no cycle
    if gh.has_cycle(G):
        raise ValueError("The graph must not have any cycle")

    for node in G.nodes:
        if is_composition_model(node):
            successors = list(G.successors(node))

            if len(successors) == 0:
                raise ValueError(f"Composition node {node} has no successor")

            for successor in successors:
                predecessors = list(G.predecessors(successor))
                if predecessors != [node]:
                    raise ValueError(f"The node {successor} has more than one parent, "
                                     f"which is impossible for a child of a composition node ({node})")
