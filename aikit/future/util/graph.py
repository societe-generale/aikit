import itertools

import networkx as nx

from .list import unlist


def add_node_after(G, new_node, *from_nodes, verbose=False):  # noqa
    """ helper function to add a new node within a graph and place it after some nodes

    Parameters
    ----------
    G : nx.DiGraph
        the directed graph

    new_node :
        name of the new node

    from_nodes :
        if exists, parent of new_node

    verbose : boolean
        if True will print what is done

    Returns
    -------
    modified Graph
    """
    G.add_node(new_node)
    for f in from_nodes:
        G.add_edge(f, new_node)

    if verbose and len(from_nodes) == 0:
        print(f"add node '{new_node}'")
    elif verbose and len(from_nodes) == 1:
        print(f"add node '{new_node}' <<--- '{from_nodes[0]}'")
    elif verbose:
        print(f"add node '{new_node}' <<--- {from_nodes}")

    return G


def edges_from_edges_string(edges_string):
    """Create a graph from a DOT string:

    * use : "A -> B -> C" to indicate edge between A and B and B and C
    * can also use '-' : "A - B"
    * group of edges separated by ';' : 'A - B - C ; D - C '

    A -> B

    Parameters
    ----------
    edges_string: string
        representation of the graph

    Returns
    -------
    corresponding list of edges
    """
    def try_int_convert(s):
        try:
            r = int(s)
        except ValueError:
            return s
        return r

    edges = [tuple(x.split("-")) for x in edges_string.replace(" ", "").replace("->", "-").split(";") if x != ""]
    edges = [tuple([try_int_convert(s) for s in x]) for x in edges]
    return edges


def edges_from_graph(G):
    """ return the edges from a graph """
    all_edges = list(sorted(set(G.edges)))  # to make sure the order of the edges doesn't change
    goon = True

    while goon:
        something_has_change = False
        for e1, e2 in itertools.product(all_edges, all_edges):
            if e1 != e2 and e1[-1] == e2[0]:
                all_edges = [e for e in all_edges if e != e1]
                all_edges = [e for e in all_edges if e != e2]
                all_edges.append(tuple(e1[0:-1]) + tuple(e2))

                something_has_change = True
            if something_has_change:
                break

        if not something_has_change:
            goon = False

    all_edges = list(all_edges)
    # Re-add node not in edges
    all_nodes_in_edge = unlist(all_edges)
    all_nodes = sorted(G.nodes)
    all_nodes = [n for n in all_nodes if n not in all_nodes_in_edge]
    all_edges += [(n,) for n in all_nodes]
    G2 = graph_from_edges(*all_edges)

    assert set(G.nodes) == set(G2.nodes)
    assert set(G.edges) == set(G2.edges)

    return all_edges


def graph_from_edges(*edges):
    """ create a graph from list of edges

    Parameters
    ----------
    edges : list or tuple
        each consecutive elements will be an edge

    Returns
    -------
    Direct Graph
    """
    # Examples :
    # G = test_graph_from_edges((1,2,3),(4,3))
    # 1 -> 2 -> 3, and 3 -> 4

    G = nx.DiGraph()  # noqa

    for list_of_edge in edges:

        if not isinstance(list_of_edge, (tuple, list)):
            raise TypeError(f"argument should be tuple or list, instead i got : '{type(list_of_edge)}'")

        if len(list_of_edge) <= 1:
            G = add_node_after(G, list_of_edge[0])  # no edge, only a solo node  # noqa
        else:
            for e1, e2 in zip(list_of_edge[:-1], list_of_edge[1:]):
                G = add_node_after(G, e2, e1)  # noqa

    return G


def graph_from_edges_string(edges_string):
    """ create a graph from a DOT string

    * use : "A -> B -> C" to indicate edge between A and B and B and C
    * can also use '-' : "A - B"
    * group of edges separated by ';' : 'A - B - C ; D - C '
    A -> B

    Parameters
    ----------
    edges_string : string
        representation of the graph

    Returns
    -------
    Direct Graph

    """
    return graph_from_edges(*edges_from_edges_string(edges_string))


def has_cycle(G):  # noqa
    """ return True if the Graph as cycle """
    if not isinstance(G, nx.DiGraph):
        raise TypeError(f"Input graph must be a DiGraph, got {type(G)}")

    try:
        nx.find_cycle(G)
    except nx.NetworkXNoCycle:
        return False

    return True


def iter_graph(G):  # noqa
    """ iterator over a DiGraph with following constraints :
    * all nodes passed exactly one
    * when a node is returned is predecessors were returned before

    # See verif
    """
    # TODO : pour optimiser la memoire dans les pipeline complique il faudrait essayer d'optimiser un peu l'ordre
    # On veut garder absolument le contrainte => noeud ressortis apres ses parents
    # Mais on voudrait en fait travailler 'branche' par 'branche'
    # Ex :
    # A -> B -> C -> D
    # E -> F -> G -> D
    # on voudrait plutot faire dans cet ordre
    # A, B,C puis E,F,G puis D

    # plutot que dans cet ordre
    # A,E, B,F ,C,G, D

    done = set()

    while True:
        something_remain = False

        for node in G.nodes():

            if node in done:
                continue

            something_remain = True

            all_done = True
            for predecessor in G.predecessors(node):
                if predecessor not in done:
                    all_done = False
                    break

            if not all_done:
                continue

            done.add(node)
            yield node

        if not something_remain:
            break
