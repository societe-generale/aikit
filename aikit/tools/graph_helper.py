# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 12:25:13 2018

@author: Lionel Massoulard
"""

import networkx as nx

import itertools

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("Matplotlib won't work")

try:
    import graphviz
except ImportError:
    graphviz = None
    print("Graphviz won't work")

from aikit.tools.helper_functions import lunique, unlist

# In[]
def add_node_after(G, new_node, *from_nodes, verbose=False):
    """ helper function to add a new node within a graph and place it after some nodes
    
    Parameters
    ----------
    G : nx.DiGraph
        the directed graph 
        
    new_node : name of the new node
    
    *from_node : if exists, parent of new_node
    
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
        print("add node '%s' " % new_node)

    elif verbose and len(from_nodes) == 1:
        print("add node '%s' <<--- '%s' " % (new_node, from_nodes[0]))

    elif verbose:
        print("add node '%s' <<--- %s" % (new_node, from_nodes))

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
    *edges : list or tuple
        each consecutive elements will be an edge

    Returns
    -------
    Direct Graph 
    
    """
    # Examples :
    # G = test_graph_from_edges((1,2,3),(4,3))
    # 1 -> 2 -> 3, and 3 -> 4

    G = nx.DiGraph()

    for list_of_edge in edges:

        if not isinstance(list_of_edge, (tuple, list)):
            raise TypeError("argument should be tuple or list, instead i got : '%s' " % type(list_of_edge))

        if len(list_of_edge) <= 1:
            G = add_node_after(G, list_of_edge[0])  # no edge, only a solo node
        else:
            for e1, e2 in zip(list_of_edge[:-1], list_of_edge[1:]):
                G = add_node_after(G, e2, e1)

    return G


def get_two_by_two_edges(*edges):
    """ create the list of edges
    
    Parameters
    ----------
    * edges : list or tuple
        each consecutive elements will be an edge
        
    Returns
    -------
    list of 2-uple for the edges
    """
    # Examples :
    # G = test_graph_from_edges((1,2,3),(4,3))
    # 1 -> 2 -> 3, and 3 -> 4

    two_by_two_edges = []

    for list_of_edge in edges:
        if len(list_of_edge) <= 1:
            raise ValueError("edges must be of length at least 2")

        if not isinstance(list_of_edge, (tuple, list)):
            raise TypeError("argument should be tuple or list, instead i got : '%s' " % type(list_of_edge))

        for e1, e2 in zip(list_of_edge[:-1], list_of_edge[1:]):
            two_by_two_edges.append((e1, e2))

    return two_by_two_edges


def is_it_a_partition(all_nodes, partitions):
    """ test if partitions is a partition of all_nodes.
    
    Parameters
    ----------
    * all_nodes : list of nodes
    * partitions : list of list of nodes
    
    Returns
    -------
    True if 'partitions' is a partitions of all_nodes, False otherwise
    """

    for nodes_subset in partitions:
        for node in nodes_subset:
            if node not in all_nodes:
                return False

    for nodes_subset1, nodes_subset2 in itertools.combinations(partitions, 2):
        if len(set.intersection(set(nodes_subset1), set(nodes_subset2))) > 0:
            return False

    s = set()
    for nodes_subset in partitions:
        s = set.union(s, set(nodes_subset))

    if s != set(all_nodes):
        return False

    return True


def get_terminal_nodes(G):
    """ retrieve terminal node of a Directed Graph """

    if not isinstance(G, nx.DiGraph):
        raise TypeError("I'm expecting a DiGraph, instead I got %s" % type(G))

    return [node for node in G.nodes() if len(list(G.successors(node))) == 0]


def get_all_successors(G, node):
    """ retrieve all the nodes bellow a given node """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("I'm expecting a DiGraph, instead I got %s" % type(G))

    if node not in G.nodes:
        raise ValueError("node %s not in graph" % node)

    successors = []

    new_nodes = [node]
    while True:
        new_successors = []
        for n in new_nodes:
            new_successors += G.successors(n)

        successors += new_successors
        if len(new_successors) > 0:
            new_nodes = new_successors
        else:
            break

    unique_successors = []
    for s in successors:
        if s not in unique_successors:
            unique_successors.append(s)

    return unique_successors


def get_all_predecessors(G, node):
    """ retrieve all the nodes above a given node """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("I'm expecting a DiGraph, instead I got %s" % type(G))

    if node not in G.nodes:
        raise ValueError("node %s not in graph" % node)

    predecessors = []

    new_nodes = [node]
    while True:
        new_predecessors = []
        for n in new_nodes:
            new_predecessors += G.predecessors(n)

        predecessors += new_predecessors
        if len(new_predecessors) > 0:
            new_nodes = new_predecessors
        else:
            break

    unique_predecessors = []
    for s in predecessors:
        if s not in unique_predecessors:
            unique_predecessors.append(s)

    return unique_predecessors


def insert_node_above(G, node, new_node):
    """ insert a new node just above a given node 
    
    
    Parameters
    ----------
    * G : Directed Graph
    the graph that will be modified
    
    * node : node of G
    the node above which we'll add a new node
    
    * new_node : name of new node
    
    Returns
    -------
    
    modified G
    """

    if not isinstance(G, nx.DiGraph):
        raise TypeError("I'm expecting a DiGraph")

    if new_node in G.nodes:
        raise ValueError("new_node %s shouldn't be in the graph already" % new_node)

    if node not in G.nodes:
        raise ValueError("node %s should be in the graph" % node)

    predeccessors = list(G.predecessors(node))
    for p in predeccessors:
        G.remove_edge(p, node)

    G.add_node(new_node)
    for p in predeccessors:
        G.add_edge(p, new_node)
    G.add_edge(new_node, node)

    return G


def remove_node_keep_connections(G, node):
    if node not in G.nodes:
        raise ValueError("The node %s isn't in the graph" % str(node))

    predeccessors = list(G.predecessors(node))
    successors = list(G.successors(node))

    # if len(predeccessors) > 1 and len(successors) > 1:
    #    raise ValueError("I can't remove a node that have more than 1 predeccessors and more than 1 successors")

    for p in predeccessors:
        G.remove_edge(p, node)

    for s in successors:
        G.remove_edge(node, s)

    for p in predeccessors:
        for s in successors:
            G.add_edge(p, s)

    G.remove_node(node)

    return G


def has_cycle(G):
    """ return True if the Graph as cycle """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("I'm expecting a DiGraph")

    try:
        nx.find_cycle(G)
    except nx.NetworkXNoCycle:
        return False

    return True


def get_starting_nodes(G):
    """ retrieve the starting node of a Directed Grap """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("I'm expecting a DiGraph")

    return [node for node in G.nodes() if len(list(G.predecessors(node))) == 0]


def get_connected_graphs(G):
    """ from a graph retrieve the list of subgraph, one for each connected components """
    connected_components = list(nx.connected_components(G.to_undirected()))
    if len(connected_components) == 1:
        return [G]
    else:
        all_sub_graphs = [G.__class__(G.subgraph(c)) for c in connected_components]
        return all_sub_graphs


def is_connected(G):
    """ returns True if the graph is connected """
    return nx.is_connected(G.to_undirected())


def iter_graph(G):
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


def graph_plot(G, ax=None, colors=None):
    """ plot a networkx graphical model """
    # TODO : use nice layout (GraphViz ?)
    if plt is None:
        raise ValueError("You need to install matplotlib")

    if ax is None:
        ax = plt.gca()
    pos = nx.spring_layout(G)  # positions for all nodes
    if colors is None:
        nx.draw(G, pos=pos, ax=ax)
    else:
        for k, v in colors.items():
            nx.draw(G, pos=pos, ax=ax, nodelist=v, node_color=k)

    nx.draw_networkx_labels(G, pos=pos, ax=ax)

    return ax


def graphviz_graph(G):
    """ create a graphviz Graph that can be plotted,
    
    Remark: graphviz can directly be displayed in an interactive environnement like IPython or Jupyter 
    """
    if graphviz is None:
        raise ValueError("You need to install graphviz")

    if isinstance(G, nx.DiGraph):
        G2 = graphviz.Digraph()
    else:
        G2 = graphviz.Graph()

    for e1, e2 in G.edges():
        G2.edge(str(e1), str(e2))

    G2.node_attr.update(color="lightblue", style="filled")
    return G2


def merge_nodes(Graph, nodes_mapping):
    """ helper function to merge node on a given Graph, the merge is done via a mapping of nodes
    
    Parameters
    ----------
    Graph : nx.DiGraph or nx.Graph
        the original graph
        
    nodes_mapping : dict
        mapping from 'node' to new node
        if a node is not in the mapping : no mapping is applied
        
    Returns
    -------
    newG : Graph of same type a G
        with nodes and edges passed by mapping
        
    If two nodes have the same mapping : they will be merged
    """
    if isinstance(Graph, nx.DiGraph):
        newG = nx.DiGraph()
    else:
        newG = nx.Graph()

    for node in Graph.nodes:
        mapped_node = nodes_mapping.get(node, node)  # default = no mapping
        if mapped_node not in newG.nodes:
            newG.add_node(mapped_node)  # Rmk : I drop other informarion in the graph

    for e1, e2 in Graph.edges:
        mapped_e1 = nodes_mapping.get(e1, e1)
        mapped_e2 = nodes_mapping.get(e2, e2)

        if mapped_e1 != mapped_e2 and (mapped_e1, mapped_e2) not in newG.edges:
            newG.add_edge(mapped_e1, mapped_e2)

    return newG


def subbranch_search(starting_node, Graph, yield_list=None, visited=None):
    """ breadth first search, starting from a node.
    Yielding only if : predecessor already visited.
    
    Parameters
    ----------
    
    starting_node : node
        the node to start
        
    Graph : nx.DiGraph
        The Graph
        
    yield_list : list of nodes or None
        if not None : will yield even if condition not verified
        
    visited : list of nodes or None
        already visited nodes
        
    Yields
    ------
    
    list of successives nodes in G

    
    """
    if starting_node not in Graph:
        raise ValueError("the node %s should be in graph" % str(starting_node))

    if yield_list is None:
        yield_list = set()
    if visited is None:
        visited = set()
    else:
        visited = set(visited)

    test_list = [starting_node]

    while True:
        new_test_list = []

        for node in test_list:
            if node in visited:
                continue

            predecessors = list(Graph.predecessors(node))

            # I yield if all the predecessor were visited already (OR if 'bypass' yield list)
            if node in yield_list or all((p in visited for p in predecessors)):
                yield node
                visited.add(node)
                new_test_list += list(Graph.successors(node))
            else:
                new_test_list.append(node)  # I'll try again at next iteration

        new_test_list = lunique(new_test_list)
        if len(new_test_list) == 0 or set(new_test_list) == set(test_list):
            # 1) no more node to test
            # 2) ... or same list as before
            return
        else:
            test_list = new_test_list
