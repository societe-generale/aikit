# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:24:48 2018

@author: Lionel Massoulard
"""

import graphviz
import networkx as nx

from aikit.tools.graph_helper import (
    graph_from_edges_string,
    graph_from_edges,
    edges_from_graph,
    is_it_a_partition,
    insert_node_above,
)
from aikit.tools.graph_helper import (
    add_node_after,
    iter_graph,
    get_all_predecessors,
    get_all_successors,
    has_cycle,
    remove_node_keep_connections,
)
from aikit.tools.graph_helper import get_terminal_nodes, get_starting_nodes, get_connected_graphs


def test_graph_from_edges_string():

    edges_strings = [
        "A -  B - C  ; D - C",
        "A - B ; B - C ; D - C",
        "A -> B -> C ; D -> C",
        "A-B;B-C;D-C",
        "A->B;B->C;D-C;",
    ]

    for edges_string in edges_strings:
        G = graph_from_edges_string(edges_string)
        assert set(G.nodes) == {"A", "B", "C", "D"}
        assert set(G.edges) == {("A", "B"), ("B", "C"), ("D", "C")}

        edges2 = edges_from_graph(G)
        G2 = graph_from_edges(*edges2)

        assert set(G2.nodes) == set(G.nodes)
        assert set(G2.edges) == set(G.edges)

    edges_strings = ["1 - 2 - 3 ; 4 - 3", "1 - 2 ; 2 - 3 ; 4 - 3"]

    for edges_string in edges_strings:
        G = graph_from_edges_string(edges_string)
        assert set(G.nodes) == {1, 2, 3, 4}
        assert set(G.edges) == {(1, 2), (2, 3), (4, 3)}

        edges2 = edges_from_graph(G)
        G2 = graph_from_edges(*edges2)

        assert set(G2.nodes) == set(G.nodes)
        assert set(G2.edges) == set(G.edges)


def test_is_it_a_partition():
    all_nodes = [1, 2, 3, 4, 5, 6]
    assert is_it_a_partition(all_nodes, [[1], [2], [3, 4, 5, 6]])
    assert is_it_a_partition(all_nodes, [[1, 2], [3, 4, 5, 6]])
    assert is_it_a_partition(all_nodes, [[1, 2, 3, 4, 5, 6], []])

    assert not is_it_a_partition(all_nodes, [[1, 2, 3, 4, 5]])
    assert not is_it_a_partition(all_nodes, [[1, 2, 3, 4], [5]])

    assert not is_it_a_partition(all_nodes, [[1, 2, 3, 4], [5, 6, 7]])
    assert not is_it_a_partition(all_nodes, [[1, 2, 3, 4, 5], [5, 6]])


def test_graph_from_edges():

    edges = [(1, 2, 3), (4, 3)]
    G = graph_from_edges(*edges)
    assert set(G.nodes) == {1, 2, 3, 4}
    assert set(G.edges) == {(1, 2), (2, 3), (4, 3)}

    edges = [("1", "2", "3"), ("4", "3")]
    G = graph_from_edges(*edges)
    assert set(G.nodes) == {"1", "2", "3", "4"}
    assert set(G.edges) == {("1", "2"), ("2", "3"), ("4", "3")}

    edges = [("A", "B", "C")]
    G = graph_from_edges(*edges)
    assert set(G.nodes) == {"A", "B", "C"}
    assert set(G.edges) == {("A", "B"), ("B", "C")}

    edges = [("A", "B", "D"), ("A", "C", "D"), ("E", "C")]
    G = graph_from_edges(*edges)
    assert set(G.nodes) == {"A", "B", "C", "D", "E"}
    assert set(G.edges) == {("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("E", "C")}

    edges = [("A", "B", "D"), ("A", "C", "D"), ("E", "C", "D")]
    assert set(G.nodes) == {"A", "B", "C", "D", "E"}
    assert set(G.edges) == {("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("E", "C")}


def test_insert_node_above():
    G = graph_from_edges_string("1 - 2 - 3")
    G2 = insert_node_above(G, 1, 0)

    assert G is G2
    assert set(G2.nodes) == {0, 1, 2, 3}
    assert set(G2.edges) == {(0, 1), (1, 2), (2, 3)}

    G = insert_node_above(G, 2, 22)
    assert set(G.nodes) == {0, 1, 2, 3, 22}
    assert set(G.edges) == {(0, 1), (1, 22), (22, 2), (2, 3)}

    G = graph_from_edges_string("1 - 2 - 3; 4 - 3")
    G = insert_node_above(G, 3, 33)
    assert set(G.edges) == {(1, 2), (2, 33), (4, 33), (33, 3)}
    assert set(G.nodes) == {1, 2, 3, 4, 33}


def test_remove_node_keep_connections():
    G = graph_from_edges_string("A - B - C; D - C")
    G = remove_node_keep_connections(G, "B")

    assert set(G.edges) == {("A", "C"), ("D", "C")}
    assert set(G.nodes) == {"A", "C", "D"}

    G = graph_from_edges_string("A - B - C; D - C")
    G = remove_node_keep_connections(G, "A")
    assert set(G.edges) == {("B", "C"), ("D", "C")}
    assert set(G.nodes) == {"B", "C", "D"}

    G = graph_from_edges_string("A - B - C - E; D - C - E")
    G = remove_node_keep_connections(G, "C")
    assert set(G.edges) == {("A", "B"), ("B", "E"), ("D", "E")}
    assert set(G.nodes) == {"A", "B", "D", "E"}

    G = graph_from_edges_string("A - B - C - D; C - E")
    G = remove_node_keep_connections(G, "C")
    assert set(G.edges) == {("A", "B"), ("B", "D"), ("B", "E")}
    assert set(G.nodes) == {"A", "B", "D", "E"}

    G = graph_from_edges_string("A - C - D; B - C - E")
    G = remove_node_keep_connections(G, "C")

    assert set(G.edges) == {("A", "D"), ("A", "E"), ("B", "D"), ("B", "E")}
    assert set(G.nodes) == {"A", "B", "D", "E"}


def test_all_graphs_functions():
    G = nx.DiGraph()
    G = add_node_after(G, 1)
    G = add_node_after(G, 2, 1)
    G = add_node_after(G, 3, 2)
    G = add_node_after(G, 4)
    G = add_node_after(G, 5, 4)
    G = add_node_after(G, 6, 5, 3)

    assert set(get_terminal_nodes(G)) == {6}
    assert set(get_starting_nodes(G)) == {1, 4}

    assert set(get_all_successors(G, 1)) == {2, 3, 6}
    assert set(get_all_successors(G, 2)) == {3, 6}
    assert set(get_all_successors(G, 3)) == {6}
    assert set(get_all_successors(G, 4)) == {5, 6}
    assert set(get_all_successors(G, 5)) == {6}
    assert set(get_all_successors(G, 6)) == set()

    assert set(get_all_predecessors(G, 1)) == set()
    assert set(get_all_predecessors(G, 2)) == {1}
    assert set(get_all_predecessors(G, 3)) == {1, 2}
    assert set(get_all_predecessors(G, 4)) == set()
    assert set(get_all_predecessors(G, 5)) == {4}
    assert set(get_all_predecessors(G, 6)) == {1, 2, 3, 4, 5}


def test_iter_graph():
    G = nx.DiGraph()
    G = add_node_after(G, 1)
    G = add_node_after(G, 2, 1)
    G = add_node_after(G, 3, 2)
    G = add_node_after(G, 4)
    G = add_node_after(G, 5, 4)
    G = add_node_after(G, 6, 5, 3)

    done = set()
    for n in iter_graph(G):
        for p in G.predecessors(n):
            assert p in done

        assert n not in done
        done.add(n)

    assert done == set(G.nodes)


def test_graphviz():
    G2 = graphviz.Digraph()
    G2.edge("a", "b")
    G2.edge("b", "c")
    G2.edge("c", "f")
    G2.edge("d", "e")
    G2.edge("e", "f")
    G2.node_attr.update(color="lightblue", style="filled")

    G2


def test_has_cycle():
    G = graph_from_edges_string("A - B - C; D - C")
    assert not has_cycle(G)

    G = graph_from_edges_string("A - B - C - A")
    assert has_cycle(G)


def test_get_connected_graph():
    G = graph_from_edges_string("A - B ; B - C")
    Gs = get_connected_graphs(G)
    assert len(Gs) == 1
    assert Gs[0] == G

    def gequal(g1, g2):
        if set(g1.nodes) != set(g2.nodes):
            return False

        if set(g1.edges) != set(g2.edges):
            return False

        return True

    G = graph_from_edges_string("A - B; C - D")
    Gs = get_connected_graphs(G)
    assert len(Gs) == 2
    g1, g2 = graph_from_edges_string("A - B"), graph_from_edges_string("C - D")
    G1, G2 = Gs
    assert (gequal(g1, G1) and gequal(g2, G2)) or (gequal(g1, G2) and gequal(g2, G1))


def verif_all():
    test_iter_graph()
    test_graph_from_edges_string()
    test_graph_from_edges()
    test_all_graphs_functions()
    test_remove_node_keep_connections()
    test_has_cycle()
    test_is_it_a_partition()
    test_insert_node_above()
    test_graphviz()
    test_get_connected_graph()
