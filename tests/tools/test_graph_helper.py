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
    add_node_after,
    iter_graph,
    get_all_predecessors,
    get_all_successors,
    has_cycle,
    remove_node_keep_connections,
    get_terminal_nodes,
    get_starting_nodes,
    get_connected_graphs,
    subbranch_search,
    merge_nodes,
    get_two_by_two_edges
)


def test_edges_from_graph():
    G = nx.DiGraph()
    G.add_node("A")
    G.add_node("B")
    
    assert edges_from_graph(G) == [("A",),("B",)]
    
    G = nx.DiGraph()
    G.add_node("B")
    G.add_node("A")
    assert edges_from_graph(G) == [("A",),("B",)]
    
    G = graph_from_edges(("A","B"),("C","D"))
    assert set(G.nodes) == {'A','B','C','D'}
    assert set(G.edges) == {('A','B'),('C','D')}
    
    assert edges_from_graph(G) == [('A','B'),('C','D')]

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
    G = graph_from_edges(*edges)
    assert set(G.nodes) == {"A", "B", "C", "D", "E"}
    assert set(G.edges) == {("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("E", "C")}


def test_get_two_by_two_edges():
    assert get_two_by_two_edges(("a","c"),("b","c")) == [('a', 'c'), ('b', 'c')]
    assert get_two_by_two_edges(("b","c"),("a","c")) == [('b', 'c'), ('a', 'c')]
    
    assert get_two_by_two_edges(("a","c","d"),("b","c")) == [('a', 'c'), ('c','d'),('b', 'c')]
    assert get_two_by_two_edges(("a","c","d"),("b","c")) == [('a', 'c'), ('c','d'),('b', 'c')]

    edges = [("A",),("B","C","D")]
    G = graph_from_edges(*edges)
    assert set(G.nodes) == {"A","B","C","D"}
    assert set(G.edges) == {("B","C"),("C","D")}

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
    G1 = nx.DiGraph()
    G1 = add_node_after(G1, 1)
    G1 = add_node_after(G1, 2, 1)
    G1 = add_node_after(G1, 3, 2)
    G1 = add_node_after(G1, 4)
    G1 = add_node_after(G1, 5, 4)
    G1 = add_node_after(G1, 6, 5, 3)

    G2 = graph_from_edges((1,2),(3,4))
    
    for G in (G1,G2):
        done = set()
        for n in iter_graph(G):
            for p in G.predecessors(n):
                assert p in done
    
            assert n not in done
            done.add(n)
    
        assert done == set(G.nodes)

    G3 = nx.DiGraph()
    G3.add_node(1)
    G3.add_node(2)  # 2 un-connexted nodes
    assert list(iter_graph(G3)) in ([1, 2],[2,1])


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


def test_subbranch_search():
    
    Graph = graph_from_edges(("A","B","C"),("B","D"))

    nodes = list(subbranch_search("A",Graph))
    assert nodes == ['A','B','C','D'] or nodes == ['A','B','D','C']

    nodes = list(subbranch_search("B",Graph))
    assert nodes == []
    
    nodes = list(subbranch_search("B",Graph, visited={"A"}))
    assert nodes == ['B','C','D'] or nodes == ['B','D','C']
    
    nodes = list(subbranch_search("C",Graph, visited={"B"}))
    assert nodes == ["C"]
  
    nodes = list(subbranch_search("D",Graph, visited={"B"}))
    assert nodes == ["D"]
    
    # Graph2
    Graph = graph_from_edges(("A","B","C","E"),("B","D","E"))
    nodes = list(subbranch_search("A",Graph))
    assert nodes == ['A','B','C','D','E'] or nodes == ['A','B','D','C','E']
    
    Graph = graph_from_edges(("A","B","C","E"),("B","D","E"))
    nodes = list(subbranch_search("B",Graph, visited = {"A"}))
    assert nodes == ['B','C','D','E'] or nodes == ['B','D','C','E']
    
    Graph = graph_from_edges(("A","B","C","E"),("B","D","E"))
    nodes = list(subbranch_search("C",Graph, visited = {"B"}))
    assert nodes == ['C']
    
    Graph = graph_from_edges(("A","B","C"),("A","C"))
    nodes = list(subbranch_search("A",Graph))
    assert nodes == ["A","B","C"]
    
    nodes = list(subbranch_search("B",Graph, visited={"A"}))
    assert nodes == ['B', 'C']
    
    nodes = list(subbranch_search("C",Graph, visited={"B"}))
    assert nodes == []

def test_merge_nodes():
    Graph = graph_from_edges(("A","B","C"),("B","D"))
    
    mGraph = merge_nodes(Graph,{"B":"A"})
    assert set(mGraph.nodes) == {"A","C","D"}
    assert set(mGraph.edges) == {("A","C"),("A","D")}
    # check Graph wasn't modified
    assert set(Graph.nodes) == {"A","B","C","D"}
    
    mGraph = merge_nodes(Graph,{"C":"B","D":"B"})
    assert set(mGraph.nodes) == {"A","B"}
    assert set(mGraph.edges) == {('A', 'B')}
    