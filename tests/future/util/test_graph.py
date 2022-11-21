import networkx as nx

from aikit.future.util.graph import iter_graph, has_cycle, add_node_after, graph_from_edges, graph_from_edges_string, \
    edges_from_graph


def test_edges_from_graph():
    G = nx.DiGraph()
    G.add_node("A")
    G.add_node("B")
    assert edges_from_graph(G) == [("A",), ("B",)]

    G = nx.DiGraph()
    G.add_node("B")
    G.add_node("A")
    assert edges_from_graph(G) == [("A",), ("B",)]

    G = graph_from_edges(("A", "B"), ("C", "D"))
    assert set(G.nodes) == {"A", "B", "C", "D"}
    assert set(G.edges) == {("A", "B"), ("C", "D")}
    assert edges_from_graph(G) == [("A", "B"), ("C", "D")]


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


def test_iter_graph():
    G1 = nx.DiGraph()
    G1 = add_node_after(G1, 1)
    G1 = add_node_after(G1, 2, 1)
    G1 = add_node_after(G1, 3, 2)
    G1 = add_node_after(G1, 4)
    G1 = add_node_after(G1, 5, 4)
    G1 = add_node_after(G1, 6, 5, 3)
    G2 = graph_from_edges((1, 2), (3, 4))

    for G in (G1, G2):
        done = set()
        for n in iter_graph(G):
            for p in G.predecessors(n):
                assert p in done

            assert n not in done
            done.add(n)

        assert done == set(G.nodes)

    G3 = nx.DiGraph()
    G3.add_node(1)
    G3.add_node(2)  # 2 unconnected nodes
    assert list(iter_graph(G3)) in ([1, 2], [2, 1])


def test_has_cycle():
    G = graph_from_edges_string("A - B - C; D - C")
    assert not has_cycle(G)

    G = graph_from_edges_string("A - B - C - A")
    assert has_cycle(G)
