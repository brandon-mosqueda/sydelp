from typing import Union

import networkx as nx

def create_random_geometric_graph(nodes_num: int,
                                  max_degree: int,
                                  edge_prob: float = 0.5,
                                  seed: Union[None, int] = None):
    # Create a random geometric graph
    graph: nx.Graph = nx.random_geometric_graph(nodes_num, edge_prob, seed=seed)
    mst: nx.Graph = nx.minimum_spanning_tree(graph)

    # Remove edges of both nodes exceeding the max_degree and without
    # disconecting the graph
    for node in list(graph.nodes):
        remove_num: int = graph.degree(node) - max_degree  # type: ignore

        for s, t in list(graph.edges(node)):
            if remove_num <= 0:
                break

            if (graph.degree(s) > max_degree # type: ignore
                and graph.degree(t) > max_degree  # type: ignore
                and (s, t) not in mst.edges):
                graph.remove_edge(s, t)
                remove_num -= 1

    # Remove to limit degree but without disconecting the result graph
    for node in list(graph.nodes):
        remove_num: int = graph.degree(node) - max_degree  # type: ignore

        for s, t in list(graph.edges(node)):
            if remove_num <= 0:
                break

            if (s, t) not in mst.edges:
                graph.remove_edge(s, t)
                remove_num -= 1

    return graph
