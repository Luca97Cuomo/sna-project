import typing

import networkx as nx

from shapley_centrality import CentralityValues


def shapley_degree(graph: nx.Graph) -> CentralityValues:
    # O(n + m)
    # this is just a simplified version of shapley_threshold(graph, 1)
    values = {}

    degree_view = graph.degree
    for node, degree in degree_view:
        values[node] = 1 / (1 + degree)
        for neighbour in graph.neighbors(node):
            values[node] += 1 / (1 + degree_view[neighbour])

    return values
