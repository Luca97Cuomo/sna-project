import networkx as nx

from shapley_centrality import ShapleyValues


def shapley_threshold(graph: nx.Graph, k: int) -> ShapleyValues:
    # O(n + m)
    values = {}

    degree_view = graph.degree
    for node, degree in degree_view:
        values[node] = min(1, k / (1 + degree))
        for neighbour in graph.neighbors(node):
            neighbour_degree = degree_view[neighbour]
            values[node] += max(0, (neighbour_degree - k + 1) / (neighbour_degree * (1 + neighbour_degree)))

    return values
