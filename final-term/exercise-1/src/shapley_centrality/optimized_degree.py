import typing

import networkx as nx


def shapley_degree(graph: nx.Graph) -> typing.Dict[int, float]:
    values = {}

    degree_view = graph.degree
    for node, degree in degree_view:
        values[node] = 1 / (1 + degree)
        for neighbour in graph.neighbors(node):
            values[node] += 1 / (1 + degree_view[neighbour])

    return values