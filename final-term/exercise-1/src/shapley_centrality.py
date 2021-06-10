import typing
import networkx as nx
import itertools
import math

import characteristic_functions


def _shapley_value(graph: nx.Graph, characteristic_function: characteristic_functions.CharacteristicFunction, node: int) -> float:
    nodes_without_current = list(filter(lambda element: element != node, graph.nodes))

    value = 0
    for i in range(1, len(nodes_without_current) + 1):
        coalitions_with_i_nodes = itertools.combinations(nodes_without_current, i)
        for coalition in coalitions_with_i_nodes:
            weight = math.factorial(len(coalition)) * math.factorial(len(graph.nodes) - len(coalition) - 1) / math.factorial(len(graph.nodes))
            coalition_set = set(coalition)
            coalition_value_without_current = characteristic_function(graph, coalition_set)

            coalition_set.add(node)
            coalition_value_with_current = characteristic_function(graph, coalition_set)

            marginal_contribution_of_current = coalition_value_with_current - coalition_value_without_current

            value += weight * marginal_contribution_of_current

    return value


def naive_shapley_centrality(graph: nx.Graph, characteristic_function: characteristic_functions.CharacteristicFunction) -> typing.Dict[int, float]:
    shapley_values = {}
    for node in graph.nodes:
        shapley_values[node] = _shapley_value(graph, characteristic_function, node)
    return shapley_values

