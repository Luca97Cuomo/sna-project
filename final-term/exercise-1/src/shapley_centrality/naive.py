import typing
import networkx as nx
import itertools
import math

import characteristic_functions
from shapley_centrality import ShapleyValues

ShapleyValueFunction = typing.Callable[[nx.Graph, characteristic_functions.CharacteristicFunction, int], float]


def shapley_value_combinations(graph: nx.Graph,
                               characteristic_function: characteristic_functions.CharacteristicFunction,
                               node: int) -> float:
    nodes_without_current = list(filter(lambda element: element != node, graph.nodes))

    value = 0
    for i in range(0, len(nodes_without_current) + 1):
        coalitions_with_i_nodes = itertools.combinations(nodes_without_current, i)
        for coalition in coalitions_with_i_nodes:
            weight = math.factorial(len(coalition)) * math.factorial(len(graph.nodes) - len(coalition) - 1) \
                     / math.factorial(len(graph.nodes))
            coalition_set = set(coalition)
            coalition_value_without_current = characteristic_function(graph, coalition_set)

            coalition_set.add(node)
            coalition_value_with_current = characteristic_function(graph, coalition_set)

            marginal_contribution_of_current = coalition_value_with_current - coalition_value_without_current

            value += weight * marginal_contribution_of_current

    return value


def shapley_value_permutations(graph: nx.Graph, characteristic_function: characteristic_functions.CharacteristicFunction, node: int) -> float:
    permutations = itertools.permutations(graph.nodes())

    value = 0
    for permutation in permutations:
        # evaluate permutation without current node
        coalition = set()
        for i in range(len(permutation)):
            if permutation[i] == node:
                break
            else:
                coalition.add(permutation[i])

        # check if we have to continue or not
        # if len(coalition) == 0:
        #    continue

        coalition_value_without_current = characteristic_function(graph, coalition)

        coalition.add(node)
        coalition_value_with_current = characteristic_function(graph, coalition)

        marginal_contribution_of_current = coalition_value_with_current - coalition_value_without_current

        value += marginal_contribution_of_current

    return value / math.factorial(len(graph.nodes))


def naive_shapley_centrality(graph: nx.Graph, characteristic_function: characteristic_functions.CharacteristicFunction,
                             shapley_value_function: ShapleyValueFunction) -> ShapleyValues:
    shapley_values = {}
    for node in graph.nodes:
        shapley_values[node] = shapley_value_function(graph, characteristic_function, node)
    return shapley_values

