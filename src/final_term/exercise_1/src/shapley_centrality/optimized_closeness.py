import typing

import networkx as nx

from final_term.exercise_1.src.shapley_centrality.characteristic_functions import adjusted_distance_reciprocal
from final_term.exercise_1.src.shapley_centrality import CentralityValues


def shapley_closeness(graph: nx.Graph,
                      decreasing_distance_function: typing.Callable[[float], float] = None) -> CentralityValues:
    # O(nm + n^2log n)

    if decreasing_distance_function is None:
        decreasing_distance_function = adjusted_distance_reciprocal

    values = {node: 0 for node in graph.nodes}
    for node in graph.nodes:
        # from python 3.6 the dict mantains the insertion order
        descending_distances = {n: d for n, d in
                                sorted(nx.shortest_path_length(graph, node).items(), key=lambda item: -item[1])}
        sum = 0
        previous_distance = -1
        previous_shapley = -1
        index = len(graph.nodes) - 1
        for target_node, distance in descending_distances.items():
            distance_value = decreasing_distance_function(distance)
            if distance == previous_distance:
                current_shapley = previous_shapley
            else:
                current_shapley = (distance_value / (1 + index)) - sum

            values[target_node] += current_shapley
            sum += distance_value / (index * (1 + index))

            previous_distance = distance
            previous_shapley = current_shapley
            index -= 1

            # The last element of descending_distances is the current node itself
            if index == 0:
                break

        values[node] += decreasing_distance_function(0) - sum

    return values
