import copy
import typing
import networkx as nx


CharacteristicFunction = typing.Callable[[nx.Graph, typing.Set], float]


def degree(graph: nx.Graph, coalition: typing.Set) -> float:
    fringe = copy.deepcopy(coalition)

    for node in coalition:
        neighbours = graph.neighbors(node)
        for neighbour in neighbours:
            fringe.add(neighbour)

    return len(fringe)


def threshold(graph: nx.Graph, coalition: typing.Set, threshold_value: int) -> float:
    added = copy.deepcopy(coalition)

    for node in coalition:
        neighbours = graph.neighbors(node)
        for neighbour in neighbours:
            # check threshold
            if _coalition_neighbours(graph, coalition, neighbour) >= threshold_value:
                added.add(neighbour)

    return len(added)


def closeness(graph: nx.Graph, coalition: typing.Set,
              decreasing_distance_function: typing.Callable[[float], float] = None) -> float:

    if decreasing_distance_function is None:
        decreasing_distance_function = adjusted_distance_reciprocal

    value = 0

    for node in graph.nodes():
        # compute distance between all the nodes in the coalition and take the minimum distance
        min_distance = float("inf")
        for coalition_node in coalition:
            distance = nx.shortest_path_length(graph, coalition_node, node)
            if distance < min_distance:
                min_distance = distance

        if min_distance == float("inf"):
            continue

        value += decreasing_distance_function(min_distance)

    return value


def _coalition_neighbours(graph: nx.Graph, coalition: typing.Set, node: int) -> int:
    num_neighbours = 0

    for neighbour in graph.neighbors(node):
        if neighbour in coalition:
            num_neighbours += 1

    return num_neighbours


def adjusted_distance_reciprocal(distance: float) -> float:
    """
    It returns 1 if the distance is 0
    """
    return 1 / (1 + distance)


def distance_reciprocal(distance: float) -> float:
    if distance == 0:
        return 0

    return 1 / distance
