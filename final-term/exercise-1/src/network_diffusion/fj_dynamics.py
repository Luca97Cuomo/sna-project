import typing

import networkx as nx
import numpy as np

OpinionsDict = typing.Dict[int, float]


def is_dynamics_converged(prev_opinions: OpinionsDict, current_opinions: OpinionsDict, precision_digits: int) -> bool:
    for node in prev_opinions.keys():
        prev_opinion = prev_opinions[node]
        current_opinion = current_opinions[node]

        if round(abs(prev_opinion - current_opinion), precision_digits) != 0:
            return False

    return True


def optimized_is_dynamics_converged(prev_opinions: np.array, current_opinions: np.array, precision_digits: int) -> bool:
    for i in range(prev_opinions.size):
        if round(abs(prev_opinions[i] - current_opinions[i]), precision_digits) != 0:
            return False

    return True


def _evaluate_max_convergence_error(prev_opinions: OpinionsDict, current_opinions: OpinionsDict) -> float:
    max_error = 0

    for node in prev_opinions.keys():
        prev_opinion = prev_opinions[node]
        current_opinion = current_opinions[node]

        error = abs(prev_opinion - current_opinion)

        if error > max_error:
            max_error = error

    return max_error


def _optimized_evaluate_neighborhood_opinion(graph: nx.Graph, node: int,
                                             prev_opinions: np.array, nodes_to_indices: typing.Dict[int, int]):
    neighbours_sum = 0
    neighbours = graph.neighbors(node)
    for neighbour in neighbours:
        index = nodes_to_indices[neighbour]
        prev_opinion = prev_opinions[index]
        neighbours_sum += prev_opinion

    return neighbours_sum / graph.degree[node]


def _evaluate_neighborhood_opinion(graph: nx.Graph, node: int, prev_opinions: OpinionsDict):
    neighbours_sum = 0
    neighbours = graph.neighbors(node)
    for neighbour in neighbours:
        prev_opinion = prev_opinions[neighbour]
        neighbours_sum += prev_opinion

    return neighbours_sum / graph.degree[node]


def optimized_fj_dynamics(graph: nx.Graph, convergence_digits: int = 5) -> OpinionsDict:
    # initialize
    nodes = graph.nodes()
    nodes_list = list(nodes)
    prev_opinions = np.array(list(nodes[node]["private_belief"] for node in nodes), dtype=np.float32)
    current_opinions = np.zeros(len(nodes_list), dtype=np.float32)
    nodes_to_indices = {nodes_list[i]: i for i in range(len(nodes_list))}

    is_converged = False
    time_step = 0
    while not is_converged:
        time_step += 1
        for i in range(len(nodes_list)):
            # update current_opinion
            private_belief = nodes[nodes_list[i]]["private_belief"]
            stubbornness = nodes[nodes_list[i]]["stubbornness"]

            neighborhood_opinion = _optimized_evaluate_neighborhood_opinion(graph, nodes_list[i],
                                                                            prev_opinions, nodes_to_indices)

            current_opinions[i] = stubbornness * private_belief + (1 - stubbornness) * neighborhood_opinion

        is_converged = optimized_is_dynamics_converged(prev_opinions, current_opinions, convergence_digits)

        # copy current to prev
        prev_opinions = np.copy(current_opinions)

    # built dict
    current_opinions_dict = {nodes_list[i]: current_opinions[i] for i in range(len(nodes_list))}

    print(f"Number of iterations required to converge: {time_step}")

    return current_opinions_dict


def fj_dynamics(graph: nx.Graph, convergence_digits: int = 5) -> OpinionsDict:
    """
    Each node has 2 attributes:
    private_belief -> [0, 1] float
    stubbornness -> [0, 1] float

    It returns a dict containing the final opinions in the form:
    {node: opinion}

    Convergence is checked on 5 digits
    """

    # initialize
    prev_opinions = {node: graph.nodes[node]["private_belief"] for node in graph.nodes()}
    current_opinions = {}

    is_converged = False
    time_step = 0
    while not is_converged:
        time_step += 1
        for node in graph.nodes():
            # update current_opinion
            private_belief = graph.nodes[node]["private_belief"]
            stubbornness = graph.nodes[node]["stubbornness"]

            neighborhood_opinion = _evaluate_neighborhood_opinion(graph, node, prev_opinions)

            current_opinions[node] = stubbornness * private_belief + (1 - stubbornness) * neighborhood_opinion

        is_converged = is_dynamics_converged(prev_opinions, current_opinions, convergence_digits)

        """
        if time_step % 50 == 0:
            max_convergence_error = _evaluate_max_convergence_error(prev_opinions, current_opinions)
            print(f"Partial number of iterations: {time_step}. Max convergence error: {max_convergence_error}")
        """

        prev_opinions = current_opinions
        current_opinions = {}

    print(f"Number of iterations required to converge: {time_step}")

    return prev_opinions

