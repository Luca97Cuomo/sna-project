import typing

import networkx as nx

OpinionsDict = typing.Dict[int, float]


def is_dynamics_converged(prev_opinions: OpinionsDict, current_opinions: OpinionsDict, precision_digits: int) -> bool:
    for node in prev_opinions.keys():
        prev_opinion = prev_opinions[node]
        current_opinion = current_opinions[node]

        if round(abs(prev_opinion - current_opinion), precision_digits) != 0:
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


def _evaluate_neighborhood_opinion(graph: nx.Graph, node: int, prev_opinions: OpinionsDict):
    neighbours_sum = 0
    neighbours = graph.neighbors(node)
    for neighbour in neighbours:
        prev_opinion = prev_opinions[neighbour]
        neighbours_sum += prev_opinion

    return neighbours_sum / graph.degree[node]


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

        if time_step % 50 == 0:
            max_convergence_error = _evaluate_max_convergence_error(prev_opinions, current_opinions)
            print(f"Partial number of iterations: {time_step}. Max convergence error: {max_convergence_error}")

        prev_opinions = current_opinions
        current_opinions = {}

    print(f"Number of iterations required to converge: {time_step}")

    return prev_opinions

