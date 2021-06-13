import logging
import random
import typing

import networkx as nx

from election import Candidate
from shapley_centrality import CentralityValues, shapley_degree, shapley_threshold, shapley_closeness

CentralityFunction = typing.Callable[[nx.Graph], CentralityValues]
logger = logging.getLogger("final_term_exercise_3_logger")


def _evaluate_marginal_contribution(graph, candidates, target_candidate_id, node):
    return 10


def greedy_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                       number_of_seeds: int, seed: int) -> typing.Dict[int, float]:
    """
    It takes the nodes with the highest marginal contributions
    """

    # Evaluate marginal contribution of each node as the only seed
    nodes_to_contribution_dict = {}
    for node in graph.nodes():
        nodes_to_contribution_dict[node] = _evaluate_marginal_contribution(graph, candidates, target_candidate_id, node)

    # sort in ascending order
    nodes_to_contribution = sorted(nodes_to_contribution_dict.items(), key=lambda element: -element[1])

    seeds = {}
    for i in range(number_of_seeds):
        seeds[nodes_to_contribution[i][0]] = nodes_to_contribution[1]

    return seeds


def shapley_closeness_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                                  number_of_seeds: int, seed: int) -> typing.Dict[int, float]:
    return centrality_based_manipulator(graph, candidates, target_candidate_id, number_of_seeds,
                                        seed, shapley_closeness)


def shapley_threshold_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                                  number_of_seeds: int, seed: int) -> typing.Dict[int, float]:
    threshold = 1000
    logger.info(f"\nTHRESHOLD: {threshold}")

    return centrality_based_manipulator(graph, candidates, target_candidate_id, number_of_seeds, seed,
                                        lambda graph: shapley_threshold(graph, threshold))


def shapley_degree_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                               number_of_seeds: int, seed: int) -> typing.Dict[int, float]:
    return centrality_based_manipulator(graph, candidates, target_candidate_id, number_of_seeds, seed, shapley_degree)


def centrality_based_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                                 number_of_seeds: int, seed: int,
                                 centrality_function: CentralityFunction) -> typing.Dict[int, float]:
    centrality_values = centrality_function(graph)

    seeds = {}

    # Sort in descending order
    # The bigger is the centrality value, more central is the node
    sorted_values = sorted(centrality_values.items(), key=lambda element: -element[1])

    target_candidate = get_candidate_by_id(candidates, target_candidate_id)
    if target_candidate is None:
        raise Exception("The target candidate is None")

    count = 0
    for node, value in sorted_values:
        if count == number_of_seeds:
            break

        seeds[node] = target_candidate.position

        count += 1

    return seeds


def random_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                       number_of_seeds: int, seed: int = 42) -> typing.Dict[int, float]:

    random.seed(seed)

    seeds = {}
    nodes = list(graph.nodes())
    chosen_nodes = random.sample(nodes, number_of_seeds)

    target_candidate = get_candidate_by_id(candidates, target_candidate_id)
    if target_candidate is None:
        raise Exception("The target candidate is None")

    for node in chosen_nodes:
        seeds[node] = target_candidate.position

    return seeds


def get_candidate_by_id(candidates: typing.List[Candidate], candidate_id: int) -> typing.Optional[Candidate]:
    for i in range(len(candidates)):
        if candidates[i].id == candidate_id:
            return candidates[i]

    return None
