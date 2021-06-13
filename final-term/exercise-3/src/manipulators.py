import copy
import logging
import random
import typing
from tqdm import tqdm

import networkx as nx

from election import Candidate, run_election
from network_diffusion.fj_dynamics import fj_dynamics
from shapley_centrality import CentralityValues, shapley_degree, shapley_threshold, shapley_closeness

CentralityFunction = typing.Callable[[nx.Graph], CentralityValues]
logger = logging.getLogger("final_term_exercise_3_logger")


def _evaluate_marginal_contribution(graph, candidates, target_candidate, seed_node, truthful_score):
    seed_preference = target_candidate.position

    # set seed
    old_private_belief = graph.nodes[seed_node]["private_belief"]
    old_stubbornness = graph.nodes[seed_node]["stubbornness"]

    graph.nodes[seed_node]["private_belief"] = seed_preference
    graph.nodes[seed_node]["stubbornness"] = 1

    preferences = fj_dynamics(graph)

    # update graph after dynamics
    for node, preference in preferences.items():
        graph.nodes[node]["peak_preference"] = preference

    # run election after dynamics
    results = run_election(graph, candidates)

    manipulated_score = results[target_candidate.id]

    # set private belief and stubbornness of the seed node with the older values
    graph.nodes[seed_node]["private_belief"] = old_private_belief
    graph.nodes[seed_node]["stubbornness"] = old_stubbornness

    return manipulated_score - truthful_score


def greedy_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                       number_of_seeds: int, seed: int) -> typing.Dict[int, float]:
    """
    It takes the nodes with the highest marginal contributions

    Evaluating the marginal contribution takes around 2s, so it is infeasible to do that for each node of the graph.
    """

    target_candidate = get_candidate_by_id(candidates, target_candidate_id)
    if target_candidate is None:
        raise Exception("The target candidate is None")

    # evaluate score without seeds
    preferences = fj_dynamics(graph)

    # update graph after dynamics
    for node, preference in preferences.items():
        graph.nodes[node]["peak_preference"] = preference

    # run election after dynamics
    results = run_election(graph, candidates)
    score = results[target_candidate.id]

    # Evaluate marginal contribution of each node
    max_number_of_nodes_to_evaluate = 200
    number_of_nodes_to_evaluate = number_of_seeds
    if number_of_nodes_to_evaluate < max_number_of_nodes_to_evaluate:
        number_of_nodes_to_evaluate = max_number_of_nodes_to_evaluate

    logger.info(f"\nNUMBER OF NODES TO EVALUATE: {number_of_nodes_to_evaluate}\n")

    random.seed(seed)

    nodes = list(graph.nodes())
    chosen_nodes = random.sample(nodes, number_of_nodes_to_evaluate)

    nodes_to_contribution_dict = {}
    with tqdm(total=len(chosen_nodes)) as bar:
        for node in chosen_nodes:
            score_difference = _evaluate_marginal_contribution(graph, candidates, target_candidate,
                                                               node, score)
            nodes_to_contribution_dict[node] = score_difference
            print(f"node contribution: {score_difference}") #
            bar.update(1)

    # sort in ascending order
    nodes_to_contribution = sorted(nodes_to_contribution_dict.items(), key=lambda element: -element[1])

    seeds = {}
    for i in range(number_of_seeds):
        seeds[nodes_to_contribution[i][0]] = target_candidate.position

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
