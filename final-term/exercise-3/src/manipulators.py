import copy
import logging
import math
import random
import typing

from joblib import Parallel, delayed
from tqdm import tqdm

import networkx as nx

from election import Candidate, run_election
from network_diffusion.fj_dynamics import fj_dynamics
from shapley_centrality import CentralityValues, shapley_degree, shapley_threshold, shapley_closeness

CentralityFunction = typing.Callable[[nx.Graph], CentralityValues]
logger = logging.getLogger("final_term_exercise_3_logger")


def _compute_marginal_contribution_on_nodes(graph, candidates, target_candidate, chunk,
                                            truthful_score, index, number_of_digits):
    marginal_contributions = {}

    if index == 0:
        # only if index == 0 use tqdm
        with tqdm(total=len(chunk)) as bar:
            for node in chunk:
                marginal_contribution = _compute_marginal_contribution(graph, candidates,
                                                                       target_candidate, node, truthful_score,
                                                                       number_of_digits)
                marginal_contributions[node] = marginal_contribution
                print(f"marginal contribution of {node} is {marginal_contribution}")

                bar.update(1)
    else:
        for node in chunk:
            marginal_contribution = _compute_marginal_contribution(graph, candidates,
                                                                   target_candidate, node, truthful_score,
                                                                   number_of_digits)
            marginal_contributions[node] = marginal_contribution
            print(f"marginal contribution of {node} is {marginal_contribution}")

    return marginal_contributions


def _compute_marginal_contribution(graph, candidates, target_candidate, seed_node,
                                   dynamics_score, number_of_digits: int):
    seed_preference = target_candidate.position

    # set seed
    old_private_belief = graph.nodes[seed_node]["private_belief"]
    old_stubbornness = graph.nodes[seed_node]["stubbornness"]

    graph.nodes[seed_node]["private_belief"] = seed_preference
    graph.nodes[seed_node]["stubbornness"] = 1

    preferences = fj_dynamics(graph, convergence_digits=number_of_digits)

    # update graph after dynamics
    for node, preference in preferences.items():
        graph.nodes[node]["peak_preference"] = preference

    # run election after dynamics
    results = run_election(graph, candidates)

    manipulated_score = results[target_candidate.id]

    # set private belief and stubbornness of the seed node with the older values
    graph.nodes[seed_node]["private_belief"] = old_private_belief
    graph.nodes[seed_node]["stubbornness"] = old_stubbornness

    return manipulated_score - dynamics_score


def multi_level_greedy_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                                   number_of_seeds: int, seed: int, number_of_jobs: int) -> typing.Dict[int, float]:
    NUMBER_OF_DIGITS = 2

    # The graph has to be copied because it will be modified by this function
    copied_graph = copy.deepcopy(graph)

    seeds: typing.Dict[int, float] = {}  # node_id -> preference

    # get target candidate instance
    target_candidate = get_candidate_by_id(candidates, target_candidate_id)
    if target_candidate is None:
        raise Exception("The target candidate is None")

    # fix the max number of iterations
    MIN_NUMBER_OF_ITERATIONS = 2000
    number_of_iterations = number_of_seeds
    if number_of_iterations < MIN_NUMBER_OF_ITERATIONS:
        number_of_iterations = MIN_NUMBER_OF_ITERATIONS

    nodes_for_each_iteration = math.floor(number_of_iterations / number_of_seeds)

    logger.info(f"\nTOTAL NUMBER OF ITERATIONS: {number_of_iterations},"
                f" NUMBER OF NODES FOR EACH ITERATION: {nodes_for_each_iteration}\n"
                f"\nNUMBER_OF_DIGITS: {NUMBER_OF_DIGITS}\n")

    random.seed(seed)

    with tqdm(total=number_of_seeds) as bar:
        for i in range(number_of_seeds):
            # evaluate score with seeds
            preferences = fj_dynamics(copied_graph, NUMBER_OF_DIGITS)

            # update graph after dynamics
            for node, preference in preferences.items():
                copied_graph.nodes[node]["peak_preference"] = preference

            # run election after dynamics
            results = run_election(copied_graph, candidates)
            score = results[target_candidate.id]

            ##########
            # evaluate marginal contributions
            ##########

            number_of_nodes = nodes_for_each_iteration
            if i == 0:
                number_of_nodes += number_of_iterations - (nodes_for_each_iteration * number_of_seeds)

            # compute chosen nodes
            all_nodes_without_seeds = list(filter(lambda element: element not in seeds, copied_graph.nodes()))
            chosen_nodes = random.sample(all_nodes_without_seeds, number_of_nodes)

            with Parallel(n_jobs=number_of_jobs) as parallel:
                # compute chunks
                chunks = []
                chunk_size = math.ceil(len(chosen_nodes) / number_of_jobs)
                for k in range(number_of_jobs):
                    chunks.append(chosen_nodes[k * chunk_size: (k + 1) * chunk_size])

                results = parallel(
                    delayed(_compute_marginal_contribution_on_nodes)(copied_graph, candidates,
                                                                     target_candidate, chunk, score,
                                                                     index + 1,
                                                                     NUMBER_OF_DIGITS) for index, chunk in enumerate(chunks))

            nodes_to_contribution_dict = {}
            for result in results:
                for node, value in result.items():
                    nodes_to_contribution_dict[node] = value

            # take the node with the higher marginal contribution
            max_node = None
            max_value = None
            for node, value in nodes_to_contribution_dict.items():
                if max_node is None or value > max_value:
                    max_node = node
                    max_value = value

            # add max_node to seeds
            seeds[max_node] = target_candidate.position

            # update graph with current seed
            copied_graph.nodes[max_node]["private_belief"] = target_candidate.position
            copied_graph.nodes[max_node]["stubbornness"] = 1

            bar.update(1)

    return seeds


def greedy_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                       number_of_seeds: int, seed: int, number_of_jobs: int) -> typing.Dict[int, float]:
    """
    It takes the nodes with the highest marginal contributions

    Evaluating the marginal contribution takes around 2s, so it is infeasible to do that for each node of the graph.
    """

    NUMBER_OF_DIGITS = 5

    target_candidate = get_candidate_by_id(candidates, target_candidate_id)
    if target_candidate is None:
        raise Exception("The target candidate is None")

    # evaluate score without seeds
    preferences = fj_dynamics(graph, NUMBER_OF_DIGITS)

    # update graph after dynamics
    for node, preference in preferences.items():
        graph.nodes[node]["peak_preference"] = preference

    # run election after dynamics
    results = run_election(graph, candidates)
    score = results[target_candidate.id]

    # Evaluate marginal contribution of each node
    max_number_of_nodes_to_evaluate = len(graph.nodes())
    number_of_nodes_to_evaluate = number_of_seeds
    if number_of_nodes_to_evaluate < max_number_of_nodes_to_evaluate:
        number_of_nodes_to_evaluate = max_number_of_nodes_to_evaluate

    logger.info(f"\nNUMBER OF NODES TO EVALUATE: {number_of_nodes_to_evaluate}"
                f"\nNUMBER_OF_DIGITS: {NUMBER_OF_DIGITS}\n")

    random.seed(seed)

    nodes = list(graph.nodes())
    chosen_nodes = random.sample(nodes, number_of_nodes_to_evaluate)

    with Parallel(n_jobs=number_of_jobs) as parallel:
        # compute chunks
        chunks = []
        chunk_size = math.ceil(len(chosen_nodes) / number_of_jobs)
        for i in range(number_of_jobs):
            chunks.append(chosen_nodes[i * chunk_size: (i + 1) * chunk_size])

        results = parallel(
            delayed(_compute_marginal_contribution_on_nodes)(graph, candidates, target_candidate, chunk,
                                                             score, index, NUMBER_OF_DIGITS) for index, chunk in enumerate(chunks))
    nodes_to_contribution_dict = {}
    for result in results:
        for node, value in result.items():
            nodes_to_contribution_dict[node] = value

    # sort in ascending order
    nodes_to_contribution = sorted(nodes_to_contribution_dict.items(), key=lambda element: -element[1])

    seeds = {}
    for i in range(number_of_seeds):
        seeds[nodes_to_contribution[i][0]] = target_candidate.position

    return seeds


def shapley_closeness_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                                  number_of_seeds: int, seed: int, number_of_jobs: int = 1) -> typing.Dict[int, float]:
    return centrality_based_manipulator(graph, candidates, target_candidate_id, number_of_seeds,
                                        seed, shapley_closeness)


def shapley_threshold_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                                  number_of_seeds: int, seed: int, number_of_jobs: int = 1) -> typing.Dict[int, float]:
    threshold = 1000
    logger.info(f"\nTHRESHOLD: {threshold}")

    return centrality_based_manipulator(graph, candidates, target_candidate_id, number_of_seeds, seed,
                                        lambda graph: shapley_threshold(graph, threshold))


def shapley_degree_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                               number_of_seeds: int, seed: int, number_of_jobs: int = 1) -> typing.Dict[int, float]:
    return centrality_based_manipulator(graph, candidates, target_candidate_id, number_of_seeds,
                                        seed, number_of_jobs, shapley_degree)


def centrality_based_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                                 number_of_seeds: int, seed: int, number_of_jobs: int,
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
                       number_of_seeds: int, seed: int = 42, number_of_jobs: int = 1) -> typing.Dict[int, float]:

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
