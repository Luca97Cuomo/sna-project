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


NUMBER_OF_DIGITS = 2
MIN_NUMBER_OF_ITERATIONS = 44940
N_PREFERENCES = 5


def _get_preferences(target_candidate_position: float, n_preferences: int):
    preferences = [target_candidate_position]
    for _ in range(n_preferences - 1):
        preferences.append(random.random())
    return preferences


def _compute_marginal_contribution_on_nodes(graph, candidates, target_candidate, chunk,
                                            truthful_score, index, number_of_digits):
    marginal_contributions = {}

    if index == 0:
        bar = tqdm(total=len(chunk))
    for node in chunk:
        max_marginal_contribution = float('-inf')
        max_preference = None
        for preference in _get_preferences(target_candidate.position, N_PREFERENCES):
            marginal_contribution = _compute_marginal_contribution(graph, candidates,
                                                                   target_candidate, node, truthful_score,
                                                                   number_of_digits, preference)
            if max_marginal_contribution < marginal_contribution:
                max_marginal_contribution = marginal_contribution
                max_preference = preference
        marginal_contributions[node] = (max_marginal_contribution, max_preference)
        print(f"marginal contribution of {node} is {max_marginal_contribution} (pref: {max_preference} target: {target_candidate.position})")
        if index == 0:
            bar.update(1)
    return marginal_contributions


def _compute_marginal_contribution(graph, candidates, target_candidate, seed_node,
                                   dynamics_score, number_of_digits: int, seed_preference: float):
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

    # The graph has to be copied because it will be modified by this function
    copied_graph = copy.deepcopy(graph)

    seeds: typing.Dict[int, float] = {}  # node_id -> preference

    # get target candidate instance
    target_candidate = get_candidate_by_id(candidates, target_candidate_id)
    if target_candidate is None:
        raise Exception("The target candidate is None")

    # fix the max number of iterations
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

            # if number of nodes is greater that the available nodes then pick few nodes
            min_number_of_nodes = min(number_of_nodes, len(all_nodes_without_seeds))
            if min_number_of_nodes != number_of_nodes:
                logger.info("Nodes per iteration would have been too many")

            chosen_nodes = random.sample(all_nodes_without_seeds, min_number_of_nodes)

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
                                                                     NUMBER_OF_DIGITS) for index, chunk in
                    enumerate(chunks))

            nodes_to_contribution_dict = {}
            for result in results:
                for node, value in result.items():
                    nodes_to_contribution_dict[node] = value

            # take the node with the higher marginal contribution
            max_node = None
            max_contribution = None
            max_preference = None
            for node, value in nodes_to_contribution_dict.items():
                contribution = value[0]
                preference = value[1]
                if max_node is None or contribution > max_contribution:
                    max_node = node
                    max_contribution = contribution
                    max_preference = preference

            # add max_node to seeds
            seeds[max_node] = max_preference

            # update graph with current seed
            copied_graph.nodes[max_node]["private_belief"] = seeds[max_node]
            copied_graph.nodes[max_node]["stubbornness"] = 1

            bar.update(1)

    return seeds


def _get_nodes_sorted_with_centrality(graph: nx.Graph, centrality_function: typing.Callable[[nx.Graph], CentralityValues]) -> typing.List[int]:
    centrality_values = centrality_function(graph)
    centrality_values = sorted(centrality_values.items(), key=lambda element: element[1], reverse=True)
    return [item[0] for item in centrality_values]


def multi_level_greedy_manipulator_with_centrality_sampling(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                                   number_of_seeds: int, seed: int, number_of_jobs: int) -> typing.Dict[int, float]:
    # The graph has to be copied because it will be modified by this function
    copied_graph = copy.deepcopy(graph)

    seeds: typing.Dict[int, float] = {}  # node_id -> preference

    # get target candidate instance
    target_candidate = get_candidate_by_id(candidates, target_candidate_id)
    if target_candidate is None:
        raise Exception("The target candidate is None")

    # fix the max number of iterations
    number_of_iterations = number_of_seeds
    if number_of_iterations < MIN_NUMBER_OF_ITERATIONS:
        number_of_iterations = MIN_NUMBER_OF_ITERATIONS

    nodes_for_each_iteration = math.floor(number_of_iterations / number_of_seeds)

    logger.info(f"\nTOTAL NUMBER OF ITERATIONS: {number_of_iterations},"
                f" NUMBER OF NODES FOR EACH ITERATION: {nodes_for_each_iteration}\n"
                f"\nNUMBER_OF_DIGITS: {NUMBER_OF_DIGITS}\n")

    random.seed(seed)

    degree_bucket = _get_nodes_sorted_with_centrality(graph, shapley_degree)
    distance_bucket = _most_distant_nodes(graph, target_candidate.position)
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

            # if number of nodes is greater that the available nodes then pick few nodes
            min_number_of_nodes = min(number_of_nodes, len(all_nodes_without_seeds))
            if min_number_of_nodes != number_of_nodes:
                logger.info("Nodes per iteration would have been too many")

            current_random_bucket = random.sample(all_nodes_without_seeds, min(min_number_of_nodes * 2, len(all_nodes_without_seeds)))
            current_degree_bucket = degree_bucket[:min_number_of_nodes]
            current_distance_bucket = distance_bucket[:min_number_of_nodes]

            bucket = set(current_random_bucket + current_degree_bucket + current_distance_bucket)
            chosen_nodes = random.sample(bucket, min_number_of_nodes)

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
                                                                     NUMBER_OF_DIGITS) for index, chunk in
                    enumerate(chunks))

            nodes_to_contribution_dict = {}
            for result in results:
                for node, value in result.items():
                    nodes_to_contribution_dict[node] = value

            # take the node with the higher marginal contribution
            max_node = None
            max_contribution = None
            max_preference = None
            for node, value in nodes_to_contribution_dict.items():
                contribution = value[0]
                preference = value[1]
                if max_node is None or contribution > max_contribution:
                    max_node = node
                    max_contribution = contribution
                    max_preference = preference

            # add max_node to seeds
            seeds[max_node] = max_preference

            # update graph with current seed
            copied_graph.nodes[max_node]["private_belief"] = seeds[max_node]
            copied_graph.nodes[max_node]["stubbornness"] = 1

            # update buckets
            degree_bucket.remove(max_node)
            distance_bucket.remove(max_node)

            bar.update(1)

    return seeds


def greedy_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                       number_of_seeds: int, seed: int, number_of_jobs: int) -> typing.Dict[int, float]:
    """
    It takes the nodes with the highest marginal contributions

    Evaluating the marginal contribution takes around 2s, so it is infeasible to do that for each node of the graph.
    """
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
    max_number_of_nodes_to_evaluate = MIN_NUMBER_OF_ITERATIONS
    number_of_nodes_to_evaluate = number_of_seeds
    if number_of_nodes_to_evaluate < max_number_of_nodes_to_evaluate:
        number_of_nodes_to_evaluate = max_number_of_nodes_to_evaluate

    if number_of_nodes_to_evaluate > len(graph.nodes()):
        raise Exception("The number of nodes to evaluate has to be smaller that the size of the graph")

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
                                                             score, index, NUMBER_OF_DIGITS) for index, chunk in
            enumerate(chunks))
    nodes_to_contribution_dict = {}
    for result in results:
        for node, value in result.items():
            nodes_to_contribution_dict[node] = value

    # sort in ascending order
    nodes_to_contribution = sorted(nodes_to_contribution_dict.items(), key=lambda element: -element[1][0])

    seeds = {}
    for i in range(number_of_seeds):
        seeds[nodes_to_contribution[i][0]] = nodes_to_contribution[i][1][1]

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


def _most_distant_nodes(graph: nx.Graph, target_candidate_position: float) -> typing.List[int]:
    nodes_with_distance = []
    for node in graph.nodes:
        nodes_with_distance.append((node, abs(graph.nodes[node]['private_belief'] - target_candidate_position)))
    nodes_with_distance = sorted(nodes_with_distance, key=lambda item: item[1], reverse=True)
    nodes_with_distance = [item[0] for item in nodes_with_distance]
    return nodes_with_distance


def furthest_voters_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                                number_of_seeds: int, seed: int = 42, number_of_jobs: int = 1) -> typing.Dict[int, float]:
    target_candidate_position = get_candidate_by_id(candidates, target_candidate_id).position

    sorted_nodes = _most_distant_nodes(graph, target_candidate_position)
    seeds = {}
    for i in range(number_of_seeds):
        node = sorted_nodes[i]
        seeds[node] = target_candidate_position

    return seeds


def get_candidate_by_id(candidates: typing.List[Candidate], candidate_id: int) -> typing.Optional[Candidate]:
    for i in range(len(candidates)):
        if candidates[i].id == candidate_id:
            return candidates[i]

    return None
