import copy
import logging
import math
import random
import time
import typing

from joblib import Parallel, delayed
from tqdm import tqdm

import networkx as nx

from mid_term.exercise_2.centrality_measures import algebraic_page_rank, naive_hits
from final_term.exercise_3.src.election import Candidate, run_election, get_full_results_election
from final_term.exercise_1.src.network_diffusion.fj_dynamics import fj_dynamics
from final_term.exercise_1.src.shapley_centrality import CentralityValues, shapley_degree, shapley_threshold, shapley_closeness

CentralityFunction = typing.Callable[[nx.Graph], CentralityValues]
Seeds = typing.Dict[int, float]
Manipulator = typing.Callable[[nx.Graph, typing.List[Candidate], int, int, int, int], Seeds]

logger = logging.getLogger("final_term_exercise_3_logger")


NUMBER_OF_DIGITS = 2
MIN_NUMBER_OF_ITERATIONS = 468
N_PREFERENCES = 1


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
        # Evaluate the marginal contribution of the current node for multiple preferences
        for preference in _get_preferences(target_candidate.position, N_PREFERENCES):
            marginal_contribution = _compute_marginal_contribution(graph, candidates,
                                                                   target_candidate, node, truthful_score,
                                                                   number_of_digits, preference)
            if max_marginal_contribution < marginal_contribution:
                max_marginal_contribution = marginal_contribution
                max_preference = preference
        marginal_contributions[node] = (max_marginal_contribution, max_preference)
        if index == 0:
            bar.update(1)
    return marginal_contributions


def _compute_marginal_contribution(graph, candidates, target_candidate, seed_node,
                                   dynamics_score, number_of_digits: int, seed_preference: float):
    # Set seed preference and stubbornness
    old_private_belief = graph.nodes[seed_node]["private_belief"]
    old_stubbornness = graph.nodes[seed_node]["stubbornness"]

    graph.nodes[seed_node]["private_belief"] = seed_preference
    graph.nodes[seed_node]["stubbornness"] = 1

    preferences = fj_dynamics(graph, convergence_digits=number_of_digits)

    # Update graph after dynamics
    for node, preference in preferences.items():
        graph.nodes[node]["peak_preference"] = preference

    # Run election after dynamics
    results = run_election(graph, candidates)

    manipulated_score = results[target_candidate.id]

    # set private belief and stubbornness of the seed node with the older values
    graph.nodes[seed_node]["private_belief"] = old_private_belief
    graph.nodes[seed_node]["stubbornness"] = old_stubbornness

    return manipulated_score - dynamics_score


def multi_level_greedy_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                                   number_of_seeds: int, seed: int, number_of_jobs: int, *args) -> typing.Dict[int, float]:

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
                f"\nNUMBER_OF_DIGITS: {NUMBER_OF_DIGITS}\n"
                f"NODES TO EXCLUDE: TRUE"
                f"\nNUMBER OF PREFERENCES: {N_PREFERENCES}")

    random.seed(seed)

    with tqdm(total=number_of_seeds) as bar:
        with Parallel(n_jobs=number_of_jobs) as parallel:
            for i in range(number_of_seeds):
                # evaluate score with seeds
                preferences = fj_dynamics(copied_graph, NUMBER_OF_DIGITS)

                # update graph after dynamics
                for node, preference in preferences.items():
                    copied_graph.nodes[node]["peak_preference"] = preference

                # run election after dynamics
                results, voters_to_candidates = get_full_results_election(copied_graph, candidates)
                score = results[target_candidate.id]

                # get nodes to exclude
                nodes_to_exclude = []
                for node, candidate_id in voters_to_candidates.items():
                    if candidate_id == target_candidate_id:
                        nodes_to_exclude.append(node)

                ##########
                # evaluate marginal contributions
                ##########

                number_of_nodes = nodes_for_each_iteration
                if i == 0:
                    number_of_nodes += number_of_iterations - (nodes_for_each_iteration * number_of_seeds)

                # compute chosen nodes
                all_nodes_without_seeds = list(filter(lambda element: element not in seeds, copied_graph.nodes()))

                print(f"len all_nodes: {len(all_nodes_without_seeds)} - len to exclude: {len(nodes_to_exclude)}")

                # exclude also node that vote for me
                all_nodes_without_seeds = list(filter(lambda element: element not in nodes_to_exclude, all_nodes_without_seeds))

                print(f"len all_nodes after exclusion: {len(all_nodes_without_seeds)}")

                # if number of nodes is greater that the available nodes then pick few nodes
                min_number_of_nodes = min(number_of_nodes, len(all_nodes_without_seeds))
                if min_number_of_nodes != number_of_nodes:
                    logger.info("Nodes per iteration would have been too many")

                chosen_nodes = random.sample(all_nodes_without_seeds, min_number_of_nodes)

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
                                   number_of_seeds: int, seed: int, number_of_jobs: int, *args) -> typing.Dict[int, float]:
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
        with Parallel(n_jobs=number_of_jobs) as parallel:
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
                       number_of_seeds: int, seed: int, number_of_jobs: int, *args) -> typing.Dict[int, float]:
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


def belief_degree_centrality(graph: nx.Graph, target_candidate_position: float) -> CentralityValues:
    values = {}

    for node in graph.nodes():
        distance = abs(graph.nodes[node]["private_belief"] - target_candidate_position)
        for neighbour in graph.neighbors(node):
            distance += abs(graph.nodes[neighbour]["private_belief"] - target_candidate_position)

        values[node] = distance

    return values


def shapley_closeness_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                                  number_of_seeds: int, seed: int, number_of_jobs: int = 1, *args) -> typing.Dict[int, float]:
    return centrality_based_manipulator(graph, candidates, target_candidate_id, number_of_seeds,
                                        seed, shapley_closeness)


def shapley_threshold_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                                  number_of_seeds: int, seed: int, number_of_jobs: int = 1, *args) -> typing.Dict[int, float]:
    threshold = 1000
    logger.info(f"\nTHRESHOLD: {threshold}")

    return centrality_based_manipulator(graph, candidates, target_candidate_id, number_of_seeds, seed,
                                        lambda graph: shapley_threshold(graph, threshold))


def shapley_degree_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                               number_of_seeds: int, seed: int, number_of_jobs: int = 1, *args) -> typing.Dict[int, float]:
    return centrality_based_manipulator(graph, candidates, target_candidate_id, number_of_seeds,
                                        seed, number_of_jobs, shapley_degree)


def belief_degree_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                              number_of_seeds: int, seed: int, number_of_jobs: int = 1, *args) -> typing.Dict[int, float]:

    target_candidate = get_candidate_by_id(candidates, target_candidate_id)
    if target_candidate is None:
        raise Exception("The target candidate is None")
    target_candidate_position = target_candidate.position

    return centrality_based_manipulator(graph, candidates, target_candidate_id, number_of_seeds,
                                        seed, number_of_jobs,
                                        lambda graph: belief_degree_centrality(graph, target_candidate_position))


def page_rank_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                              number_of_seeds: int, seed: int, number_of_jobs: int = 1, *args, **kwargs) -> typing.Dict[int, float]:
    return centrality_based_manipulator(graph, candidates, target_candidate_id, number_of_seeds, seed, number_of_jobs,
                                        algebraic_page_rank)


def hubs_based_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                           number_of_seeds: int, seed: int, number_of_jobs: int = 1, *args, **kwargs) -> Seeds:
    def naive_hits_hubs(graph: nx.Graph) -> CentralityValues:
        return naive_hits(graph)[0]

    return centrality_based_manipulator(graph, candidates, target_candidate_id, number_of_seeds, seed,
                                        number_of_jobs, naive_hits_hubs)


def triangles_based_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                          number_of_seeds: int, seed: int, number_of_jobs: int = 1, *args, **kwargs) -> typing.Dict[int, float]:
    return centrality_based_manipulator(graph, candidates, target_candidate_id, number_of_seeds, seed, number_of_jobs,
                                        nx.clustering)


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
                       number_of_seeds: int, seed: int = 42, number_of_jobs: int = 1, *args) -> typing.Dict[int, float]:
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


def _parallel_time_improvement_factor_heuristic(number_of_jobs: int):
    if number_of_jobs <= 0:
        return 1

    return math.log(number_of_jobs, 2) + 1


def _estimate_number_of_iterations(max_running_time_s: int, number_of_seeds: int,
                                   number_of_digits: int, graph: nx.Graph, candidates, number_of_jobs: int):
    """
    It returns the number of nodes for which the marginal contribution has to be evaluated in the next iteration.
    """

    # total_time = marginal_contribution_time * num_total_iterations
    # num_total_iterations = total_time / marginal_contribution_time
    # num_iterations_for_each_seed = num_total_iterations / number_of_seeds

    nodes = list(graph.nodes())

    start_time_s = time.time()

    # Estimate the marginal contribution time, considering only one job.
    _ = _compute_marginal_contribution(graph, candidates,
                                       candidates[0], nodes[0], 2000,
                                       number_of_digits, seed_preference=0.5)
    end_time_s = time.time()

    marginal_contribution_time_s = (end_time_s - start_time_s) / _parallel_time_improvement_factor_heuristic(number_of_jobs)

    logger.info(f"The estimate of the marginal contribution time with one job is: {end_time_s - start_time_s}")
    logger.info(f"The estimate of the marginal contribution time"
                f" with {number_of_jobs} job is: {marginal_contribution_time_s}")

    num_total_iterations = max_running_time_s / marginal_contribution_time_s
    num_iterations_for_each_seed = math.floor(num_total_iterations / number_of_seeds)

    if num_iterations_for_each_seed < 1:
        return 1
    return num_iterations_for_each_seed


def _estimate_number_of_iterations_at_execution_time(last_iteration_time_s: int, last_num_iterations_for_each_seed: int,
                                                     remaining_seeds: int, remaining_time_s, number_of_jobs: int):
    """
    Note: it remaining_time_s is < 0 num_iterations_for_each_seed will be set to number_of_jobs
    """

    marginal_contribution_time_s = last_iteration_time_s / last_num_iterations_for_each_seed

    logger.debug(f"The estimate of the marginal contribution time with one job is: {marginal_contribution_time_s}")

    num_total_iterations = remaining_time_s / marginal_contribution_time_s
    num_iterations_for_each_seed = math.floor(num_total_iterations / remaining_seeds)

    logger.debug(f"The number of iterations for the next level is : {num_iterations_for_each_seed}")

    if num_iterations_for_each_seed < 1:
        return 1
    return num_iterations_for_each_seed


def timed_multi_level_greedy_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                                         number_of_seeds: int, seed: int,
                                         number_of_jobs: int, max_running_time_s: int) -> typing.Dict[int, float]:

    # The graph has to be copied because it will be modified during the execution of this function
    copied_graph = copy.deepcopy(graph)

    # Initialize seeds dict
    seeds: typing.Dict[int, float] = {}  # node_id -> preference

    # Get target candidate instance
    target_candidate = get_candidate_by_id(candidates, target_candidate_id)
    if target_candidate is None:
        raise Exception("The target candidate is None")

    # Estimate the number of iterations to be done based on the max_running_time_s
    # fix the max number of iterations
    number_of_iterations = _estimate_number_of_iterations(max_running_time_s, number_of_seeds, NUMBER_OF_DIGITS,
                                                          copied_graph, candidates, number_of_jobs)

    # Logging some infos
    logger.info(f"\nNUMBER OF ITERATIONS FOR EACH SEED: {number_of_iterations},"
                f"\nNUMBER_OF_DIGITS: {NUMBER_OF_DIGITS}"
                f"\nNODES TO EXCLUDE: TRUE"
                f"\nNUMBER OF PREFERENCES: {N_PREFERENCES}")

    # Setting seed
    random.seed(seed)

    remaining_time = max_running_time_s
    total_performed_iterations = 0  # for debugging purposes

    with tqdm(total=number_of_seeds) as bar:
        with Parallel(n_jobs=number_of_jobs) as parallel:
            for i in range(number_of_seeds):
                start_time_s = time.time()

                ##########
                # Evaluate target candidate score with the actual seeds
                ##########

                preferences = fj_dynamics(copied_graph, NUMBER_OF_DIGITS)

                # Update graph after dynamics
                for node, preference in preferences.items():
                    copied_graph.nodes[node]["peak_preference"] = preference

                # Run election after dynamics
                results, voters_to_candidates = get_full_results_election(copied_graph, candidates)
                score = results[target_candidate.id]

                # Get nodes to exclude
                # If a node votes already for me, I exclude it from the seed candidates
                nodes_to_exclude = []
                for node, candidate_id in voters_to_candidates.items():
                    if candidate_id == target_candidate_id:
                        nodes_to_exclude.append(node)

                ##########
                # Evaluate marginal contributions
                ##########

                # Exclude nodes that are already in seeds
                all_nodes_without_seeds = list(filter(lambda element: element not in seeds, copied_graph.nodes()))
                logger.debug(f"len all_nodes: {len(all_nodes_without_seeds)} - len to exclude: {len(nodes_to_exclude)}")

                all_nodes_without_seeds = list(filter(lambda element: element not in nodes_to_exclude,
                                                      all_nodes_without_seeds))
                logger.debug(f"len all_nodes after exclusion: {len(all_nodes_without_seeds)}")

                # If the number of nodes to analyze is greater that the available number of nodes then pick few nodes
                min_number_of_nodes = min(number_of_iterations, len(all_nodes_without_seeds))
                if min_number_of_nodes != number_of_iterations:
                    logger.info(f"Number of iterations {number_of_iterations} is greater the the available nodes to"
                                f" analyze {len(all_nodes_without_seeds)}")

                # What happens if min_number_of_nodes is 0 or is fewer then the number of jobs?
                if min_number_of_nodes <= 0:
                    # All the nodes are in seeds or all the nodes votes for the target candidate
                    logger.info(f"Stopping prematurely compute seeds algorithm because"
                                f" all the non-seeds nodes votes already for the target candidate"
                                f" or the number of iterations is 0")
                    break

                total_performed_iterations += min_number_of_nodes  # for debug purposes

                # Choose the nodes for which to evaluate the marginal contribution
                chosen_nodes = random.sample(all_nodes_without_seeds, min_number_of_nodes)

                # Compute chunks
                # The chunks computation is robust to len(choses_nodes) <= number_of_jobs
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

                # Take the results
                nodes_to_contribution_dict = {}
                for result in results:
                    for node, value in result.items():
                        nodes_to_contribution_dict[node] = value

                # Take the node with the higher marginal contribution
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

                # Add max_node to seeds
                seeds[max_node] = max_preference

                # Update the graph with current seed
                copied_graph.nodes[max_node]["private_belief"] = seeds[max_node]
                copied_graph.nodes[max_node]["stubbornness"] = 1

                bar.update(1)

                end_time_s = time.time()

                # Updated estimate
                last_iteration_time_s = end_time_s - start_time_s
                remaining_seeds = number_of_seeds - i - 1
                if remaining_seeds > 0:
                    remaining_time -= last_iteration_time_s
                    number_of_iterations = _estimate_number_of_iterations_at_execution_time(last_iteration_time_s,
                                                                                        number_of_iterations,
                                                                                        remaining_seeds,
                                                                                        remaining_time,
                                                                                        number_of_jobs)

    logger.info(f"THE TOTAL NUMBER OF PERFORMED ITERATIONS IS: {total_performed_iterations}")

    return seeds
