import logging
import logging.config
import os
import random
import sys
import typing
import datetime
from pathlib import Path
import time
import getpass

import networkx as nx

import utils
from final_term.exercise_3.src.election import Candidate, run_election
from final_term.final_term_utils import populate_dynamics_parameters
from final_term.exercise_3.src.manipulators.manipulators import timed_multi_level_greedy_manipulator
from final_term.exercise_1.src.network_diffusion.fj_dynamics import fj_dynamics

FACEBOOK_PATH_TO_NODES = "facebook_large/musae_facebook_target.csv"
FACEBOOK_PATH_TO_EDGES = "facebook_large/musae_facebook_edges.csv"

logger = logging.getLogger("final_term_exercise_3_logger")


def _set_number_of_jobs(number_of_free_cpus: int):
    number_of_jobs = os.cpu_count()
    if number_of_jobs is None:
        print(f"WARNING: No cpus were detected. Using one job.")
        return 1

    number_of_jobs -= number_of_free_cpus
    if number_of_jobs < 1:
        return 1

    return number_of_jobs


COMPUTE_SEEDS = timed_multi_level_greedy_manipulator

SEED = 45
POPULATE_DYNAMICS_SEED = SEED + 1
RUN_EXPERIMENT_SEED = POPULATE_DYNAMICS_SEED + 1

MAX_RUNNING_TIME_S = 14400  # 4h

NUMBER_OF_CANDIDATES = 10
NUMBER_OF_SEEDS = 200

GRAPH_NAME = "Facebook Graph"
GRAPH, _ = utils.load_graph_and_clusters(FACEBOOK_PATH_TO_NODES, FACEBOOK_PATH_TO_EDGES)

# path = pathlib.Path("../../graph-3000-4000")

# with open(path, 'rb') as file:
#    GRAPH = pickle.load(file)


def run_experiment(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int, number_of_seeds: int,
                   compute_seeds: typing.Callable, seed: int, number_of_jobs: int, max_running_time_s: int,
                   max_results_to_print: int = 10) -> typing.Tuple[int, int]:
    """
    :param graph:                   The graph of the voters. Each node has the attributes private_belief
                                    and stubbornness. The stubbornness value should 0.5 for each node.
    :param candidates:
    :param target_candidate_id:        The id of the candidate that has to be publicized.
    :param number_of_seeds:
    :param compute_seeds:
    :param seed:
    :param number_of_jobs:
    :param max_running_time_s:
    :param max_results_to_print:
    :return:                        A tuple in which the first element is the score obtained by the target candidate
                                    in the truthful election and the second element is the score obtained in the
                                    manipulated one.
    """

    ##########
    # Logging
    ##########

    _set_logger_configuration(compute_seeds.__name__)

    # start logging a new experiment
    logger.info("\n##########\n")

    # log date
    logger.info(f"DATE: {datetime.datetime.now().strftime('%d-%m-%H-%M-%S')}")

    # log seeds
    logger.info(f"SEED: {SEED}, POPULATE_DYNAMICS_SEED: {POPULATE_DYNAMICS_SEED},"
                f" RUN_EXPERIMENT_SEED: {RUN_EXPERIMENT_SEED}")

    # log parameters
    logger.info(f"NUMBER_OF_CANDIDATES: {len(candidates)},"
                f" TARGET_CANDIDATE: {target_candidate_id}, NUMBER_OF_SEEDS: {number_of_seeds},"
                f" COMPUTE_SEEDS: {compute_seeds.__name__}")

    # log max time
    logger.info(f"\nMAX RUNNING TIME S: {max_running_time_s}")

    # log graph parameters
    logger.info(f"NUMBER_OF_NODES: {len(graph.nodes)},"
                f" NUMBER_0F_EDGES: {len(graph.edges)}")

    logger.info(f"NUMBER_OF_JOBS: {number_of_jobs}")

    ##########

    start_time = time.time()

    # Check input
    target_is_in_candidate = False
    for candidate in candidates:
        if target_candidate_id == candidate.id:
            target_is_in_candidate = True
            break
    if not target_is_in_candidate:
        raise Exception(f"Target candidate {target_candidate_id} is not in candidates")

    if number_of_seeds > len(graph.nodes):
        raise Exception(f"Number of seeds {number_of_seeds} is greater that tha number of nodes {len(graph.nodes)}")

    ##########
    # Evaluate truthful score
    ##########

    # set peak_preferences
    for node in graph.nodes():
        graph.nodes[node]["peak_preference"] = graph.nodes[node]["private_belief"]

    # run truthful election
    results = run_election(graph, candidates)

    # logging
    logger.debug("TRUTHFUL ELECTION RESULTS")
    _log_election_results(results, max_results_to_print)

    truthful_score = results[target_candidate_id]

    ##########

    # run dynamics (with 5 digits) and election (for debugging purposes)
    preferences_after_dynamics = fj_dynamics(graph)

    # update graph after dynamics
    for node, preference in preferences_after_dynamics.items():
        graph.nodes[node]["peak_preference"] = preference

    # run election after dynamics
    results = run_election(graph, candidates)

    # logging
    logger.debug("")
    logger.debug("ELECTION RESULTS AFTER DYNAMICS")
    _log_election_results(results, max_results_to_print)

    dynamics_score = results[target_candidate_id]

    ##########

    # compute seeds
    seeds = compute_seeds(graph, candidates, target_candidate_id, number_of_seeds,
                          seed, number_of_jobs, max_running_time_s=max_running_time_s)
    logger.info(f"Computed {len(seeds)} seeds. The budget was {number_of_seeds}")
    if len(seeds) > number_of_seeds:
        raise Exception(f"The length of computed seeds {len(seeds)} is greater "
                        f"than the number of seeds {number_of_seeds}")

    # Update graph with seeds
    for node, preference in seeds.items():
        graph.nodes[node]["private_belief"] = preference
        graph.nodes[node]["stubbornness"] = 1

    manipulated_preferences = fj_dynamics(graph)

    # Update graph with manipulated preferences
    for node, preference in manipulated_preferences.items():
        graph.nodes[node]["peak_preference"] = preference

    # run manipulated election
    results = run_election(graph, candidates)
    manipulated_score = results[target_candidate_id]

    end_time = time.time()

    ############

    score_difference = manipulated_score - truthful_score

    logger.info("")

    # log score
    logger.info(f"TRUTHFUL_SCORE: {truthful_score}, MANIPULATED_SCORE: {manipulated_score},"
                f" SCORE_DIFFERENCE: {score_difference}")

    # log time
    logger.info(f"RUN_TIME: {end_time - start_time} seconds. ({getpass.getuser()})")

    # logging
    logger.debug("")
    logger.debug("MANIPULATED ELECTION RESULTS")
    _log_election_results(results, max_results_to_print)

    ##########

    logger.info("")
    logger.info(f"DIFFERENCE BETWEEN MANIPULATED AND DYNAMICS SCORE: {manipulated_score - dynamics_score}")

    # end logging for this experiment
    logger.info("\n##########\n")

    ##########

    return truthful_score, manipulated_score


def _log_election_results(results: typing.Dict[int, int], max_results_to_print: int) -> None:
    ordered_results = sorted(results.items(), key=lambda element: -element[1])

    count = 0
    for candidate_id, score in ordered_results:
        if count == max_results_to_print:
            break

        logger.debug(f"CANDIDATE ID: {candidate_id}: SCORE: {score}")
        count += 1


def _set_logger_configuration(compute_seeds_function_name) -> None:
    # logger configuration
    results_dir = Path('src/final_term/exercise_3/src/experimental_results')
    results_dir.mkdir(exist_ok=True)

    logging.config.dictConfig({
        "version": 1,
        "formatters": {
            "unnamed": {
                "format": "%(message)s"
            }
        },
        "handlers": {
            "console-unnamed": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "unnamed"
            },
            "file-unnamed": {
                "class": "logging.FileHandler",
                "filename": results_dir / f"experiments_{compute_seeds_function_name}_"
                                          f"{datetime.datetime.now().strftime('%d-%m-%H-%M-%S')}.txt",
                "formatter": "unnamed"
            },
        },
        "loggers": {
            "final_term_exercise_3_logger": {
                "propagate": False,
                "level": "WARNING",
                "handlers": ["console-unnamed", "file-unnamed"]
            },
        },
    })


def manipulation(G: nx.Graph, p: typing.List[float], c: int, B: int, b: typing.List[float]):
    # populate graph
    index = 0
    node_list = G.nodes()
    for node in node_list:
        G.nodes[node]["private_belief"] = b[index]
        G.nodes[node]["stubbornness"] = 0.5
        index += 1

    # candidates
    candidates = []
    for i in range(len(p)):
        candidates.append(Candidate(i, p[i]))

    number_of_jobs = _set_number_of_jobs(number_of_free_cpus=1)

    truthful_score, manipulated_score = run_experiment(G, candidates, c, B,
                                                       COMPUTE_SEEDS, RUN_EXPERIMENT_SEED,
                                                       number_of_jobs, MAX_RUNNING_TIME_S)

    print(f"1,{truthful_score},{manipulated_score}")


def main():
    random.seed(SEED)

    target_candidate_index = random.randint(0, NUMBER_OF_CANDIDATES - 1)
    candidate_positions = []
    for i in range(NUMBER_OF_CANDIDATES):
        candidate_positions.append(random.random())

    nodes_private_belief = []
    for i in range(len(GRAPH.nodes())):
        nodes_private_belief.append(random.random())

    manipulation(GRAPH, candidate_positions, target_candidate_index, NUMBER_OF_SEEDS, nodes_private_belief)


def set_parameters_and_launch_experiment():
    # parameters
    random.seed(SEED)

    TARGET_CANDIDATE = random.randint(0, NUMBER_OF_CANDIDATES - 1)

    NUMBER_OF_JOBS = _set_number_of_jobs(2)

    STUBBORNNESS = 0.5

    CANDIDATES = []
    for i in range(NUMBER_OF_CANDIDATES):
        position = random.random()
        CANDIDATES.append(Candidate(i, position))

    populate_dynamics_parameters(GRAPH, POPULATE_DYNAMICS_SEED, None, STUBBORNNESS)

    # log candidates positions
    logger.debug("\nCANDIDATES POSITIONS")
    for i in range(NUMBER_OF_CANDIDATES):
        candidate_id = CANDIDATES[i].id
        position = CANDIDATES[i].position
        logger.debug(f"CANDIDATE: {candidate_id}: {position}")

    logger.debug("")

    # measure time
    truthful_score, manipulated_score = run_experiment(GRAPH, CANDIDATES, TARGET_CANDIDATE, NUMBER_OF_SEEDS,
                                                       COMPUTE_SEEDS, RUN_EXPERIMENT_SEED,
                                                       NUMBER_OF_JOBS, MAX_RUNNING_TIME_S)


if __name__ == "__main__":
    main()
