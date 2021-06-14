import logging
import logging.config
import random
import sys
import typing
import datetime
from pathlib import Path
import time
import getpass

import networkx as nx

import utils
from election import Candidate, run_election
from final_term_utils import populate_dynamics_parameters
from manipulators import *
from network_diffusion.fj_dynamics import fj_dynamics

FACEBOOK_PATH_TO_NODES = "../../../mid-term/exercise-1-2/facebook_large/musae_facebook_target.csv"
FACEBOOK_PATH_TO_EDGES = "../../../mid-term/exercise-1-2/facebook_large/musae_facebook_edges.csv"

logger = logging.getLogger("final_term_exercise_3_logger")


def run_experiment(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int, number_of_seeds: int,
                   compute_seeds: typing.Callable, seed: int, number_of_jobs: int,
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
    :param max_results_to_print:
    :return:                        A tuple in which the first element is the score obtained by the target candidate
                                    in the truthful election and the second element is the score obtained in the
                                    manipulated one.
    """

    # check input
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

    # run dynamics and election (for debugging purposes)
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
    seeds = compute_seeds(graph, candidates, target_candidate_id, number_of_seeds, seed, number_of_jobs)
    if len(seeds) > number_of_seeds:
        raise Exception(f"The length of computed seeds {len(seeds)} is greater "
                        f"than the number of seeds {number_of_seeds}")

    # update graph with seeds
    for node, preference in seeds.items():
        graph.nodes[node]["private_belief"] = preference
        graph.nodes[node]["stubbornness"] = 1

    manipulated_preferences = fj_dynamics(graph)

    # update graph with manipulated preferences
    for node, preference in manipulated_preferences.items():
        graph.nodes[node]["peak_preference"] = preference

    # run manipulated election
    results = run_election(graph, candidates)

    # logging
    logger.debug("")
    logger.debug("MANIPULATED ELECTION RESULTS")
    _log_election_results(results, max_results_to_print)

    manipulated_score = results[target_candidate_id]

    ##########

    logger.info("")
    logger.info(f"DIFFERENCE BETWEEN MANIPULATED AND DYNAMICS SCORE: {manipulated_score - dynamics_score}")

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
    results_dir = Path('experimental_results')
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
                "level": "INFO",
                "handlers": ["console-unnamed", "file-unnamed"]
            },
        },
    })


def main():
    # parameters
    SEED = 45
    POPULATE_DYNAMICS_SEED = SEED + 1
    RUN_EXPERIMENT_SEED = POPULATE_DYNAMICS_SEED + 1
    random.seed(SEED)

    NUMBER_OF_CANDIDATES = 10
    TARGET_CANDIDATE = random.randint(0, NUMBER_OF_CANDIDATES - 1)
    NUMBER_OF_SEEDS = 200
    COMPUTE_SEEDS = multi_level_greedy_manipulator_with_centrality_sampling
    GRAPH_NAME = "Facebook Graph"
    GRAPH, _ = utils.load_graph_and_clusters(FACEBOOK_PATH_TO_NODES, FACEBOOK_PATH_TO_EDGES)
    STUBBORNNESS = 0.5

    NUMBER_OF_JOBS = 14

    CANDIDATES = []
    for i in range(NUMBER_OF_CANDIDATES):
        position = random.random()
        CANDIDATES.append(Candidate(i, position))

    populate_dynamics_parameters(GRAPH, POPULATE_DYNAMICS_SEED, None, STUBBORNNESS)

    ##########
    # Logging
    ##########

    _set_logger_configuration(COMPUTE_SEEDS.__name__)

    # start logging a new experiment
    logger.info("\n##########\n")

    # log date
    logger.info(f"DATE: {datetime.datetime.now().strftime('%d-%m-%H-%M-%S')}")

    # log seeds
    logger.info(f"SEED: {SEED}, POPULATE_DYNAMICS_SEED: {POPULATE_DYNAMICS_SEED},"
                f" RUN_EXPERIMENT_SEED: {RUN_EXPERIMENT_SEED}")

    # log parameters
    logger.info(f"NUMBER_OF_CANDIDATES: {NUMBER_OF_CANDIDATES},"
                f" TARGET_CANDIDATE: {TARGET_CANDIDATE}, NUMBER_OF_SEEDS: {NUMBER_OF_SEEDS},"
                f" COMPUTE_SEEDS: {COMPUTE_SEEDS.__name__}, STUBBORNNESS: {STUBBORNNESS}")

    # log graph parameters
    logger.info(f"GRAPH_NAME: {GRAPH_NAME}, NUMBER_OF_NODES: {len(GRAPH.nodes)},"
                f" NUMBER_0F_EDGES: {len(GRAPH.edges)}")

    logger.info(f"NUMBER_OF_JOBS: {NUMBER_OF_JOBS}")

    logger.debug("")

    # log candidates positions
    logger.debug("CANDIDATES POSITIONS")
    for i in range(NUMBER_OF_CANDIDATES):
        candidate_id = CANDIDATES[i].id
        position = CANDIDATES[i].position
        logger.debug(f"CANDIDATE: {candidate_id}: {position}")

    logger.debug("")

    # measure time
    start_time = time.time()
    truthful_score, manipulated_score = run_experiment(GRAPH, CANDIDATES, TARGET_CANDIDATE, NUMBER_OF_SEEDS,
                                                       COMPUTE_SEEDS, RUN_EXPERIMENT_SEED, NUMBER_OF_JOBS)
    end_time = time.time()
    score_difference = manipulated_score - truthful_score

    logger.info("")

    # log score
    logger.info(f"TRUTHFUL_SCORE: {truthful_score}, MANIPULATED_SCORE: {manipulated_score},"
                f" SCORE_DIFFERENCE: {score_difference}")

    logger.info("")

    # log time
    logger.info(f"RUN_TIME: {end_time - start_time} seconds. ({getpass.getuser()})")

    # end logging for this experiment
    logger.info("\n##########\n")


if __name__ == "__main__":
    main()
