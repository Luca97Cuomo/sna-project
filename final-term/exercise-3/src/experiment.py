import logging
import logging.config
import random
import sys
import typing
import datetime
from pathlib import Path
import time

import networkx as nx

import utils
from election import Candidate, run_election
from final_term_utils import populate_dynamics_parameters
from manipulators import bogo_manipulator
from network_diffusion.fj_dynamics import fj_dynamics

FACEBOOK_PATH_TO_NODES = "../../../mid-term/exercise-1-2/facebook_large/musae_facebook_target.csv"
FACEBOOK_PATH_TO_EDGES = "../../../mid-term/exercise-1-2/facebook_large/musae_facebook_edges.csv"


def run_experiment(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate: int, number_of_seeds: int,
                   compute_seeds: typing.Callable, seed: int = 42) -> typing.Tuple[int, int]:
    """
    :param graph:                   The graph of the voters. Each node has the attribute private_belief
                                    and stubbornness. The stubbornness value should 0.5 for each node.
    :param candidates:
    :param target_candidate:        The id of the candidate that has to be publicized.
    :param number_of_seeds:
    :param compute_seeds:
    :param seed:
    :return:                        A tuple in which the first element is the score obtained by the target candidate
                                    in the truthful election and the second element is the score obtained in the
                                    manipulated one.
    """

    # run truthful election
    results = run_election(graph, candidates)
    truthful_score = results[target_candidate]

    # seeds is a dict {node: preference}
    seeds = compute_seeds(graph, candidates, target_candidate, number_of_seeds, seed)

    # update graph with seeds
    for node, preference in seeds.items():
        graph.nodes[node]["private_belief"] = preference
        graph.nodes[node]["stubbornness"] = 1

    manipulated_preferences = fj_dynamics(graph)

    # update graph with manipulated preferences
    for node, preference in manipulated_preferences.items():
        graph.nodes[node]["private_belief"] = preference

    # run manipulated election
    results = run_election(graph, candidates)
    manipulated_score = results[target_candidate]

    return truthful_score, manipulated_score


def main():
    # logger configuration
    results_dir = Path('results')
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
                "filename": results_dir / f"experiments_{datetime.datetime.now().strftime('%d-%m')}.txt",
                "formatter": "unnamed"
            },
        },
        "loggers": {
            "final_term_exercise_3_logger": {
                "level": "INFO",
                "handlers": ["console-unnamed", "file-unnamed"]
            },
        },
    })

    logger = logging.getLogger("final_term_exercise_3_logger")
    logger.setLevel(logging.INFO)

    SEED = 45
    POPULATE_DYNAMICS_SEED = SEED + 1
    RUN_EXPERIMENT_SEED = POPULATE_DYNAMICS_SEED + 1
    random.seed(SEED)

    NUMBER_OF_CANDIDATES = 2
    TARGET_CANDIDATE = random.randint(0, NUMBER_OF_CANDIDATES - 1)
    NUMBER_OF_SEEDS = 10
    COMPUTE_SEEDS = bogo_manipulator
    GRAPH_NAME = "Facebook Graph"
    GRAPH, _ = utils.load_graph_and_clusters(FACEBOOK_PATH_TO_NODES, FACEBOOK_PATH_TO_EDGES)
    STUBBORNNESS = 0.5

    CANDIDATES = []
    for i in range(NUMBER_OF_CANDIDATES):
        CANDIDATES.append(Candidate(i, random.random()))

    populate_dynamics_parameters(GRAPH, POPULATE_DYNAMICS_SEED, None, STUBBORNNESS)

    # measure time
    start_time = time.time()
    truthful_score, manipulated_score = run_experiment(GRAPH, CANDIDATES, TARGET_CANDIDATE, NUMBER_OF_SEEDS,
                                                       COMPUTE_SEEDS, RUN_EXPERIMENT_SEED)
    end_time = time.time()
    score_difference = manipulated_score - truthful_score

    ##########
    # Logging
    ##########

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

    # log score
    logger.info(f"TRUTHFUL_SCORE: {truthful_score}, MANIPULATED_SCORE: {manipulated_score},"
                f" SCORE_DIFFERENCE: {score_difference}")

    # log time
    logger.info(f"RUN_TIME: {end_time - start_time} seconds")

    # end logging for this experiment
    logger.info("\n##########\n")


if __name__ == "__main__":
    main()
