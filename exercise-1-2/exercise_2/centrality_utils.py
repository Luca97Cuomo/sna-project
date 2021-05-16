import sys
import numpy as np
from tqdm import tqdm

sys.path.append("../")
from pathlib import Path
import datetime
from priorityq import PriorityQueue
import logging
import time


class Centrality:
    def __init__(self, centrality_algorithms_with_kwargs, graph, seed=42, logger_level=logging.INFO, k=500,
                 output_file_path=f"centrality_results/results_{datetime.datetime.now().strftime('%d-%m-%H-%M-%S')}.txt"):
        self.centrality_algorithms_with_kwargs = centrality_algorithms_with_kwargs
        self.graph = graph
        self.seed = seed
        self.k = k
        self.output_file_path = Path(output_file_path).absolute()
        self.logger_level = logger_level
        logging.basicConfig(level=self.logger_level,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=str(self.output_file_path),
                            filemode='w+')

        console = logging.StreamHandler()
        console.setLevel(self.logger_level)
        formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)

    def __evaluate(self, centrality_algorithm, kwargs):
        logger = logging.getLogger(f"{centrality_algorithm.__name__}")
        logger.info(f"Evaluating {centrality_algorithm.__name__} algorithm, with these arguments : {kwargs}")
        start = time.perf_counter()
        node_to_centrality = centrality_algorithm(self.graph, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        logger.info(f"The centrality algorithm: {centrality_algorithm.__name__} took {elapsed} seconds")
        k_nodes = k_most_central_nodes(self.graph, node_to_centrality, self.k)
        logger.info(
            f"The {str(self.k)} most central nodes for the algorithm {centrality_algorithm.__name__} are : {k_nodes}")

    def evaluate_all(self):
        for algorithm, kwargs in self.centrality_algorithms_with_kwargs:
            self.__evaluate(algorithm, kwargs)


def k_most_central_nodes(graph, node_to_measure, k):
    pq = PriorityQueue()
    for node in graph.nodes():
        pq.add(node, -node_to_measure[node])
    out = []

    for i in range(k):
        node = pq.pop()
        out.append((node, node_to_measure[node]))

    return out


def print_centrality_results(node_to_centrality, output_dir_path, centrality_name):
    output_file_path = Path(output_dir_path).absolute()
    output_file_path = output_file_path.joinpath(
        f"results_{centrality_name}_{datetime.datetime.now().strftime('%d-%m-%H-%M-%S')}.txt")

    node_to_centrality.insert(0, ("Node", centrality_name))
    with open(output_file_path, "w") as output_file:
        for entry in node_to_centrality:
            output_file.write(str(entry[0]) + ":" + str(entry[1]) + "\n")
    return


def load_centrality_results(results_file_path):
    results_file_path = Path(results_file_path).absolute()
    results = []

    with open(results_file_path, "r") as results_file:
        content = results_file.readlines()
        content.pop(0)
        for line in content:
            line = line.strip().split(":")
            results.append((line[0], line[1]))

    return results


def compute_transition_matrix(graph, node_list):
    n = len(node_list)
    matrix = np.zeros((n, len(graph)))
    for index, node in enumerate(node_list):
        node_degree = graph.degree(node)
        for neighbor in graph[node]:
            matrix[index, :][neighbor] = 1 / node_degree

    return matrix
