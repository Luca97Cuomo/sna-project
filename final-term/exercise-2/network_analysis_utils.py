import logging
import logging_configuration
import networkx as nx
from centrality_measures import *
from utils import *

logger = logging.getLogger(__name__)


class NetworkAnalyzer:
    def __init__(self, degree_mean_threshold, degree_std_threshold, network_diameter_threshold,
                 network_clustering_coefficient_threshold, target_network, network_generation_algorithms_with_kwargs):
        self.network_generation_algorithms_with_kwargs = network_generation_algorithms_with_kwargs
        self.target_network = target_network
        self.target_network_degree_mean, self.target_network_degree_std = analyze_degree_distribution(
            degree_centrality(target_network), "net_1", None, save=True)
        self.target_network_diameter = nx.algorithms.diameter(target_network)
        self.target_network_avg_clustering_coeff = nx.algorithms.cluster.average_clustering(target_network)
        logger.info(
            f"The target network has: degree mean = {self.target_network_degree_mean}, degree std = {self.target_network_degree_std}, "
            f"diameter = {self.target_network_diameter}, average clustering coefficient = {self.target_network_avg_clustering_coeff}")

        self.degree_mean_threshold = degree_mean_threshold
        self.degree_std_threshold = degree_std_threshold
        self.network_diameter_threshold = network_diameter_threshold
        self.network_clustering_coefficient_threshold = network_clustering_coefficient_threshold
        self.possible_models = []

    def __analyze_and_compare_network(self, network_generation_algorithm, kwargs):
        logger.info(f"Evaluating {network_generation_algorithm.__name__} algorithm, with these arguments : {kwargs}")
        network = network_generation_algorithm(**kwargs)
        node_to_degree = degree_centrality(network)
        if network_generation_algorithm.__name__ == "configurationG":
            degree_mean, degree_std = analyze_degree_distribution(node_to_degree, network_generation_algorithm.__name__,
                                                                  None, save=True)
        else:
            degree_mean, degree_std = analyze_degree_distribution(node_to_degree, network_generation_algorithm.__name__,
                                                                  list(kwargs.items()), save=True)
        network_diameter = None
        if nx.is_connected(network):
            network_diameter = nx.algorithms.diameter(network)
            logger.info(f"Network diameter : {network_diameter}")
        else:
            logger.info(f"The network is not connected, can not evaluate the diameter")

        avg_clustering_coeff = nx.algorithms.cluster.average_clustering(network)
        logger.info(f"Network average clustering coefficient : {avg_clustering_coeff}")

        possible_model = True
        if abs(self.target_network_degree_mean - degree_mean) <= self.degree_mean_threshold:
            logger.info(f"The current network has a degree mean that is within the threshold")
        else:
            possible_model = False

        if abs(self.target_network_degree_std - degree_std) <= self.degree_std_threshold:
            logger.info(f"The current network has a degree std that is within the threshold")
        else:
            possible_model = False

        if network_diameter is not None and abs(
                self.target_network_diameter - network_diameter) <= self.network_diameter_threshold:
            logger.info(f"The current network has a diameter that is within the threshold")
        else:
            possible_model = False

        if abs(self.target_network_avg_clustering_coeff - avg_clustering_coeff) <= self.network_clustering_coefficient_threshold:
            logger.info(f"The current network has a network clustering coefficient that is within the threshold")
        else:
            possible_model = False

        if possible_model:
            self.possible_models.append((network_generation_algorithm.__name__, kwargs))

        return

    def analyze_all(self):
        logger.info(f"Starting the analysis of the models proposed\n")
        for network_generation_algorithm, kwargs in self.network_generation_algorithms_with_kwargs:
            self.__analyze_and_compare_network(network_generation_algorithm, kwargs)

        logger.info(f"The analysis of the proprosed models is finished")
        if len(self.possible_models) == 0:
            logger.info(f"None of the evaluated models seems to reflect the target network")
        else:
            logger.info(f"This models : {self.possible_models} reflect the target network")
