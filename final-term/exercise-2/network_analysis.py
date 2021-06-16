from utils import *
import logging
import logging_configuration
from network_analysis_utils import NetworkAnalyzer
from lesson4 import *
import numpy as np
from centrality_measures import *

NETWORK_PATH = "nets/net_1"
NETWORK_NUM_NODES = 10000
DEGREE_MEAN_THRESHOLD = 15
DEGREE_STD_THRESHOLD = 10
DIAMETER_THRESHOLD = 2
AVG_CLUSTERING_COEF_THRESHOLD = 0.20
logger = logging.getLogger(__name__)



def main():
    target_network = load_network(NETWORK_PATH)

    network_analyzer = NetworkAnalyzer(degree_mean_threshold=DEGREE_MEAN_THRESHOLD,
                                       degree_std_threshold=DEGREE_STD_THRESHOLD,
                                       network_diameter_threshold=DIAMETER_THRESHOLD,
                                       network_clustering_coefficient_threshold=AVG_CLUSTERING_COEF_THRESHOLD,
                                       target_network=target_network, network_generation_algorithms_with_kwargs=[
            (GenWS2DG, {"n": NETWORK_NUM_NODES, "r": 5, "k": 11, "q": 3}),
        ])

    network_analyzer.analyze_all()

if __name__ == "__main__":
    logging_configuration.set_logging()

    main()
