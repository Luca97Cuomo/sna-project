from utils import *
from centrality_measures import *
import logging
import logging_configuration

NETWORK_PATH = "nets/net_1"


def main():
    network = load_network(NETWORK_PATH)

    node_to_degree = degree_centrality(network)
    plot_degree_distribution(node_to_degree)


if __name__ == "__main__":
    logging_configuration.set_logging()

    main()
