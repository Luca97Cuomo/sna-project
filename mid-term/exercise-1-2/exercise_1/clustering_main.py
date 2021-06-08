import logging

import clustering_utils
import clustering_algorithms
import sys

import utils

PATH_TO_NODES = "../facebook_large/musae_facebook_target.csv"
PATH_TO_EDGES = "../facebook_large/musae_facebook_edges.csv"


def main():
    graph, true_clusters = utils.load_graph_and_clusters(PATH_TO_NODES, PATH_TO_EDGES)

    clustering = clustering_utils.Clustering([
        (clustering_algorithms.hierarchical_optimized, {"seed": 42, "desired_clusters": 4}),
        (clustering_algorithms.k_means, {"centrality_measure": None, "seed": 42, "k": 4,
                                         "equality_threshold": 0.001, "max_iterations": 10000,
                                         "centers": None, "verbose": False}),
        (clustering_algorithms.k_means, {"centrality_measure": "degree_centrality", "seed": 42, "k": 4,
                                         "equality_threshold": 0.001, "max_iterations": 10000,
                                         "centers": None, "verbose": False}),
        (clustering_algorithms.k_means, {"centrality_measure": "closeness_centrality", "seed": 42, "k": 4,
                                         "equality_threshold": 0.001, "max_iterations": 10,
                                         "centers": None, "verbose": False}),
        (clustering_algorithms.k_means, {"centrality_measure": "nodes_betweenness_centrality", "seed": 42, "k": 4,
                                         "equality_threshold": 0.001, "max_iterations": 10,
                                         "centers": None, "verbose": False}),
        (clustering_algorithms.k_means, {"centrality_measure": "pagerank", "seed": 42, "k": 4,
                                         "equality_threshold": 0.001, "max_iterations": 50,
                                         "centers": None, "verbose": False}),
        (clustering_algorithms.girvan_newman, {"centrality_measure": "edges_betweenness_centrality", "seed": 42,
                                               "k": 4, "verbose": False, "optimized": True}),
        (clustering_algorithms.spectral, {"k": 4})],
        graph, true_clusters, draw_graph=False)

    clustering.evaluate_all()


if __name__ == "__main__":
    main()
