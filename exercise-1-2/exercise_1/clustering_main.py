import clustering_utils
import clustering_algorithms
import sys
sys.path.append('../')
import utils

PATH_TO_NODES = "../facebook_large/musae_facebook_target.csv"
PATH_TO_EDGES = "../facebook_large/musae_facebook_edges.csv"


def main():
    graph, true_clusters = utils.load_graph(PATH_TO_NODES, PATH_TO_EDGES)
    # graph = utils.build_random_graph(200, 0.30, 4)
    """
    clustering = utils.Clustering([
        (clustering_algorithms.k_means_one_iteration, {"seed": 42, "k": 4}),
        (clustering_algorithms.k_means, {"centrality_measure": None, "seed": 42, "k": 4,
                                         "equality_threshold": 0.1, "max_iterations": 1, "centers": None}),
        (clustering_algorithms.k_means, {"centrality_measure": None, "seed": 42, "k": 4,
                                         "equality_threshold": 0.001, "max_iterations": 10000,
                                         "centers": None, "verbose": True}),
        (clustering_algorithms.k_means, {"centrality_measure": "degree_centrality", "seed": 42, "k": 4,
                                         "equality_threshold": 0.001, "max_iterations": 10000,
                                         "centers": None, "verbose": True}),
        (clustering_algorithms.k_means, {"centrality_measure": "betweenness_centrality", "seed": 42, "k": 4,
                                         "equality_threshold": 0.001, "max_iterations": 50,
                                         "centers": None, "verbose": True})],
        graph, true_clusters, verbose=True, draw_graph=False)

    clustering.evaluate_all()
    """
    clustering = clustering_utils.Clustering([
        (clustering_algorithms.spectral, {"k": 4})], graph,
        true_clusters,
        verbose=True, draw_graph=False)

    clustering.evaluate_all()


if __name__ == "__main__":
    main()
