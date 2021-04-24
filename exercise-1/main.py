import utils
import clustering_algorithms

PATH_TO_NODES = "facebook_large\\musae_facebook_target.csv"
PATH_TO_EDGES = "facebook_large\\musae_facebook_edges.csv"


def main():
    graph, true_clusters = utils.load_graph(PATH_TO_NODES, PATH_TO_EDGES)
    # graph, node_to_labels = utils.build_random_graph(11, 0.20, 4)

    clustering = utils.Clustering([
        # (clustering_algorithms.hierarchical_optimized, {"seed": 42, "desired_clusters": 4}),
        (clustering_algorithms.k_means_one_iteration, {"seed": 42, "k": 4})
    ], graph, true_clusters, verbose=True, draw_graph=False)
    clustering.evaluate_all()


if __name__ == "__main__":
    main()
