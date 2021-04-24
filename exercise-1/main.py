import utils
import clustering_algorithms

PATH_TO_NODES = "facebook_large\\musae_facebook_target.csv"
PATH_TO_EDGES = "facebook_large\\musae_facebook_edges.csv"


def main():
    graph, node_to_labels = utils.load_graph(PATH_TO_NODES, PATH_TO_EDGES)
    # graph, node_to_labels = utils.build_random_graph(11, 0.20, 4)

    clustering = utils.Clustering([clustering_algorithms.hierarchical_optimized], graph, node_to_labels, verbose=True, draw_graph=False)
    clustering.evaluate_all()


if __name__ == "__main__":
    main()
