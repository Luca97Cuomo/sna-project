import centrality_utils
import centrality_measures
import sys
sys.path.append('../')
import utils

PATH_TO_NODES = "../facebook_large/musae_facebook_target.csv"
PATH_TO_EDGES = "../facebook_large/musae_facebook_edges.csv"


def main():
    # graph, true_clusters = utils.load_graph(PATH_TO_NODES, PATH_TO_EDGES)
    graph = utils.build_random_graph(10, 0.40, 4)


if __name__ == "__main__":
    main()
