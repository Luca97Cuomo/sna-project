import centrality_utils
from centrality_measures import *
import sys
import logging

sys.path.append('../')
import utils

PATH_TO_NODES = "../facebook_large/musae_facebook_target.csv"
PATH_TO_EDGES = "../facebook_large/musae_facebook_edges.csv"


def main():
    graph, _ = utils.load_graph_and_clusters(PATH_TO_NODES, PATH_TO_EDGES)

    centrality = centrality_utils.Centrality([
        (parallel_closeness_centrality, {"n_jobs": 8})],
        graph)

    centrality.evaluate_all()


if __name__ == "__main__":
    main()
