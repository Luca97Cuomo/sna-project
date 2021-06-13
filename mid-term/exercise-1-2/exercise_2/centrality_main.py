import centrality_utils
from centrality_measures import *
import sys
import logging
from clustering_utils import CENTRALITY_MEASURES

import utils

PATH_TO_NODES = "../facebook_large/musae_facebook_target.csv"
PATH_TO_EDGES = "../facebook_large/musae_facebook_edges.csv"


def main():
    graph, _ = utils.load_graph_and_clusters(PATH_TO_NODES, PATH_TO_EDGES)

    centrality = centrality_utils.Centrality([
        (basic_page_rank, {"max_iterations": 10000, "delta_rel": 0.4}),
        (parallel_basic_page_rank, {"max_iterations": 10000, "delta_rel": 0.4, "jobs": 8}),
        (algebraic_page_rank, {"max_iterations": 10000, "alpha": 1, "delta_rel": 0.4}),

        (basic_page_rank, {"max_iterations": 10000, "delta_rel": 0.6}),
        (parallel_basic_page_rank, {"max_iterations": 10000, "delta_rel": 0.6, "jobs": 8}),
        (algebraic_page_rank, {"max_iterations": 10000, "alpha": 1, "delta_rel": 0.6}),

        (basic_page_rank, {"max_iterations": 10000, "delta_rel": 0.8}),
        (parallel_basic_page_rank, {"max_iterations": 10000, "delta_rel": 0.8, "jobs": 8}),
        (algebraic_page_rank, {"max_iterations": 10000, "alpha": 1, "delta_rel": 0.8}),

        (basic_page_rank, {"max_iterations": 10000, "delta_rel": 1}),
        (parallel_basic_page_rank, {"max_iterations": 10000, "delta_rel": 1, "jobs": 8}),
        (algebraic_page_rank, {"max_iterations": 10000, "alpha": 1, "delta_rel": 1}),

    ],
        graph)

    centrality.evaluate_all()


if __name__ == "__main__":
    main()
