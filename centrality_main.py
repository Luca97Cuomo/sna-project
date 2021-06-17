import mid_term.exercise_2.centrality_utils as centrality_utils
from mid_term.exercise_2.centrality_measures import *

import utils

PATH_TO_NODES = "facebook_large/musae_facebook_target.csv"
PATH_TO_EDGES = "facebook_large/musae_facebook_edges.csv"


def main():
    graph, _ = utils.load_graph_and_clusters(PATH_TO_NODES, PATH_TO_EDGES)

    centrality = centrality_utils.Centrality([
        (naive_edge_hits, {'max_iterations': 100}),
        # (parallel_edge_hits, {'max_iterations': 100}),
        (naive_hits, {'max_iterations': 100}),
        # (parallel_naive_hits, {'max_iterations': 100}),
    ],
        graph)

    centrality.evaluate_all()


if __name__ == "__main__":
    main()
