import mid_term.exercise_2.centrality_utils as centrality_utils
from mid_term.exercise_2.centrality_measures import *

import utils

PATH_TO_NODES = "facebook_large/musae_facebook_target.csv"
PATH_TO_EDGES = "facebook_large/musae_facebook_edges.csv"


def main():
    graph, _ = utils.load_graph_and_clusters(PATH_TO_NODES, PATH_TO_EDGES)

    centrality = centrality_utils.Centrality([
        (degree_centrality, {}),
        (closeness_centrality, {'samples': list(graph.nodes())}),
        (parallel_closeness_centrality, {'n_jobs': 4}),
        (basic_page_rank, {'max_iterations': 100, 'delta_rel': 0.6}),
        (algebraic_page_rank, {'alpha': 1, 'max_iterations': 100, 'delta_rel': 0.6}),
        (parallel_basic_page_rank, {'alpha': 1, 'max_iterations': 100, 'jobs': 4, 'delta_rel': 0.6}),
        (naive_edge_hits, {'max_iterations': 100}),
        (parallel_edge_hits, {'max_iterations': 100}),
        (naive_hits, {'max_iterations': 100}),
        (parallel_naive_hits, {'max_iterations': 100}),
        (betweenness_centrality, {'nodes': list(graph.nodes())}),
        (parallel_betweenness_centrality, {}),
    ],
        graph)

    centrality.evaluate_all()


if __name__ == "__main__":
    main()
