import centrality_utils
import centrality_measures
import sys
import logging

sys.path.append('../')
import utils

PATH_TO_NODES = "../facebook_large/musae_facebook_target.csv"
PATH_TO_EDGES = "../facebook_large/musae_facebook_edges.csv"


def main():
    graph, _ = utils.load_graph_and_clusters(PATH_TO_NODES, PATH_TO_EDGES)
    """
    centrality = centrality_utils.Centrality([
        (centrality_measures.degree_centrality, {}),
        (centrality_measures.closeness_centrality, {}),
        (centrality_measures.basic_page_rank, {"max_iterations": 100}),
        (centrality_measures.algebraic_page_rank, {"alpha": 1, "max_iterations": 100}),
        (centrality_measures.hits, {"max_iterations": 100})],
        graph, logger_level=logging.DEBUG)
    """

    centrality = centrality_utils.Centrality([
        (centrality_measures.closeness_centrality, {})],
        graph, logger_level=logging.DEBUG)

    centrality.evaluate_all()


if __name__ == "__main__":
    main()
