import centrality_utils
import centrality_measures
import sys
sys.path.append('../')
import utils
import networkx as nx
import time

PATH_TO_NODES = "../facebook_large/musae_facebook_target.csv"
PATH_TO_EDGES = "../facebook_large/musae_facebook_edges.csv"


def main():
    graph, true_clusters = utils.load_graph(PATH_TO_NODES, PATH_TO_EDGES)
    # graph = utils.build_random_graph(500, 0.40, 4)

    """
    start = time.time()
    results = nx.algorithms.link_analysis.pagerank_alg.pagerank(graph)
    end = time.time()

    print(end - start) # seconds
    """

    #results = centrality_measures.basic_page_rank(graph)
    #print(results)

if __name__ == "__main__":
    main()
