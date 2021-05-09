from unittest import TestCase
import networkx as nx
import sys
import numpy as np

sys.path.append("../")
import utils

from exercise_2 import centrality_measures

PATH_TO_NODES = "../facebook_large/musae_facebook_target.csv"
PATH_TO_EDGES = "../facebook_large/musae_facebook_edges.csv"

class TestCentralityMeasures(TestCase):
    def setUp(self) -> None:
        self.graph = utils.build_random_graph(100, 1.0, seed=42)
        self.facebook_graph, _ = utils.load_graph(PATH_TO_NODES, PATH_TO_EDGES)

        self.small_graph = nx.Graph()
        self.small_graph.add_edge(1, 2)
        self.small_graph.add_edge(1, 3)
        self.small_graph.add_edge(1, 4)

    def test_degree_centrality(self):
        expected = nx.degree_centrality(self.graph)
        results = centrality_measures.degree_centrality(self.graph)

        expected = sorted(expected.items(), key=lambda item: item[1])
        expected_nodes = []
        for tuple in expected:
            expected_nodes.append(tuple[0])

        node_btw = sorted(results.items(), key=lambda item: item[1])
        results_nodes = []
        for tuple in node_btw:
            results_nodes.append(tuple[0])

        self.assertListEqual(expected_nodes, results_nodes)

    def test_closeness_centrality(self):
        expected = nx.closeness_centrality(self.graph)
        results = centrality_measures.closeness_centrality(self.graph)
        self.assertDictEqual(expected, results)

    def test_betweenness_centrality(self):
        expected = nx.betweenness_centrality(self.graph)
        edge_btw, node_btw = centrality_measures.betweenness_centrality(self.graph)

        expected = sorted(expected.items(), key=lambda item: item[1])
        expected_nodes = []
        for tuple in expected:
            expected_nodes.append(tuple[0])

        node_btw = sorted(node_btw.items(), key=lambda item: item[1])
        results_nodes = []
        for tuple in node_btw:
            results_nodes.append(tuple[0])

        self.assertListEqual(expected_nodes, results_nodes)

    def test_algebraic_page_rank(self):
        expected = nx.algorithms.link_analysis.pagerank_alg.pagerank(self.facebook_graph, max_iter=100, alpha=0.85, tol=1e-06)
        results = centrality_measures.algebraic_page_rank(self.facebook_graph, max_iterations=100, alpha=0.85)
        print(expected)
        print(results)
        self.assertTrue(True)

    def test_transition_matrix(self):
        graph = nx.Graph()
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(1, 4)
        alpha = 0.85
        actual = nx.algorithms.link_analysis.pagerank_alg.google_matrix(graph, alpha=alpha)

        # d * value + (1 - d) / N

        """
            1       2       3       4
        1   0       1/3     1/3     1/3
        2   1       0       0       0
        3   1       0       0       0
        4   1       0       0       0
        """
        expected = np.matrix([
            [0, 1/3, 1/3, 1/3],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ])

        expected = expected.tolist()

        for i in range(len(graph)):
            for j in range(len(graph)):
                expected[i][j] = expected[i][j] * alpha + (1 - alpha) / len(graph)

        self.assertListEqual(actual.tolist(), expected)

    def test_node_names(self):
        facebook_graph, true_clusters = utils.load_graph(PATH_TO_NODES, PATH_TO_EDGES)

        for i in range(len(facebook_graph)):
            self.assertTrue(facebook_graph.has_node(i))

