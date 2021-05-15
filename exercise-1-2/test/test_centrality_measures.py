import time
from unittest import TestCase
import networkx as nx
import sys
import numpy as np
from pytest import approx

sys.path.append("../")
import utils

from exercise_2 import centrality_measures, centrality_utils

PATH_TO_NODES = "../facebook_large/musae_facebook_target.csv"
PATH_TO_EDGES = "../facebook_large/musae_facebook_edges.csv"


class TestCentralityMeasures(TestCase):
    def setUp(self) -> None:
        self.graph = utils.build_random_graph(1000, 0.30, seed=42)
        self.facebook_graph, _ = utils.load_graph(PATH_TO_NODES, PATH_TO_EDGES)

        self.small_graph = nx.Graph()
        self.small_graph.add_edge(0, 1)
        self.small_graph.add_edge(0, 2)
        self.small_graph.add_edge(0, 3)

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
        expected = centrality_measures.basic_page_rank(self.facebook_graph, max_iterations=100)
        results = centrality_measures.algebraic_page_rank(self.facebook_graph, max_iterations=100, alpha=1)
        for node in self.facebook_graph.nodes():
            self.assertEqual(approx(expected[node], rel=0.5), results[node])

    def test_transition_matrix(self):
        alpha = 1
        graph = self.facebook_graph
        node_list = list(range(len(graph)))
        google_matrix_start = time.perf_counter()
        google_matrix = nx.algorithms.link_analysis.pagerank_alg.google_matrix(graph, alpha=alpha, nodelist=node_list)
        google_matrix_end = time.perf_counter()

        our_matrix = centrality_utils.compute_transition_matrix(graph, node_list)
        our_matrix_end = time.perf_counter()

        print(
            f"The google matrix took {google_matrix_end - google_matrix_start} seconds, our matrix tool {our_matrix_end - google_matrix_end} seconds")
        google_matrix_list = google_matrix.tolist()
        print("Google matrix list done")
        for i in range(len(graph)):
            for j in range(len(graph)):
                self.assertAlmostEqual(our_matrix[i][j], google_matrix_list[i][j], delta=1e-2)

    def test_node_names(self):
        facebook_graph, true_clusters = utils.load_graph(PATH_TO_NODES, PATH_TO_EDGES)

        for i in range(len(facebook_graph)):
            self.assertTrue(facebook_graph.has_node(i))
