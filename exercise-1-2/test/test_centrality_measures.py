import time
from unittest import TestCase
import networkx as nx
import sys
import numpy as np
from pytest import approx

sys.path.append("../")
import utils

from exercise_2 import centrality_measures, centrality_utils
from exercise_2 import betweenness_centrality

PATH_TO_NODES = "../facebook_large/musae_facebook_target.csv"
PATH_TO_EDGES = "../facebook_large/musae_facebook_edges.csv"


class TestCentralityMeasures(TestCase):
    def setUp(self) -> None:
        self.graph = utils.build_random_graph(100, 0.10, seed=42)
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

    def test_naive_betweenness_centrality(self):
        # required time for: self.graph = utils.build_random_graph(1000, 0.10, seed=42) is:
        # 141.765s

        expected = nx.betweenness_centrality(self.graph)
        edge_btw, node_btw = betweenness_centrality.lecture_betweenness_centrality(self.graph)

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
        expected = centrality_measures.basic_page_rank(self.facebook_graph, max_iterations=200, delta=1e-5)
        results = centrality_measures.algebraic_page_rank(self.facebook_graph, max_iterations=200, alpha=1, delta=1e-5)
        for node in self.facebook_graph.nodes():
            self.assertEqual(approx(expected[node], rel=0.5), results[node])

    def test_parallel_page_rank(self):
        actual = centrality_measures.parallel_basic_page_rank(self.facebook_graph, max_iterations=200, delta=1e-5, jobs=4)
        expected = centrality_measures.basic_page_rank(self.facebook_graph, max_iterations=200, delta=1e-5)
        for node in self.facebook_graph.nodes():
            self.assertEqual(approx(expected[node], rel=0.5), actual[node])


    def test_transition_matrix(self):
        alpha = 1
        graph = self.facebook_graph
        node_list = list(range(len(graph)))
        google_matrix_start = time.perf_counter()
        google_matrix = nx.algorithms.link_analysis.pagerank_alg.google_matrix(graph, alpha=alpha, nodelist=node_list)
        google_matrix_end = time.perf_counter()

        transition_matrix = centrality_utils.compute_transition_matrix(graph, node_list)
        transition_matrix_end = time.perf_counter()

        print(
            f"The google matrix took {google_matrix_end - google_matrix_start} seconds, transition matrix took {transition_matrix_end - google_matrix_end} seconds")
        google_matrix_list = google_matrix.tolist()
        for i in range(len(graph)):
            for j in range(len(graph)):
                self.assertEqual(approx(transition_matrix[i][j], rel=0.5), google_matrix_list[i][j])

    def test_hits(self):
        graph = self.facebook_graph
        nx_start_time = time.perf_counter()
        nx_hubs, nx_authorities = nx.algorithms.link_analysis.hits_alg.hits(graph, max_iter=100)
        nx_end_time = time.perf_counter()

        hubs, authorities = centrality_measures.hits(graph, max_iterations=100)
        end_time = time.perf_counter()

        print(f"The nx hits took {nx_end_time - nx_start_time} seconds, hits took {end_time - nx_end_time} seconds")

        for node in graph.nodes():
            self.assertEqual(approx(nx_hubs[node], rel=1e-1), hubs[node])
            self.assertEqual(approx(nx_authorities[node], rel=1e-1), authorities[node])


    def test_parallel_hits(self):
        graph = self.graph
        expected_hubs, expected_authorities = centrality_measures.hits(graph, max_iterations=100)
        result_hubs, result_authorities = centrality_measures.parallel_hits(graph, max_iterations=100)

        for node in graph.nodes():
            self.assertEqual(approx(expected_hubs[node], rel=1e-1), result_hubs[node])
            self.assertEqual(approx(expected_authorities[node], rel=1e-1), result_authorities[node])

    def test_node_names(self):
        facebook_graph, true_clusters = utils.load_graph(PATH_TO_NODES, PATH_TO_EDGES)

        for i in range(len(facebook_graph)):
            self.assertTrue(facebook_graph.has_node(i))


    def test_naive_betweenness_bfs(self):
        bfs_graph = nx.Graph()
        bfs_graph.add_edge(0, 1)
        bfs_graph.add_edge(1, 2)
        bfs_graph.add_edge(1, 3)
        bfs_graph.add_edge(3, 4)
        bfs_graph.add_edge(2, 4)

        bfs_graph.add_edge(3, 5)
        bfs_graph.add_edge(5, 4)

        bfs_graph.add_edge(1, 10)
        bfs_graph.add_edge(10, 4)

        actual = betweenness_centrality.naive_betweenness_bfs(bfs_graph, 0, 4, {})
        expected = 3

        self.assertEqual(actual, expected)
