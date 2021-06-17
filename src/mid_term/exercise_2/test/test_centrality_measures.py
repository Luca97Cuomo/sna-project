from unittest import TestCase
import networkx as nx
from pytest import approx
import utils
from mid_term.exercise_2 import centrality_measures
import logging
from final_term.exercise_2 import logging_configuration

logging_configuration.set_logging()

logger = logging.getLogger()
PATH_TO_NODES = "../../../../facebook_large/musae_facebook_target.csv"
PATH_TO_EDGES = "../../../../facebook_large/musae_facebook_edges.csv"


class TestCentralityMeasures(TestCase):
    def setUp(self) -> None:
        self.graph = utils.build_random_graph(1000, 0.10, seed=42)
        self.facebook_graph, _ = utils.load_graph_and_clusters(PATH_TO_NODES, PATH_TO_EDGES)

        self.small_graph = nx.Graph()
        self.small_graph.add_edge(0, 1)
        self.small_graph.add_edge(0, 2)
        self.small_graph.add_edge(0, 3)

    def test_degree_centrality(self):
        expected = nx.degree_centrality(self.graph)
        results = centrality_measures.degree_centrality(self.graph)

        expected = sorted(expected.items(), key=lambda item: item[1])
        expected_nodes = []
        for entry in expected:
            expected_nodes.append(entry[0])

        node_btw = sorted(results.items(), key=lambda item: item[1])
        results_nodes = []
        for entry in node_btw:
            results_nodes.append(entry[0])

        self.assertListEqual(expected_nodes, results_nodes)

    def test_closeness_centrality(self):
        expected = nx.closeness_centrality(self.graph)
        results = centrality_measures.closeness_centrality(self.graph, self.graph.nodes())
        self.assertDictEqual(expected, results)

    def test_parallel_closeness_centrality(self):
        logger.debug('Evaluating nx closeness centrality')
        expected = nx.closeness_centrality(self.facebook_graph)
        logger.debug('Evaluating parallel closeness centrality')
        results = centrality_measures.parallel_closeness_centrality(self.facebook_graph, n_jobs=4)
        logger.debug('Assert if results are equals')
        self.assertDictEqual(expected, results)
        # la closeness di networkx restituisce gli stessi identici risultati della nostra closeness parallela e anche di quella normale.

    def test_parallel_betweenneess_centrality(self):
        expected = nx.betweenness_centrality(self.graph)

        edge_btw, node_btw = centrality_measures.parallel_betweenness_centrality(self.graph, 8)
        expected = sorted(expected.items(), key=lambda item: item[1])
        expected_nodes = []
        for entry in expected:
            expected_nodes.append(entry[0])

        node_btw = sorted(node_btw.items(), key=lambda item: item[1])
        results_nodes = []
        for entry in node_btw:
            results_nodes.append(entry[0])

        self.assertListEqual(expected_nodes, results_nodes)

    def test_equality_of_betweenness(self):
        graph = self.graph
        expected_edge_btw, expected_node_btw = centrality_measures.betweenness_centrality(graph, graph.nodes())
        number_of_concurrent_jobs = 8
        resuts_edge_btw, results_node_btw = centrality_measures.parallel_betweenness_centrality(graph,
                                                                                                number_of_concurrent_jobs)

        for node, btw in results_node_btw.items():
            self.assertEqual(approx(expected_node_btw[node], rel=0.01), results_node_btw[node])

        for edge, btw in resuts_edge_btw.items():
            self.assertEqual(approx(expected_edge_btw[edge], rel=0.01), resuts_edge_btw[edge])
        # Ritornano gli stessi risultati

    def test_algebraic_page_rank(self):
        expected = centrality_measures.basic_page_rank(self.facebook_graph, max_iterations=500, delta_rel=0.2)
        results = centrality_measures.algebraic_page_rank(self.facebook_graph, max_iterations=500, alpha=1,
                                                          delta_rel=0.2)
        for node in expected.keys():
            self.assertEqual(approx(expected[node], rel=0.5), results[node])

    def test_parallel_page_rank(self):
        results = centrality_measures.parallel_basic_page_rank(self.facebook_graph, max_iterations=200, delta_rel=0.2,
                                                               jobs=4)
        expected = centrality_measures.basic_page_rank(self.facebook_graph, max_iterations=200, delta_rel=1e-5)
        for node in results.keys():
            self.assertEqual(approx(expected[node], rel=0.5), results[node])

    def test_naive_edge_hits(self):
        graph = self.facebook_graph
        nx_hubs, nx_authorities = nx.algorithms.link_analysis.hits_alg.hits(graph, max_iter=100)

        hubs, authorities = centrality_measures.naive_edge_hits(graph, max_iterations=100)

        for node in graph.nodes():
            self.assertEqual(approx(nx_hubs[node], rel=1e-1), hubs[node])
            self.assertEqual(approx(nx_authorities[node], rel=1e-1), authorities[node])

    def test_parallel_edge_hits(self):
        graph = self.facebook_graph

        expected_hubs, expected_authorities = centrality_measures.naive_edge_hits(graph, max_iterations=100)
        result_hubs, result_authorities = centrality_measures.parallel_edge_hits(graph, max_iterations=100)

        for node in graph.nodes():
            self.assertEqual(approx(expected_hubs[node], rel=1e-3), result_hubs[node])
            self.assertEqual(approx(expected_authorities[node], rel=1e-3), result_authorities[node])

    def test_naive_hits(self):
        graph = self.facebook_graph
        nx_hubs, nx_authorities = nx.algorithms.link_analysis.hits_alg.hits(graph, max_iter=100)

        hubs, authorities = centrality_measures.naive_hits(graph, max_iterations=100)

        for node in graph.nodes():
            self.assertEqual(approx(nx_hubs[node], rel=1e-1), hubs[node])
            self.assertEqual(approx(nx_authorities[node], rel=1e-1), authorities[node])

    def test_parallel_naive_hits(self):
        graph = self.facebook_graph
        expected_hubs, expected_authorities = centrality_measures.naive_hits(graph, max_iterations=100)

        hubs, authorities = centrality_measures.parallel_naive_hits(graph, max_iterations=100)

        for node in graph.nodes():
            self.assertEqual(approx(expected_hubs[node], rel=1e-1), hubs[node])
            self.assertEqual(approx(expected_authorities[node], rel=1e-1), authorities[node])

    def test_node_names(self):
        facebook_graph, true_clusters = utils.load_graph_and_clusters(PATH_TO_NODES, PATH_TO_EDGES)

        for i in range(len(facebook_graph)):
            self.assertTrue(facebook_graph.has_node(i))
