from unittest import TestCase
import networkx as nx
import sys

sys.path.append("../")
import utils

from exercise_2 import centrality_measures


class TestCentralityMeasures(TestCase):
    def setUp(self) -> None:
        self.graph = utils.build_random_graph(100, 0.30, seed=42)

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
