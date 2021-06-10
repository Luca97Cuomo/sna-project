from unittest import TestCase

import networkx as nx

import characteristic_functions
from shapley_centrality import naive_shapley_centrality, shapley_degree


class Test(TestCase):
    def setUp(self) -> None:
        self.graph = nx.Graph()
        self.graph.add_edge(1, 2)
        self.graph.add_edge(1, 4)
        self.graph.add_edge(2, 4)
        self.graph.add_edge(2, 5)
        self.graph.add_edge(3, 5)
        self.graph.add_edge(4, 5)

    def test_naive_shapley_centrality(self):
        expected = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        actual = naive_shapley_centrality(self.graph, lambda graph, coalition: 42)
        self.assertDictAlmostEqual(expected, actual)

        # slide errate?
        # expected = {1: 10/12, 2: 13/12, 3: 9/12, 4: 13/12, 5: 15/12}
        # actual = naive_shapley_centrality(self.graph, characteristic_functions.degree)
        # self.assertDictAlmostEqual(expected, actual)

        expected = {1: 17/6, 2: 11/6, 3: 2/6}
        slide_graph = nx.Graph()
        slide_graph.add_edge(1, 2)
        slide_graph.add_edge(1, 3)
        def slide_value(graph, coalition):
            if coalition == {1} or coalition == {2} or coalition == {3} or coalition == {2, 3}:
                return 0
            if coalition == {1, 2} or coalition == {1, 2, 3}:
                return 5
            return 2
        actual = naive_shapley_centrality(slide_graph, slide_value)
        self.assertDictAlmostEqual(expected, actual)

    def test_shapley_degree(self):
        expected = {1: 10/12, 2: 13/12, 3: 9/12, 4: 13/12, 5: 15/12}
        actual = shapley_degree(self.graph)
        self.assertDictAlmostEqual(expected, actual)

    def assertDictAlmostEqual(self, expected, actual):
        for key, value in expected.items():
            self.assertAlmostEqual(value, actual[key])
