from unittest import TestCase

import networkx as nx

from final_term.exercise_1.src.shapley_centrality import naive_shapley_centrality, shapley_degree, shapley_threshold, shapley_closeness, \
    characteristic_functions
from final_term.exercise_1.src.shapley_centrality.naive import shapley_value_combinations, shapley_value_permutations


class Test(TestCase):
    def setUp(self) -> None:
        self.seed = 42

        self.graph = nx.Graph()
        self.graph.add_edge(1, 2)
        self.graph.add_edge(1, 4)
        self.graph.add_edge(2, 4)
        self.graph.add_edge(2, 5)
        self.graph.add_edge(3, 5)
        self.graph.add_edge(4, 5)

        # with n = 10,000 and p = 0,01 we have 500,000 edges
        self.big_graph = nx.fast_gnp_random_graph(10000, 0.01, seed=self.seed)

    def slide_value(self, graph, coalition):
        if len(coalition) == 0:
            return 0

        if coalition == {1} or coalition == {2} or coalition == {3} or coalition == {2, 3}:
            return 0
        if coalition == {1, 2} or coalition == {1, 2, 3}:
            return 5
        if coalition == {1, 3}:
            return 2

        raise Exception("Combination non found")

    def test_naive_shapley_centrality_combinations(self):
        expected = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        actual = naive_shapley_centrality(self.graph, lambda graph, coalition: 42, shapley_value_combinations)
        self.assertDictAlmostEqual(expected, actual)

        expected = {1: 17/6, 2: 11/6, 3: 2/6}
        slide_graph = nx.Graph()
        slide_graph.add_edge(1, 2)
        slide_graph.add_edge(1, 3)

        # Tested shapley value with values in slide 64 of game theory centrality

        actual = naive_shapley_centrality(slide_graph, self.slide_value, shapley_value_combinations)
        self.assertDictAlmostEqual(expected, actual)

    def test_naive_shapley_centrality_permutations(self):
        expected = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        actual = naive_shapley_centrality(self.graph, lambda graph, coalition: 42, shapley_value_permutations)
        self.assertDictAlmostEqual(expected, actual)

        expected = {1: 17/6, 2: 11/6, 3: 2/6}
        slide_graph = nx.Graph()
        slide_graph.add_edge(1, 2)
        slide_graph.add_edge(1, 3)

        # Tested shapley value with values in slide 64 of game theory centrality

        actual = naive_shapley_centrality(slide_graph, self.slide_value, shapley_value_permutations)
        self.assertDictAlmostEqual(expected, actual)

    def test_shapley_degree(self):
        expected = {1: 10/12, 2: 13/12, 3: 9/12, 4: 13/12, 5: 15/12}
        actual = shapley_degree(self.graph)
        self.assertDictAlmostEqual(expected, actual)

    def test_shapley_degree_equal_to_naive(self):
        naive = naive_shapley_centrality(self.graph, characteristic_functions.degree, shapley_value_combinations)
        optimized = shapley_degree(self.graph)
        self.assertDictAlmostEqual(naive, optimized)

    def test_shapley_threshold(self):
        expected = {1: 10/12, 2: 13/12, 3: 9/12, 4: 13/12, 5: 15/12}
        actual = shapley_threshold(self.graph, 1)
        self.assertDictAlmostEqual(expected, actual)

        actual = shapley_threshold(self.graph, 2)
        self.assertDictNotAlmostEqual(expected, actual)

    def test_shapley_threshold_equal_to_naive(self):
        threshold = 2
        naive = naive_shapley_centrality(self.graph, lambda graph, coalition:
                                         characteristic_functions.threshold(graph, coalition, threshold),
                                         shapley_value_combinations)
        optimized = shapley_threshold(self.graph, threshold)
        self.assertDictAlmostEqual(naive, optimized)

    def test_shapley_closeness_equal_to_naive(self):
        naive = naive_shapley_centrality(self.graph, characteristic_functions.closeness, shapley_value_combinations)
        optimized = shapley_closeness(self.graph)
        self.assertDictAlmostEqual(naive, optimized)

    def test_shapley_degree_time(self):
        # time 2.268s (LordCatello)

        shapley_degree(self.big_graph)
        self.assertTrue(True)

    def test_shapley_threshold_time(self):
        # time 2.131s (LordCatello)

        shapley_threshold(self.big_graph, 5)
        self.assertTrue(True)

    def test_shapley_closeness_time(self):
        # time 490.042s (LordCatello)

        shapley_closeness(self.big_graph)
        self.assertTrue(True)

    def assertDictAlmostEqual(self, expected, actual):
        for key, value in expected.items():
            self.assertAlmostEqual(value, actual[key])

    def assertDictNotAlmostEqual(self, expected, actual):
        for key, value in expected.items():
            self.assertNotAlmostEqual(value, actual[key])
