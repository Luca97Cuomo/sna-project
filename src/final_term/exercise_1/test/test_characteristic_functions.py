from unittest import TestCase
import networkx as nx
from final_term.exercise_1.src.shapley_centrality import characteristic_functions


class TestCharacteristicFunctions(TestCase):
    def setUp(self) -> None:
        self.graph = nx.Graph()
        self.graph.add_edge(1, 2)
        self.graph.add_edge(1, 4)
        self.graph.add_edge(2, 4)
        self.graph.add_edge(2, 5)
        self.graph.add_edge(3, 5)
        self.graph.add_edge(4, 5)

        self.coalitions = [
            {1},
            {3},
            {2, 4},
            {2, 4, 5}
        ]

    def test_degree(self):
        expected_values = [3, 2, 4, 5]
        for expected, coalition in zip(expected_values, self.coalitions):
            actual = characteristic_functions.degree(self.graph, coalition)
            self.assertEqual(expected, actual)

    def test_threshold(self):
        threshold = 2
        expected_values = [1, 1, 4, 4]
        for expected, coalition in zip(expected_values, self.coalitions):
            actual = characteristic_functions.threshold(self.graph, coalition, threshold)
            self.assertEqual(expected, actual)

    def test_closeness(self):
        expected_values = [7, 8, 4, 2]
        for expected, coalition in zip(expected_values, self.coalitions):
            actual = characteristic_functions.closeness(self.graph, coalition, lambda distance: distance)
            self.assertEqual(expected, actual)
