from unittest import TestCase

import networkx as nx

from election import nearest_candidate_id, Candidate, run_election


class TestElection(TestCase):
    def setUp(self) -> None:
        self.candidates = [
            Candidate(40, 0.3),
            Candidate(41, 0.4),
            Candidate(42, 0.5),
            Candidate(43, 0.6),
            Candidate(44, 0.7),
        ]

    def test_run_election(self):
        graph = nx.Graph()
        graph.add_node(1, peak_preference=0.4)  # 41
        graph.add_node(2, peak_preference=0.39)  # 41
        graph.add_node(3, peak_preference=0.5)  # 42
        graph.add_node(4, peak_preference=0.0)  # 40
        graph.add_node(5, peak_preference=1.0)  # 44
        graph.add_node(6, peak_preference=0.66)  # 44
        graph.add_node(7, peak_preference=0.52)  # 42

        expected = {
            40: 1,
            41: 2,
            42: 2,
            43: 0,
            44: 2
        }
        actual = run_election(graph, self.candidates)

        self.assertDictEqual(expected, actual)

    def test_nearest_candidate_exact(self):
        voting_position = 0.5
        expected = 42
        actual = nearest_candidate_id(voting_position, self.candidates)

        self.assertEqual(expected, actual)

    def test_nearest_candidate_left_preference_in_ties(self):
        voting_position = 0.45
        expected = 41
        actual = nearest_candidate_id(voting_position, self.candidates)

        self.assertEqual(expected, actual)
