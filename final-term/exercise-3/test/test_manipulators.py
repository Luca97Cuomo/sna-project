from unittest import TestCase

import networkx as nx

from election import Candidate
from manipulators import bogo_manipulator, get_candidate_by_id

SEED = 42


class TestManipulators(TestCase):
    def setUp(self) -> None:
        self.candidates = [
            Candidate(40, 0.3),
            Candidate(41, 0.4),
            Candidate(42, 0.5),
            Candidate(43, 0.6),
            Candidate(44, 0.7),
        ]

        self.graph = nx.Graph()
        self.graph.add_node(1, private_belief=0.4)
        self.graph.add_node(2, private_belief=0.39)
        self.graph.add_node(3, private_belief=0.5)
        self.graph.add_node(4, private_belief=0.0)
        self.graph.add_node(5, private_belief=1.0)
        self.graph.add_node(6, private_belief=0.66)
        self.graph.add_node(7, private_belief=0.52)

    def test_bogo_manipulator(self):
        target_candidate_id = 41

        seeds = bogo_manipulator(self.graph, self.candidates, target_candidate_id, 2, SEED)

        seeds_set = set()

        # search candidate position
        target_candidate = get_candidate_by_id(self.candidates, target_candidate_id)
        if target_candidate is None:
            raise Exception("The target candidate is None")

        for node, value in seeds.items():
            if node in seeds_set:
                self.fail()

            if value != target_candidate.position:
                self.fail()

            seeds_set.add(node)
