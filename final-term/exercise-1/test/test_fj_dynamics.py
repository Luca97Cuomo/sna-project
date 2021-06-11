import random
from unittest import TestCase
import networkx as nx

from network_diffusion.fj_dynamics import is_dynamics_converged, fj_dynamics

SEED = 42

class TestFjDynamics(TestCase):
    def test_is_dynamics_converged_true(self):
        prev_opinions = {3: 6.7865342189, 5: 99.6374829}
        current_opinions = {3: 6.7865312189, 5: 99.6374839}
        actual = is_dynamics_converged(prev_opinions, current_opinions, 5)
        self.assertTrue(actual)

    def test_is_dynamics_converged_false(self):
        prev_opinions = {3: 6.7865342189, 5: 99.6374829}
        current_opinions = {3: 6.7865392189, 5: 99.6374839}
        actual = is_dynamics_converged(prev_opinions, current_opinions, 5)
        self.assertFalse(actual)

    def test_fj_dynamics_stubborn(self):
        # create graph
        graph = nx.Graph()
        graph.add_edge(1, 2)
        graph.add_edge(1, 4)
        graph.add_edge(2, 4)
        graph.add_edge(2, 5)
        graph.add_edge(3, 5)
        graph.add_edge(4, 5)

        # init belief and stubbornness
        graph.nodes[1]["private_belief"] = 0.5
        graph.nodes[2]["private_belief"] = 1
        graph.nodes[3]["private_belief"] = 0.2
        graph.nodes[4]["private_belief"] = 0.7
        graph.nodes[5]["private_belief"] = 0.9

        graph.nodes[1]["stubbornness"] = 1
        graph.nodes[2]["stubbornness"] = 1
        graph.nodes[3]["stubbornness"] = 1
        graph.nodes[4]["stubbornness"] = 1
        graph.nodes[5]["stubbornness"] = 1

        opinions = fj_dynamics(graph)

        # assert
        for node in opinions.keys():
            opinion = opinions[node]
            private_belief = graph.nodes[node]["private_belief"]
            self.assertEqual(opinion, private_belief)

    def test_fj_dynamics_convergence(self):
        """
        It converges in 23 iterations and 4.231 s
        """
        # with n = 10,000 and p = 0,01 we have 500,000 edges
        graph = nx.fast_gnp_random_graph(10000, 0.01, seed=SEED)

        # print(len(graph.edges()))

        self._populate_dynamics_parameters(graph, seed=SEED)

        fj_dynamics(graph)
        self.assertTrue(True)

    def test_fj_dynamics_convergence_zero_stubbornness(self):
        """
        It converges in 9 iterations and 2.256 s
        """
        # with n = 10,000 and p = 0,01 we have 500,000 edges
        graph = nx.fast_gnp_random_graph(10000, 0.01, seed=SEED)

        # print(len(graph.edges()))

        self._populate_dynamics_parameters(graph, seed=SEED, stubbornness=0)

        fj_dynamics(graph)
        self.assertTrue(True)

    def _populate_dynamics_parameters(self, graph: nx.Graph, seed, private_belief=None, stubbornness=None):
        random.seed(seed)
        for node in graph.nodes():
            if private_belief is not None:
                graph.nodes[node]["private_belief"] = private_belief
            else:
                graph.nodes[node]["private_belief"] = random.random()

            if stubbornness is not None:
                graph.nodes[node]["stubbornness"] = stubbornness
            else:
                graph.nodes[node]["stubbornness"] = random.random()
