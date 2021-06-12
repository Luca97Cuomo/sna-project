from unittest import TestCase
import networkx as nx
from tqdm import tqdm

import utils
from final_term_utils import populate_dynamics_parameters
from network_diffusion.fj_dynamics import is_dynamics_converged, fj_dynamics

SEED = 42
FACEBOOK_PATH_TO_NODES = "../../../mid-term/exercise-1-2/facebook_large/musae_facebook_target.csv"
FACEBOOK_PATH_TO_EDGES = "../../../mid-term/exercise-1-2/facebook_large/musae_facebook_edges.csv"


class TestFjDynamics(TestCase):
    def test_is_dynamics_converged_true(self):
        prev_opinions = {3: 6.7865342189, 5: 99.6374829}
        current_opinions = {3: 6.7865312189, 5: 99.6374839}
        actual = is_dynamics_converged(prev_opinions, current_opinions, 5)
        self.assertTrue(actual)

    def test_is_dynamics_converged_false(self):
        prev_opinions = {3: 6.7865242189, 5: 99.6374829}
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

    def test_fj_dynamics_convergence_big_graph(self):
        """
        It converges in 23 iterations and 4.231 s
        It took 7.420 s without overclocking cpu (with cpu at 2.4 Ghz max)

        It converges in 22 iterations and 5.818 (2.5 Ghz) s with truncate instead of round
        """
        # with n = 10,000 and p = 0,01 we have 500,000 edges
        graph = nx.fast_gnp_random_graph(10000, 0.01, seed=SEED)

        # print(len(graph.edges()))

        populate_dynamics_parameters(graph, seed=SEED)

        fj_dynamics(graph)
        self.assertTrue(True)

    def test_fj_dynamics_convergence_zero_stubbornness(self):
        """
        It converges in 9 iterations and 2.256 s
        """
        # with n = 10,000 and p = 0,01 we have 500,000 edges
        graph = nx.fast_gnp_random_graph(10000, 0.01, seed=SEED)

        # print(len(graph.edges()))

        populate_dynamics_parameters(graph, seed=SEED, stubbornness=0)

        fj_dynamics(graph)
        self.assertTrue(True)

    def test_fj_dynamics_convergence_facebook_graph_zero_stubbornness(self):
        """

        """

        graph, _ = utils.load_graph_and_clusters(FACEBOOK_PATH_TO_NODES, FACEBOOK_PATH_TO_EDGES)

        print(len(graph.edges()))

        populate_dynamics_parameters(graph, seed=SEED, stubbornness=0)

        fj_dynamics(graph)
        self.assertTrue(True)


    def test_fj_dynamics_convergence_facebook_graph_random(self):
        """
        It converges in 50 iterations and 6.164 s
        """

        graph, _ = utils.load_graph_and_clusters(FACEBOOK_PATH_TO_NODES, FACEBOOK_PATH_TO_EDGES)

        print(len(graph.edges()))

        populate_dynamics_parameters(graph, seed=SEED)

        fj_dynamics(graph)
        self.assertTrue(True)

    def test_fj_dynamics_convergence_facebook_graph_half_stubborness(self):
        """
        It converges in 16 iterations and 2.4 s
        """

        graph, _ = utils.load_graph_and_clusters(FACEBOOK_PATH_TO_NODES, FACEBOOK_PATH_TO_EDGES)

        print(len(graph.edges()))

        populate_dynamics_parameters(graph, seed=SEED, stubbornness=1 / 2)

        fj_dynamics(graph)
        self.assertTrue(True)

    def test_fj_dynamics_convergence_multiple_graphs(self):
        """
        It converges in max 25 iterations
        """
        iterations = 100
        with tqdm(total=iterations) as pbar:
            for i in range(iterations):
                graph = nx.fast_gnp_random_graph(10000, 0.01, seed=i)

                populate_dynamics_parameters(graph, seed=i)

                fj_dynamics(graph)
                self.assertTrue(True)

                pbar.update(1)
