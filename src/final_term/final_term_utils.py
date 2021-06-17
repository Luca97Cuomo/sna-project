import random

import networkx as nx

SEED = 42


def populate_dynamics_parameters(graph: nx.Graph, seed, private_belief=None, stubbornness=None):
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