import typing

import networkx as nx
from election import Candidate, run_election
from network_diffusion.fj_dynamics import fj_dynamics


def run_experiment(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate: int, number_of_seeds: int,
                   compute_seeds: typing.Callable) -> typing.Tuple[int, int]:
    """
    :param graph:                   The graph of the voters. Each node has the attribute peak_preference
                                    and stubbornness. The stubbornness value should 0.5 for each node.
    :param candidates:
    :param target_candidate:        The id of the candidate that has to be publicized.
    :param number_of_seeds:
    :param compute_seeds:
    :return:                        A tuple in which the first element is the score obtained by the target candidate
                                    in the truthful election and the second element is the score obtained in the
                                    manipulated one.
    """

    # run truthful election
    results = run_election(graph, candidates)
    truthful_score = results[target_candidate]

    # seeds is a dict {node: preference}
    seeds = compute_seeds(graph, candidates, target_candidate, number_of_seeds)

    # update graph with seeds
    for node, preference in seeds.items():
        graph.nodes[node]["peak_preference"] = preference
        graph.nodes[node]["stubbornness"] = 1

    manipulated_preferences = fj_dynamics(graph)

    # update graph with manipulated preferences
    for node, preference in manipulated_preferences.items():
        graph.nodes[node]["peak_preference"] = preference

    # run manipulated election
    results = run_election(graph, candidates)
    manipulated_score = results[target_candidate]

    return truthful_score, manipulated_score
