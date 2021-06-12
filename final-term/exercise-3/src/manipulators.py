import random
import typing

import networkx as nx

from election import Candidate


def bogo_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                     number_of_seeds: int, seed: int = 42) -> typing.Dict[int, float]:

    random.seed(seed)

    seeds = {}
    nodes = list(graph.nodes())
    chosen_nodes = random.sample(nodes, number_of_seeds)

    target_candidate = get_candidate_by_id(candidates, target_candidate_id)
    if target_candidate is None:
        raise Exception("The target candidate is None")

    for node in chosen_nodes:
        seeds[node] = target_candidate.position

    return seeds


def get_candidate_by_id(candidates: typing.List[Candidate], candidate_id: int) -> typing.Optional[Candidate]:
    for i in range(len(candidates)):
        if candidates[i].id == candidate_id:
            return candidates[i]

    return None
