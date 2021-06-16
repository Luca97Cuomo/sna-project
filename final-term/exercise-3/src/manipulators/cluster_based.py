import math
import sys

import networkx as nx
import typing

from election import Candidate
from exercise_1.clustering_algorithms import k_means
from manipulators.manipulators import multi_level_greedy_manipulator, Manipulator, Seeds
from shapley_centrality import shapley_degree

Clusters = typing.List[typing.List[int]]

AVERAGE_SEEDS_PER_CLUSTER = 5


def k_means_with_shapley_degree(graph: nx.Graph, k: int) -> Clusters:
    return k_means(graph, centrality_measure=shapley_degree, k=k)


def _get_preferred_number_of_clusters(graph: nx.Graph, number_of_seeds: int) -> int:
    # heuristics: we want 5 seeds on average for each cluster (we don't know the final size of the cluster, but we would
    # actually like to have more seeds if the cluster is big)
    return number_of_seeds // AVERAGE_SEEDS_PER_CLUSTER


def cluster_based_manipulator(graph: nx.Graph, candidates: typing.List[Candidate], target_candidate_id: int,
                              number_of_seeds: int, seed: int, number_of_jobs: int = 1,
                              cluster_maker: typing.Callable[[nx.Graph, int], Clusters] = k_means_with_shapley_degree,
                              manipulator: Manipulator = multi_level_greedy_manipulator) -> Seeds:
    """
    Divides the graph in clusters - then, it chooses the seeds for each cluster based on some rule (greedily or
    according to a centrality measure).
    """
    n_clusters = _get_preferred_number_of_clusters(graph, number_of_seeds)
    clusters = cluster_maker(graph, n_clusters)

    n_nodes = len(graph.nodes)
    avg_nodes_per_cluster = n_nodes // n_clusters

    seeds: Seeds = {}
    used_seeds = 0

    # choosing the seeds for each cluster
    sorted_clusters = sorted(clusters, key=lambda nodes: len(nodes), reverse=True)
    n_seeds_for_cluster = []
    for cluster in sorted_clusters:
        n_cluster_nodes = len(cluster)
        # so that we have more seeds if the cluster is larger than expected and viceversa
        n_chosen_seeds = math.floor((n_cluster_nodes / avg_nodes_per_cluster) * AVERAGE_SEEDS_PER_CLUSTER)

        remaining_seeds = number_of_seeds - used_seeds
        n_chosen_seeds = n_chosen_seeds if n_chosen_seeds <= remaining_seeds else remaining_seeds
        used_seeds += n_chosen_seeds
        n_seeds_for_cluster.append(n_chosen_seeds)

    if used_seeds < number_of_seeds:
        print(f"used seeds: {used_seeds}, total: {number_of_seeds}", file=sys.stderr)
        n_seeds_for_cluster[0] += number_of_seeds - used_seeds

    print(f"n_seeds for each cluster: {n_seeds_for_cluster}", file=sys.stderr)

    for i, cluster in enumerate(sorted_clusters):
        if n_seeds_for_cluster[i] > 0:
            cluster_subgraph = graph.subgraph(cluster)
            cluster_seeds = manipulator(cluster_subgraph, candidates, target_candidate_id, n_seeds_for_cluster[i], seed,
                                        number_of_jobs)
            seeds.update(cluster_seeds)

    return seeds
