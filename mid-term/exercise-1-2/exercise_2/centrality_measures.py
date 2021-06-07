import math
import sys
from collections import defaultdict

from joblib import Parallel, delayed

import networkx as nx

sys.path.append("../")
import utils
from tqdm import tqdm
import numpy as np


def degree_centrality(graph):
    node_to_degree = {}
    with tqdm(total=len(graph)) as pbar:
        for node in graph.nodes():
            node_to_degree[node] = graph.degree(node)
            pbar.update(1)
        return node_to_degree


def closeness_centrality(graph):
    node_to_closeness = {}
    with tqdm(total=len(graph)) as pbar:
        for node in graph.nodes():
            distances_from_node = utils.bfs(graph, node)

            node_to_closeness[node] = (len(graph.nodes) - 1) / (sum(distances_from_node.values()))
            pbar.update(1)

    return node_to_closeness


def basic_page_rank(graph, max_iterations=100, delta=None):
    """
        Page rank using iterative method

        Preconditions:
            Undirected graph

        max_iterations: number of max iterations to perform
        delta: tolerance threshold for determining convergence. If given, the algorithm stops when the ranks are
            stabilized within the delta variable (for accounting for float precision).
    """

    def check_convergence(current_ranks, next_ranks, delta):
        if delta is None:
            return False
        for node, rank in current_ranks.items():
            next_rank = next_ranks[node]
            error = abs(next_rank - rank)
            if error > delta:
                return False
        return True

    # Initialize nodes weights
    current_node_to_rank = {}
    next_node_to_rank = {}
    for node in graph.nodes():
        current_node_to_rank[node] = np.float(1 / len(graph))
        next_node_to_rank[node] = np.float(0)

    with tqdm(total=max_iterations) as pbar:
        for i in range(max_iterations):  # add convergence check with tolerance
            for edge in graph.edges():
                first_endpoint = edge[0]
                second_endpoint = edge[1]

                # add alpha parameter
                # There are no dead ends and spider traps, the graph is undirected
                next_node_to_rank[first_endpoint] = next_node_to_rank[first_endpoint] + (
                        current_node_to_rank[second_endpoint] * np.float((1 / graph.degree(second_endpoint))))
                next_node_to_rank[second_endpoint] = next_node_to_rank[second_endpoint] + (
                        current_node_to_rank[first_endpoint] * np.float((1 / graph.degree(first_endpoint))))

            if check_convergence(current_node_to_rank, next_node_to_rank, delta):
                print(f"The algorithm has reached convergence at iteration {i}.")
                break

            for node, rank in next_node_to_rank.items():
                current_node_to_rank[node] = rank
                next_node_to_rank[node] = np.float(0)

            pbar.update(1)

    return current_node_to_rank


def algebraic_page_rank(graph, alpha=0.85, max_iterations=100, delta=None):
    def check_convergence(current_ranks, next_ranks, delta):
        if delta is None:
            return False

        for current_rank, next_rank in zip(current_ranks.flat, next_ranks.flat):
            error = abs(current_rank - next_rank)
            if error > delta:
                return False
        return True

    # Build the vector v and the transition matrix M
    current_v = 1 / len(graph) * np.ones((1, len(graph)))

    node_list = list(range(len(graph)))

    # Evaluate the transition matrix
    # Problematic O(N^2) memory 12GB
    matrix = nx.algorithms.link_analysis.pagerank_alg.google_matrix(graph, alpha=alpha, nodelist=node_list)
    with tqdm(total=max_iterations) as pbar:
        for i in range(max_iterations):
            next_v = np.dot(current_v, matrix)
            pbar.update(1)
            if check_convergence(current_v, next_v, delta):
                print(f"The algorithm has reached convergence at iteration {i}.")
                break
            current_v = next_v

    node_to_rank = {}
    # Luckily the array index i corresponds to the node index in the graph
    # This happens because all the nodes in the graph are numbered from 0 to len(graph)
    for i in range(len(graph)):
        node_to_rank[i] = current_v.item(i)

    return node_to_rank


def parallel_basic_page_rank(graph, max_iterations=100, jobs=4, delta=None):
    def check_convergence(current_ranks, next_ranks, delta):
        if delta is None:
            return False
        for node, rank in current_ranks.items():
            next_rank = next_ranks[node]
            error = abs(next_rank - rank)
            if error > delta:
                return False
        return True

    def chunked_page_rank_step(edges, current_node_to_rank):
        next_node_to_rank = {node: 0 for node in current_node_to_rank.keys()}
        for edge in edges:
            first_endpoint = edge[0]
            second_endpoint = edge[1]

            # add alpha parameter
            # There are no dead ends and spider traps, the graph is undirected
            next_node_to_rank[first_endpoint] = next_node_to_rank[first_endpoint] + (
                    current_node_to_rank[second_endpoint] * np.float((1 / graph.degree(second_endpoint))))
            next_node_to_rank[second_endpoint] = next_node_to_rank[second_endpoint] + (
                    current_node_to_rank[first_endpoint] * np.float((1 / graph.degree(first_endpoint))))
        return next_node_to_rank

    def aggregate_results(results):
        aggregated = defaultdict(int)
        for result in results:
            for node, rank in result.items():
                aggregated[node] += rank
        return aggregated

    # Initialize nodes weights
    current_node_to_rank = {}
    for node in graph.nodes():
        current_node_to_rank[node] = np.float(1 / len(graph))

    with tqdm(total=max_iterations) as pbar:
        with Parallel(n_jobs=jobs) as parallel:
            edges_chunks = []
            chunk_size = math.ceil(len(graph.edges) / jobs)
            for i in range(jobs):
                edges_chunks.append(list(graph.edges())[i * chunk_size: (i + 1) * chunk_size])

            for i in range(max_iterations):  # add convergence check with tolerance
                results = parallel(
                    delayed(chunked_page_rank_step)(edges_chunk, current_node_to_rank) for edges_chunk in edges_chunks)
                next_node_to_rank = aggregate_results(results)
                if check_convergence(current_node_to_rank, next_node_to_rank, delta):
                    print(f"The algorithm has reached convergence at iteration {i}.")
                    break

                current_node_to_rank = next_node_to_rank
                pbar.update(1)

    return current_node_to_rank


def hits(graph, max_iterations=100):
    node_to_authorities = {}
    node_to_hubs = {}

    for node in graph.nodes():
        node_to_authorities[node] = 1
        node_to_hubs[node] = 1

    with tqdm(total=max_iterations) as pbar:
        for i in range(max_iterations):
            for edge in graph.edges():
                first_endpoint = edge[0]
                second_endpoint = edge[1]

                node_to_authorities[first_endpoint] = node_to_authorities[first_endpoint] + node_to_hubs[
                    second_endpoint]
                node_to_authorities[second_endpoint] = node_to_authorities[second_endpoint] + node_to_hubs[
                    first_endpoint]

            for edge in graph.edges():
                first_endpoint = edge[0]
                second_endpoint = edge[1]

                node_to_hubs[first_endpoint] = node_to_hubs[first_endpoint] + node_to_authorities[second_endpoint]
                node_to_hubs[second_endpoint] = node_to_hubs[second_endpoint] + node_to_authorities[first_endpoint]

            sum_of_authorities = sum(node_to_authorities.values())
            sum_of_hubs = sum(node_to_hubs.values())

            for node in graph.nodes():
                node_to_authorities[node] = node_to_authorities[node] / sum_of_authorities
                node_to_hubs[node] = node_to_hubs[node] / sum_of_hubs

            pbar.update(1)

    return node_to_hubs, node_to_authorities


def parallel_hits(graph, max_iterations=100, jobs=4):
    def chunked_hits_step(edges, node_to_authorities, node_to_hubs):
        partial_node_to_authorities = {node: 0 for node in node_to_authorities.keys()}
        partial_node_to_hubs = {node: 0 for node in node_to_hubs.keys()}
        for edge in edges:
            first_endpoint = edge[0]
            second_endpoint = edge[1]

            partial_node_to_authorities[first_endpoint] = partial_node_to_authorities[first_endpoint] + node_to_hubs[
                second_endpoint]
            partial_node_to_authorities[second_endpoint] = partial_node_to_authorities[second_endpoint] + node_to_hubs[
                first_endpoint]

        for edge in edges:
            first_endpoint = edge[0]
            second_endpoint = edge[1]

            partial_node_to_hubs[first_endpoint] = partial_node_to_hubs[first_endpoint] + partial_node_to_authorities[
                second_endpoint]
            partial_node_to_hubs[second_endpoint] = partial_node_to_hubs[second_endpoint] + partial_node_to_authorities[
                first_endpoint]

        return partial_node_to_hubs, partial_node_to_authorities

    def aggregate_results(results, current_node_to_authorities, current_node_to_hubs):
        aggregated_authorities = defaultdict(float)
        aggregated_hubs = defaultdict(float)
        for result in results:
            results_hubs = result[0]
            results_authorities = result[1]
            for node, authority in results_authorities.items():
                aggregated_authorities[node] += authority

            for node, hub in results_hubs.items():
                aggregated_hubs[node] += hub

        for node, current_authority in current_node_to_authorities.items():
            aggregated_authorities[node] += current_authority

        for node, current_hub in current_node_to_hubs.items():
            aggregated_hubs[node] += current_hub

        sum_of_authorities = sum(aggregated_authorities.values())
        sum_of_hubs = sum(aggregated_hubs.values())

        for node in graph.nodes():
            aggregated_authorities[node] = aggregated_authorities[node] / sum_of_authorities
            aggregated_hubs[node] = aggregated_hubs[node] / sum_of_hubs

        return aggregated_authorities, aggregated_hubs

    node_to_authorities = {}
    node_to_hubs = {}

    for node in graph.nodes():
        node_to_authorities[node] = 1
        node_to_hubs[node] = 1

    with tqdm(total=max_iterations) as pbar:
        with Parallel(n_jobs=jobs) as parallel:
            edges_chunks = []
            chunk_size = math.ceil(len(graph.edges) / jobs)
            for i in range(jobs):
                edges_chunks.append(list(graph.edges())[i * chunk_size: (i + 1) * chunk_size])

            for i in range(max_iterations):
                results = parallel(
                    delayed(chunked_hits_step)(edges_chunk, node_to_authorities, node_to_hubs) for edges_chunk in
                    edges_chunks)

                node_to_authorities, node_to_hubs = aggregate_results(results,
                                                                      node_to_authorities,
                                                                      node_to_hubs)

                pbar.update(1)

    return node_to_hubs, node_to_authorities
