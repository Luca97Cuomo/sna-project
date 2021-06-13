import math
from collections import defaultdict
from joblib import Parallel, delayed
import networkx as nx
import utils
from tqdm import tqdm
import numpy as np
import logging
import logging_configuration

logger = logging.getLogger()

'''

The degree centrality measure has been implemented only with the naive algorithm
because the running time is already feasible. In fact it takes less one seconds as you can see from the results below.

```
2021-06-12 18:20:58,730 __evaluate      INFO     Evaluating degree_centrality algorithm
2021-06-12 18:20:58,756 __evaluate      INFO     The centrality algorithm: degree_centrality took 0.025079 seconds
```

'''


def degree_centrality(graph):
    node_to_degree = {}

    for node in graph.nodes():
        node_to_degree[node] = graph.degree(node)

    return node_to_degree


'''

For the closeness centrality measure a naive and a parallel version has been implemented.
The naive implementation, for each node given as input:

1. Compute the distances from the node to all the others nodes of the graph, using the BFS
2. Evaluates the closeness of the node as the inverse of the average node distance from the others
                                
                                (n-1)/sum(distances_from_graph_nodes)

```
2021-06-12 18:20:58,828 __evaluate      INFO     Evaluating closeness_centrality algorithm
2021-06-12 18:45:45,490 __evaluate      INFO     The centrality algorithm: closeness_centrality took 1486.66 seconds
```

Even if the naive implementation took more or less half of an hour, a parallel version has been implemented
that splits the nodes of the graph into different chunks (one for each job) and compute the closeness on different jobs.

```
2021-06-12 18:20:58,828 __evaluate      INFO     Evaluating closeness_centrality algorithm
2021-06-12 18:45:45,490 __evaluate      INFO     The centrality algorithm: closeness_centrality took 1486.66 seconds
```

The results of the naive, the parallel implementation and the networkx implementation, have been compared in time and
in precision. The time of the naive and the networkx implementation is almost the same, while the parallel
implementation performs better.

The closeness computed by each implementation is exactly the same.

'''


def closeness_centrality(graph, samples):
    node_to_closeness = {}
    for node in samples:
        # Dictionary mapping each node to its distance from the current node
        distances_from_node = utils.bfs(graph, node)

        node_to_closeness[node] = (len(graph.nodes) - 1) / (sum(distances_from_node.values()))

    return node_to_closeness


def parallel_closeness_centrality(graph, n_jobs):
    node_to_closeness = {i: 0 for i in graph.nodes()}

    with Parallel(n_jobs=n_jobs) as parallel:
        partial_results = parallel(delayed(closeness_centrality)(graph, chunk) for chunk in
                                   utils.chunks(graph.nodes(), math.ceil(len(graph.nodes()) / n_jobs)))

        logger.info(f"Aggregating results from parallel jobs")

        for job_node_to_closeness in partial_results:
            for node in job_node_to_closeness.keys():
                node_to_closeness[node] += job_node_to_closeness[node]

    return node_to_closeness


'''



'''


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
                logger.info(f"The algorithm has reached convergence at iteration {i}.")
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
                logger.info(f"The algorithm has reached convergence at iteration {i}.")
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


def parallel_betweenness_centrality(graph, n_jobs):
    edge_btw = {frozenset(e): 0 for e in graph.edges()}
    node_btw = {i: 0 for i in graph.nodes()}

    with Parallel(n_jobs=n_jobs) as parallel:
        partial_results = parallel(delayed(betweenness_centrality)(graph, chunk) for chunk in
                                   utils.chunks(graph.nodes(), math.ceil(len(graph.nodes()) / n_jobs)))

        logger.info(f"Aggregating results from parallel jobs")

        for job_result in partial_results:
            for key in job_result[0].keys():
                # this is partial edge_btw
                edge_btw[key] += job_result[0][key]
            for key in job_result[1].keys():
                # this is partial node_btw
                node_btw[key] += job_result[1][key]

    return edge_btw, node_btw


def betweenness_centrality(graph, sample):
    edge_btw = {frozenset(e): 0 for e in graph.edges()}
    node_btw = {i: 0 for i in graph.nodes()}

    for s in sample:
        # Compute the number of shortest paths from s to every other node
        tree = []  # it lists the nodes in the order in which they are visited
        spnum = {i: 0 for i in graph.nodes()}  # it saves the number of shortest paths from s to i
        parents = {i: [] for i in graph.nodes()}  # it saves the parents of i in each of the shortest paths from s to i
        # distance = {i: -1 for i in graph.nodes()}  # the number of shortest paths starting from s that use the edge e
        # eflow = {frozenset(e): 0 for e in graph.edges()}  # the number of shortest paths starting from s that use the edge e
        distance = {}
        vflow = {i: 1 for i in
                 graph.nodes()}  # the number of shortest paths starting from s that use the vertex i. It is initialized to 1 because the shortest path from s to i is assumed to uses that vertex once.

        # BFS
        queue = [s]
        spnum[s] = 1
        distance[s] = 0
        while queue != []:
            c = queue.pop(0)
            tree.append(c)
            for i in graph[c]:
                if i not in distance:  # if vertex i has not been visited
                    queue.append(i)
                    distance[i] = distance[c] + 1
                if distance[i] == distance[c] + 1:  # if we have just found another shortest path from s to i
                    spnum[i] += spnum[c]
                    parents[i].append(c)

        # BOTTOM-UP PHASE
        while tree != []:
            c = tree.pop()
            for i in parents[c]:
                eflow = vflow[c] * (spnum[i] / spnum[
                    c])  # the number of shortest paths using vertex c is split among the edges towards its parents proportionally to the number of shortest paths that the parents contributes
                vflow[
                    i] += eflow  # each shortest path that use an edge (i,c) where i is closest to s than c must use also vertex i
                edge_btw[frozenset({c,
                                    i})] += eflow  # betweenness of an edge is the sum over all s of the number of shortest paths from s to other nodes using that edge
            if c != s:
                node_btw[c] += vflow[
                    c]  # betweenness of a vertex is the sum over all s of the number of shortest paths from s to other nodes using that vertex

    return edge_btw, node_btw
