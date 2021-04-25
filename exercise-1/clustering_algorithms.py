import networkx as nx
import math
import itertools as it
from priorityq import PriorityQueue
import random
from utils import rand_index, CENTRALITY_MEASURES


# n = number of nodes
# m = number of edges

# Naive implementation of hierarchical clustering algorithm
def hierarchical(graph, seed=42):  # O(n^2*logn)
    # Create a priority queue with each pair of nodes indexed by distance
    pq = PriorityQueue()
    for u in graph.nodes():  # O(n^2*logn) in the worst case.
        for v in graph.nodes():
            if u != v:
                if (u, v) in graph.edges() or (v, u) in graph.edges():  # O(1)
                    pq.add(frozenset([frozenset([u]), frozenset([v])]), 0)
                else:
                    pq.add(frozenset([frozenset([u]), frozenset([v])]), 1)

    # Start with a cluster for each node
    clusters = set(frozenset([u]) for u in graph.nodes())  # O(n)

    done = False
    while not done:  # O(input*(n*logn)) # The worst case, is input=n, and there will be a single cluster containing all the nodes
        # Merge closest clusters
        s = list(pq.pop())  # O(logn)
        clusters.remove(s[0])  # O(1)
        clusters.remove(s[1])  # O(1)

        # Update the distance of other clusters from the merged cluster
        for w in clusters:  # O(n*logn)
            e1 = pq.remove(frozenset([s[0], w]))
            e2 = pq.remove(frozenset([s[1], w]))
            if e1 == 0 or e2 == 0:
                pq.add(frozenset([s[0] | s[1], w]), 0)
            else:
                pq.add(frozenset([s[0] | s[1], w]), 1)

        clusters.add(s[0] | s[1])  # O(1)

        if len(clusters) == 4:
            done = True

    return list(clusters)


def hierarchical_optimized(graph, seed=42, desired_clusters=4):
    random.seed(seed)
    node_to_cluster = {}
    cluster_to_nodes = {}

    i = 0
    for node in graph.nodes:  # O(n)
        node_to_cluster[node] = i
        cluster_to_nodes[i] = [node]
        i += 1

    while len(cluster_to_nodes.keys()) != desired_clusters:
        cluster = random.choice(list(cluster_to_nodes.keys()))
        edges = list(graph.edges(cluster_to_nodes[cluster]))  # O(n*grado(n))
        if not len(edges) == 0:
            neighbor_clusters = set()
            for edge in edges:
                if node_to_cluster[edge[0]] == cluster and node_to_cluster[edge[1]] == cluster:
                    continue
                if node_to_cluster[edge[0]] == cluster:
                    chosen_node = edge[1]
                else:
                    chosen_node = edge[0]
                neighbor_clusters.add(node_to_cluster[chosen_node])
            chosen_cluster = random.choice(list(neighbor_clusters))
            if not chosen_cluster == cluster:
                cluster_to_nodes[cluster] = cluster_to_nodes[cluster] + cluster_to_nodes[chosen_cluster]
                for node in cluster_to_nodes[chosen_cluster]:  # O(n)
                    node_to_cluster[node] = cluster
                del (cluster_to_nodes[chosen_cluster])

    return list(cluster_to_nodes.values())


def k_means_one_iteration(graph, seed=42, k=4, centers=None):
    current_nodes = []  # 0
    next_nodes = []

    # mi restituisce un vicino di root non ancora in un cluster
    def non_clustered_neighbors(graph, node):
        neighbors = graph.neighbors(node)
        for neighbor in neighbors:
            if neighbor not in node_to_cluster:
                yield neighbor

    def add_node_to_cluster(node, cluster):
        node_to_cluster[node] = cluster
        cluster_to_nodes.setdefault(cluster, []).append(node)

    node_to_cluster = {}
    cluster_to_nodes = {}
    if centers is None:
        random.seed(seed)
        nodes = list(graph.nodes())
        for i in range(k):
            center = random.choice(nodes)
            add_node_to_cluster(center, i)
            current_nodes.append(center)
            nodes.remove(center)
    else:
        assert len(centers) == k
        current_nodes = centers
        for i in range(k):
            add_node_to_cluster(centers[i], i)

    while len(current_nodes) != 0:
        while len(current_nodes) != 0:
            visited_nodes = set()
            for node in current_nodes:
                try:
                    neighbor = next(non_clustered_neighbors(graph, node))
                    add_node_to_cluster(neighbor, node_to_cluster[node])
                    next_nodes.append(neighbor)
                except StopIteration:
                    visited_nodes.add(node)
            current_nodes = list(filter(lambda node: node not in visited_nodes, current_nodes))

        current_nodes = next_nodes
        next_nodes = []

    return list(cluster_to_nodes.values())


def k_means(graph, centrality_measure=None, seed=42, k=4, equality_threshold=1e-3, max_iterations=1000, centers=None,
            verbose=False):
    # rand-index come condizione di convergenza

    # altra possibilità: prendere il più centrale tra i nodi del cluster, e poi alla prossima iterazione
    # verificare se la misura di centralità è cambiata di almeno tot (problema: due centri diversi possono avere
    # centralità simile ma generare cluster differenti).

    last_clustering = [[] for _ in range(k)]
    last_similarity = 0
    convergence = False
    iterations = 0
    centrality_measure_function = CENTRALITY_MEASURES.get(centrality_measure, None)
    if centrality_measure_function is not None and centers is None:
        centers = []
        centrality_dict = centrality_measure_function(graph)
        nodes = list(centrality_dict.keys())
        values = list(centrality_dict.values())
        for i in range(k):
            max_value = max(values)
            values.remove(max_value)
            center = random.choice([node for node in nodes if centrality_dict[node] == max_value])
            centers.append(nodes.pop(center))
    if centrality_measure_function is None and centers is not None:
        random.seed(seed)

    while not convergence and iterations < max_iterations:
        clusters = k_means_one_iteration(graph, seed, k, centers)  # note: the seed is only used once when centers=None
        if iterations > 0:
            similarity = rand_index(graph, clusters, last_clustering)
            if verbose:
                print(f"Difference between two iteration similarity: {abs(last_similarity - similarity)}")
            if abs(last_similarity - similarity) <= equality_threshold:
                convergence = True
            last_similarity = similarity

        last_clustering = clusters
        iterations += 1

        if not convergence:
            centers = []
            if centrality_measure_function is None:
                for cluster in clusters:
                    centers.append(random.choice(cluster))
            else:
                for cluster in clusters:
                    centrality_dict = centrality_measure_function(graph.subgraph(cluster))
                    values = list(centrality_dict.values())
                    max_value = max(values)
                    center = random.choice([node for node in cluster if centrality_dict[node] == max_value])
                    centers.append(center)

    return last_clustering
