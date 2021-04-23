import networkx as nx
import math
import itertools as it
from priorityq import PriorityQueue
import random


# n = number of nodes
# m = number of edges


# Naive implementation of hierarchical clustering algorithm
def hierarchical(graph, seed):  # O(n^2*logn)
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

    return clusters


def hierarchical_optimized(graph, seed):
    random.seed(seed)
    node_to_cluster = {}
    cluster_to_nodes = {}

    i = 0
    for node in graph.nodes:
        node_to_cluster[node] = i
        cluster_to_nodes[i] = [node]
        i += 1

    done = False
    while not done:

        cluster = random.choice(list(cluster_to_nodes.keys()))
        edges = list(graph.edges(cluster_to_nodes[cluster]))
        if not len(edges) == 0:
            chosen_node = random.choice(edges)[1]
            chosen_cluster = node_to_cluster[chosen_node]

            if not chosen_cluster == cluster:
                cluster_to_nodes[cluster] = cluster_to_nodes[cluster] + cluster_to_nodes[chosen_cluster]
                for node in cluster_to_nodes[chosen_cluster]:
                    node_to_cluster[node] = cluster
                del(cluster_to_nodes[chosen_cluster])

        if len(cluster_to_nodes.keys()) == 4:
            done = True

    return cluster_to_nodes.values()
