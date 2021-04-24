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
    for node in graph.nodes:  # O(n)
        node_to_cluster[node] = i
        cluster_to_nodes[i] = [node]
        i += 1

    done = False
    while not done:
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

        if len(cluster_to_nodes.keys()) == 4:
            done = True

    return cluster_to_nodes.values()


def k_means(graph, k):
    n = graph.number_of_nodes()
    prec_list = []  # 0
    curr_list = []

    # mi restituisce un vicino di root non ancora in un cluster
    def next_neighbor(graph, root):
        neighbors = graph.neighbors(root)
        for neighbor in neighbors:
            if nx.get_node_attributes(graph, "cluster")[neighbor]:  # vicino.label = None
                yield neighbor

        return None

    u = random.choice(list(G.nodes()))
    cluster0 = {u}
    for i in range(k - 1):
        v = random.choice(list(nx.non_neighbors(G, u)))

    cluster1 = {v}
    node_to_cluster = {}
    added = 2

    while len(prec_list) != 0:  # se il grafo è disconnesso esplode
        while len(prec_list) != 0:

            for root in prec_list:  # non va bene perchè esplode, bisogna salvare i nodi da eliminare e elimarli fuori dal for.
                neighbor = next_neighbor(root)
                if neighbor is not None:
                    node_to_cluster[root] = node_to_cluster[root] + neighbor
                    curr_list.append(neighbor)
                else:
                    prec_list.pop(root)

        prec_list = curr_list
