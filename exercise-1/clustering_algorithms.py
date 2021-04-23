import networkx as nx
import math
import itertools as it
from priorityq import PriorityQueue


# n = number of nodes
# m = number of edges


# Naive implementation of hierarchical clustering algorithm
def hierarchical(G):  # O(n^2*logn)
    # Create a priority queue with each pair of nodes indexed by distance
    pq = PriorityQueue()
    for u in G.nodes():  # O(n^2*logn) in the worst case.
        for v in G.nodes():
            if u != v:
                if (u, v) in G.edges() or (v, u) in G.edges():  # O(1)
                    pq.add(frozenset([frozenset([u]), frozenset([v])]), 0)
                else:
                    pq.add(frozenset([frozenset([u]), frozenset([v])]), 1)

    # Start with a cluster for each node
    clusters = set(frozenset([u]) for u in G.nodes())  # O(n)

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



def hierarchical_2(graph):
    """
    Potremmo fare un algoritmo di questo tipo: 1) calcola il degree dei vari nodi
    2) inizia dai nodi con grado più alto e inserisce in uno stesso cluster questo nodo
    con i vicini che hanno un tot di nodi in comune con il nodo di partenza

    :param graph:
    :return:
    """

    clusters = set(u for u in graph.nodes())  # O(n)

    done = False
    while not done:
        pass
