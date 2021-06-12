import time
import networkx as nx
import random
import sys
import logging
import logging_configuration

logger = logging.getLogger()

sys.path.append('../')
from clustering_utils import rand_index, CENTRALITY_MEASURES
from scipy.sparse import linalg
from networkx.linalg.laplacianmatrix import laplacian_matrix
from priorityq import PriorityQueue


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


'''

For the hierarchical clustering algorithm it has been implemented an ad hoc optimization.
In particular it uses data structures, such as dictionaries and set, in order to limit the computation time.
The main data structures used in the algorithm are:
- node_to_cluster: it is a dictionary that maps each node to the belonging cluster.
- cluster_to_nodes: it is a dictionary that maps each cluster to the list of nodes inside it.
- neighbor_clusters: it is a set that contains all the neighbours clusters of a specific cluster.

At the beginning each node is a cluster.
At each iteration:
    1. A cluster is chosen randomly
    2. All the neighbours clusters are selected
    3. Only one of the neighbours clusters, randomly chosen, is merged with the cluster selected at step 1
    4. Repeat until the number of cluster is the desired 

It does not uses optimization like parallelism or sampling because the execution time is in the order of few seconds
on the target graph.

The obtained rand_index is 0.50.

This algorithm has the following output:

```
05-29 08:31 hierarchical_optimized DEBUG    The graph was divided in 4
05-29 08:31 hierarchical_optimized DEBUG    The length of the cluster_1 is 483
05-29 08:31 hierarchical_optimized DEBUG    The length of the cluster_2 is 9335
05-29 08:31 hierarchical_optimized DEBUG    The length of the cluster_3 is 12621
05-29 08:31 hierarchical_optimized DEBUG    The length of the cluster_4 is 31
```

It is possible to see that the algorithm tends to create clusters that are not homogeneous in size.
This can due to the fact that when a cluster becomes bigger, the probability that other clusters will be merged with it
increase.

'''


def hierarchical_optimized(graph, seed=42, desired_clusters=4):
    random.seed(seed)
    node_to_cluster = {}
    cluster_to_nodes = {}

    # Initialization of the data structures containing the clusters
    # At the beginning each node is a cluster
    for i, node in enumerate(graph.nodes):  # O(n)
        node_to_cluster[node] = i
        cluster_to_nodes[i] = [node]

    while len(cluster_to_nodes.keys()) != desired_clusters:
        cluster = random.choice(list(cluster_to_nodes.keys()))

        # edges is a list that contains only the edges incident to the current cluster's nodes
        edges = list(graph.edges(cluster_to_nodes[cluster]))  # O(n*grado(n))
        if not len(edges) == 0:
            # search for neighbours clusters
            neighbor_clusters = set()
            for edge in edges:
                # if there is an edge already inside the current cluster -> continue
                if node_to_cluster[edge[0]] == cluster and node_to_cluster[edge[1]] == cluster:
                    continue
                # chosen_node is the node outside of the current cluster that it's going to be merged
                if node_to_cluster[edge[0]] == cluster:
                    chosen_node = edge[1]
                else:
                    chosen_node = edge[0]
                # It has been used a set() for neighbours clusters because we want to add a possible
                # neighbour cluster only once, in order to have the same probability in choosing the cluster to merge
                neighbor_clusters.add(node_to_cluster[chosen_node])
            chosen_cluster = random.choice(list(neighbor_clusters))
            if not chosen_cluster == cluster:
                # Update the data structures after the merge
                cluster_to_nodes[cluster] = cluster_to_nodes[cluster] + cluster_to_nodes[chosen_cluster]
                for node in cluster_to_nodes[chosen_cluster]:  # O(n)
                    node_to_cluster[node] = cluster
                del (cluster_to_nodes[chosen_cluster])

    return list(cluster_to_nodes.values())


'''

For the k_means_one_iteration clustering algorithm it has been implemented an ad hoc optimization.
In particular it uses data structures, such as dictionaries and set, in order to limit the computation time.
The main data structures used in the algorithm are:

- current_nodes: it is a list that contains the nodes for which the neighbours must be visited
- next_nodes: it is a list that contains the nodes that are going to be analyzed after the current_nodes list is empty
- node_to_cluster: it is a dictionary that maps each node to the belonging cluster.
- cluster_to_nodes: it is a dictionary that maps each cluster to the list of nodes inside it.
- visited_nodes: it is a set that contains all the completely visited nodes in the current iteration.

A node is completely visited when all its neighbours are in a cluster.

At the beginning the current_nodes are initialized with the centers that can be passed to the algorithm,
otherwise choose randomly. Each center is considered as a single node cluster.

At each iteration, while all the nodes of the graph have not been inserted in a cluster:
    1. For each current node select a neighbour that is not already in a cluster,
    add it to the same cluster of the node and to the next_nodes data structure.
    2. If a node has been completely visited, insert it into the visited_nodes data structure and remove it from the 
    current_nodes data structure.
    3. If current_nodes is not empty, repeat from step 1.
    4. Else, update current_nodes with next_nodes and reset next_nodes.
    5. Repeat from step 1.

This algorithm implements a parallel BFS starting from different roots.

'''


def k_means_one_iteration(graph, seed=42, k=4, centers=None):
    current_nodes = []
    next_nodes = []

    # This function return a neighbour of root not in a cluster yet
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

    # Centers initialization, the centers are added into the data structures
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

    # the first loop condition is used to understand all the nodes of the graph have been inserted in a cluster
    while len(current_nodes) != 0:
        # this second loop, on the same condition of the previous one, is used with the for inner loop
        # in order to select only one neighbour of each current node at time, instead of inserting all the neighbours
        # of a node in the same cluster at the same iteration.
        while len(current_nodes) != 0:
            visited_nodes = set()
            for node in current_nodes:
                try:
                    # try to select a neighbour of the current node that is not already in a cluster
                    neighbor = next(non_clustered_neighbors(graph, node))
                    add_node_to_cluster(neighbor, node_to_cluster[node])
                    next_nodes.append(neighbor)
                except StopIteration:
                    visited_nodes.add(node)
            # update current_nodes list removing the nodes that have been completely visited
            # a node is completely visited when all its neighbours are in a cluster
            current_nodes = list(filter(lambda node: node not in visited_nodes, current_nodes))

        current_nodes = next_nodes
        next_nodes = []

    return list(cluster_to_nodes.values())


'''



'''


def k_means(graph, centrality_measure=None, seed=42, k=4, equality_threshold=1e-3, max_iterations=1000, centers=None):
    # [[1, 4], [2, 3, 5]]
    last_clustering = [[] for _ in range(k)]
    last_centers = []
    last_similarity = 0
    convergence = False
    iterations = 0
    centrality_measure_function = CENTRALITY_MEASURES.get(centrality_measure, None)

    # If a centrality measure is specified but the centers are not given as input
    # the centers are chosen using the centrality measure
    if centrality_measure_function is not None and centers is None:
        centers = []
        node_to_centrality = centrality_measure_function(graph)
        nodes = list(node_to_centrality.keys())
        values = list(node_to_centrality.values())
        for i in range(k):
            max_value = max(values)
            values.remove(max_value)
            # Choosing the node with the maximum value of centrality
            # If there are more than one node with the same maximum value of centrality
            # then a node is chosen randomly between them
            center = random.choice([node for node in nodes if node_to_centrality[node] == max_value])
            centers.append(nodes.pop(center))

    # If a centrality measure is not specified and the centers are not given as input
    # the centers are all chosen randomly
    if centrality_measure_function is None and centers is not None:
        random.seed(seed)
    last_centers = centers
    while not convergence and iterations < max_iterations:
        # The centers are None only if they are not passed as input and the centrality measure is not specified
        clusters = k_means_one_iteration(graph, seed, k, centers)  # note: the seed is only used once when centers=None
        if iterations > 0 and centrality_measure is None:
            similarity = rand_index(graph, clusters, last_clustering)
            # logger.debug(f"Difference between two iteration similarity: {abs(last_similarity - similarity)}")
            if abs(last_similarity - similarity) <= equality_threshold:
                logger.info(f"The algorithm reached the convergence at {iterations} iteration with rand index metric")
                convergence = True
            last_similarity = similarity

        if not convergence and iterations < max_iterations:
            centers = []
            # If the centrality measure is None each new center is chosen randomly from a cluster
            if centrality_measure_function is None:
                for cluster in clusters:
                    centers.append(random.choice(cluster))
            else:
                # The center of each cluster is set as the node with the highest centrality measure in the cluster
                for cluster in clusters:
                    node_to_centrality = centrality_measure_function(graph.subgraph(cluster))
                    values = list(node_to_centrality.values())
                    max_value = max(values)
                    center = random.choice([node for node in cluster if node_to_centrality[node] == max_value])
                    centers.append(center)

                if sorted(centers) == sorted(last_centers):
                    convergence = True
                    logger.info(f"The algorithm reached the convergence at {iterations} iteration")
                else:
                    last_centers = centers

        last_clustering = clusters
        iterations += 1

    return last_clustering


def girvan_newman(graph, centrality_measure="edges_betweenness_centrality", seed=42, k=4, verbose=False,
                  optimized=False, decimal_digits=5):
    copy_graph = graph.copy()
    connected_components = []
    pq = PriorityQueue()

    i = 0

    btw_dict = CENTRALITY_MEASURES[centrality_measure](copy_graph, seed=seed)
    for edge, value in btw_dict.items():
        pq.add(edge, -value)

    while len(connected_components) < k:
        edges_to_remove = [pq.pop()]

        while len(pq) != 0 and round(pq.top()[0], decimal_digits) == -round(btw_dict[edges_to_remove[0]],
                                                                            decimal_digits):
            edges_to_remove.append(pq.pop())

        copy_graph.remove_edges_from(edges_to_remove)

        connected_components = list(nx.connected_components(copy_graph))
        if verbose:
            logger.debug(f"The connected components are {len(connected_components)} at the {i} iteration")

        if not optimized:
            btw_dict = CENTRALITY_MEASURES[centrality_measure](copy_graph, seed=seed)
            pq = PriorityQueue()
            for edge, value in btw_dict.items():
                pq.add(edge, -value)
        i += 1

    return connected_components


# Spectral algorithm
def spectral_one_iteration(graph, nodes):
    n = len(nodes)
    time_start = time.perf_counter()
    lap_matrix = laplacian_matrix(graph, nodes).asfptype()
    time_end = time.perf_counter()
    logger.debug(f"The laplacian matrix took {time_end - time_start} seconds")

    time_start = time.perf_counter()
    w, v = linalg.eigsh(lap_matrix, 1)  # check if we must use the greatest eigenvalue or the smallest one
    time_end = time.perf_counter()
    logger.debug(f"The linalg.eigsh took {time_end - time_start} seconds")
    c1 = []
    c2 = []
    time_start = time.perf_counter()
    for i in range(n):
        if v[i, 0] < 0:
            c1.append(nodes[i])
        else:
            c2.append(nodes[i])
    time_end = time.perf_counter()
    logger.debug(f"The for loop took {time_end - time_start} seconds")

    return [c1, c2]


def spectral(graph, k=4):
    next_clusters = []
    curr_clusters = spectral_one_iteration(graph, list(graph.nodes()))
    while len(curr_clusters) < k:
        for cluster in curr_clusters:
            sub_clusters = spectral_one_iteration(graph, cluster)
            for sub_cluster in sub_clusters:
                next_clusters.append(sub_cluster)
        curr_clusters = next_clusters

    return curr_clusters
