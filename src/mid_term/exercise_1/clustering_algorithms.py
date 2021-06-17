import time
import networkx as nx
import random
import logging
from clustering_utils import rand_index, CENTRALITY_MEASURES
from scipy.sparse import linalg
from networkx.linalg.laplacianmatrix import laplacian_matrix
from utils.priorityq import PriorityQueue
from final_term.exercise_2 import logging_configuration

logger = logging.getLogger()
logging_configuration.set_logging()

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

- current_nodes: it is a list that contains the nodes for which the neighbours must be visited.
- next_nodes: it is a list that contains the nodes that are going to be analyzed after the current_nodes list is empty.
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

The k_means algorithm execute the k_means_one_iteration `max_iterations` times if it does not converge before.
The execution of this algorithm depends on the parameters given as input:

1. If a centrality measure is not specified and the centers are not specified:
    - The centers are randomly chosen at each iteration
    - The convergence is evaluated using the rand index metric. In particular it is computed the `rand_index`
    between two consecutive graph clustering. This metric represents a similarity of two clustering. If at the next
    iteration this similarity doesn't change more than a `equality_threshold`, then the obtained clustering
    is very similar to the previous one.
2. If a centrality measure is not specified and the centers are specified:
    - At the first iteration the algorithm uses the given centers and then it follows the same steps of the first case
3. If a centrality measure is specified and the centers are not specified:
    - The centers are computed at each iteration using the K nodes with the highest centrality value. At the first
    iteration the K centers are chosen from the whole graph.
    - From the second iteration until the convergence, the new center of each cluster is chosen as the node with the
    maximum centrality value inside that cluster.
    - The algorithm converge if the centers of the clusters do not change between two consecutive iterations.
4. If a centrality measure is specified and the centers are specified:
    - At the first iteration the algorithm uses the given centers and then it follows the same steps of the third case

It was decided to not parallelize the algorithm because the bottleneck of the execution time is represented by the
computation of the centrality measures. In fact choosing a centrality measure like the degree_centrality or choosing the
nodes randomly, it takes few seconds to converge.

Changing the centrality measures, the clustering results differ a lot from each other.
In particular the best result, in terms of rand index, is obtained with the degree centrality, 
that is also the fastest one.

The rand index for the clustering algorithm k_means using degree_centrality is 0.63 and it takes 1.11 seconds.

```

2021-06-12 13:21:39,840 __evaluate                     INFO     Evaluating k_means_one_iteration algorithm, with these arguments : {'seed': 42, 'k': 4}
2021-06-12 13:21:39,940 __evaluate                     INFO     The clustering algorithm: k_means_one_iteration took 0.09838740000000001 seconds
2021-06-12 13:21:39,963 __evaluate                     INFO     The rand index for the clustering algorithm k_means_one_iteration is 0.5416639090721305
2021-06-12 13:21:39,963 __evaluate                     DEBUG    The graph was divided in 4
2021-06-12 13:21:39,963 __evaluate                     DEBUG    The length of the cluster_1 is 799
2021-06-12 13:21:39,963 __evaluate                     DEBUG    The length of the cluster_2 is 3615
2021-06-12 13:21:39,963 __evaluate                     DEBUG    The length of the cluster_3 is 3269
2021-06-12 13:21:39,963 __evaluate                     DEBUG    The length of the cluster_4 is 14787

2021-06-12 13:21:39,963 __evaluate                     INFO     Evaluating k_means algorithm, with these arguments : {'centrality_measure': None, 'seed': 42, 'k': 4, 'equality_threshold': 0.001, 'max_iterations': 1, 'centers': None}
2021-06-12 13:21:40,071 __evaluate                     INFO     The clustering algorithm: k_means took 0.10676179999999968 seconds
2021-06-12 13:21:40,083 __evaluate                     INFO     The rand index for the clustering algorithm k_means is 0.5416639090721305
2021-06-12 13:21:40,083 __evaluate                     DEBUG    The graph was divided in 4
2021-06-12 13:21:40,083 __evaluate                     DEBUG    The length of the cluster_1 is 799
2021-06-12 13:21:40,083 __evaluate                     DEBUG    The length of the cluster_2 is 3615
2021-06-12 13:21:40,083 __evaluate                     DEBUG    The length of the cluster_3 is 3269
2021-06-12 13:21:40,083 __evaluate                     DEBUG    The length of the cluster_4 is 14787

2021-06-12 13:21:40,084 __evaluate                     INFO     Evaluating k_means algorithm, with these arguments : {'centrality_measure': None, 'seed': 42, 'k': 4, 'equality_threshold': 0.001, 'max_iterations': 10000, 'centers': None}
2021-06-12 13:21:50,116 k_means                        INFO     The algorithm reached the convergence at 89 iteration with rand index metric
2021-06-12 13:21:50,117 __evaluate                     INFO     The clustering algorithm: k_means took 10.0337714 seconds
2021-06-12 13:21:50,128 __evaluate                     INFO     The rand index for the clustering algorithm k_means is 0.4067197443947051
2021-06-12 13:21:50,128 __evaluate                     DEBUG    The graph was divided in 4
2021-06-12 13:21:50,128 __evaluate                     DEBUG    The length of the cluster_1 is 495
2021-06-12 13:21:50,128 __evaluate                     DEBUG    The length of the cluster_2 is 19401
2021-06-12 13:21:50,128 __evaluate                     DEBUG    The length of the cluster_3 is 2494
2021-06-12 13:21:50,128 __evaluate                     DEBUG    The length of the cluster_4 is 80

2021-06-12 13:21:50,129 __evaluate                     INFO     Evaluating k_means algorithm, with these arguments : {'centrality_measure': 'degree_centrality', 'seed': 42, 'k': 4, 'equality_threshold': 0.001, 'max_iterations': 10000, 'centers': None}
2021-06-12 13:21:50,723 k_means                        INFO     The algorithm reached the convergence at 1 iteration
2021-06-12 13:21:50,723 __evaluate                     INFO     The clustering algorithm: k_means took 0.5947337999999984 seconds
2021-06-12 13:21:50,737 __evaluate                     INFO     The rand index for the clustering algorithm k_means is 0.6354478760362172
2021-06-12 13:21:50,737 __evaluate                     DEBUG    The graph was divided in 4
2021-06-12 13:21:50,737 __evaluate                     DEBUG    The length of the cluster_1 is 3669
2021-06-12 13:21:50,737 __evaluate                     DEBUG    The length of the cluster_2 is 9576
2021-06-12 13:21:50,737 __evaluate                     DEBUG    The length of the cluster_3 is 5056
2021-06-12 13:21:50,737 __evaluate                     DEBUG    The length of the cluster_4 is 4169

2021-06-12 13:21:50,737 __evaluate                     INFO     Evaluating k_means algorithm, with these arguments : {'centrality_measure': 'nodes_betweenness_centrality', 'seed': 42, 'k': 4, 'equality_threshold': 0.001, 'max_iterations': 10, 'centers': None}
2021-06-12 15:53:59,842 k_means                        INFO     The algorithm reached the convergence at 1 iteration
2021-06-12 15:53:59,842 __evaluate                     INFO     The clustering algorithm: k_means took 9129.104045099999 seconds
2021-06-12 15:53:59,854 __evaluate                     INFO     The rand index for the clustering algorithm k_means is 0.5328093378835772
2021-06-12 15:53:59,854 __evaluate                     DEBUG    The graph was divided in 4
2021-06-12 15:53:59,854 __evaluate                     DEBUG    The length of the cluster_1 is 14483
2021-06-12 15:53:59,854 __evaluate                     DEBUG    The length of the cluster_2 is 3793
2021-06-12 15:53:59,854 __evaluate                     DEBUG    The length of the cluster_3 is 3788
2021-06-12 15:53:59,855 __evaluate                     DEBUG    The length of the cluster_4 is 406

2021-06-12 15:53:59,855 __evaluate                     INFO     Evaluating k_means algorithm, with these arguments : {'centrality_measure': 'pagerank', 'seed': 42, 'k': 4, 'equality_threshold': 0.001, 'max_iterations': 50, 'centers': None}
2021-06-12 15:54:11,142 k_means                        INFO     The algorithm reached the convergence at 1 iteration
2021-06-12 15:54:11,143 __evaluate                     INFO     The clustering algorithm: k_means took 11.286865399999442 seconds
2021-06-12 15:54:11,154 __evaluate                     INFO     The rand index for the clustering algorithm k_means is 0.4661067140459932
2021-06-12 15:54:11,154 __evaluate                     DEBUG    The graph was divided in 4
2021-06-12 15:54:11,154 __evaluate                     DEBUG    The length of the cluster_1 is 17168
2021-06-12 15:54:11,154 __evaluate                     DEBUG    The length of the cluster_2 is 776
2021-06-12 15:54:11,155 __evaluate                     DEBUG    The length of the cluster_3 is 1713
2021-06-12 15:54:11,155 __evaluate                     DEBUG    The length of the cluster_4 is 2813

```

The fact that, using the degree centrality to choose the centers, gives a more balanced and better clustering result,
can be explained as follows: starting from K nodes, that are the ones with the maximum degree, the algorithm is able to
explore the graph more homogeneously, avoiding the creation of giant clusters. 

'''


def k_means(graph, centrality_measure=None, seed=42, k=4, equality_threshold=1e-3, max_iterations=1000, centers=None):
    # [[1, 4], [2, 3, 5]]
    last_clustering = [[] for _ in range(k)]
    last_similarity = 1  # initialized to 1 in order to avoiding errors due to the second iteration
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


'''

For the girvan newman clustering algorithm it has been implemented an ad hoc optimization that evaluates the edge
betweenness only once and not at each iteration as the real algorithm defines. This can lead to solutions that are
different from the original algorithm ones.

Anyway it uses data structures, such as dictionary and priority queue, in order to limit the computation time.
The main data structures used in the algorithm are:

- btw_dict: it is a dictionary that maps each edge to the corresponding betweenness.
- pq: it is a min priority queue, used in order to get edges with the maximum betweenness in logarithmic time.

It does not uses optimization like parallelism or sampling because the bottleneck of the execution time is represented
by the computation of the edge betweenness centrality measure.

It has been tested only with the optimization parameter set as true, because computing the edge betweenness at each
iteration, was too time expensive for the purposes of this project.

The obtained rand_index is very low, it is 0.26. In fact, as we can see from the result below, the graph was divided in 
six clusters, in which it is present a giant component. This can be due to the fact that the network is composed of 
small communities that are attached to the giant component through bridges.

This algorithm has the following output:

```
05-30 14:07 girvan_newman INFO     Evaluating girvan_newman algorithm, with these arguments : {'centrality_measure': 'edges_betweenness_centrality', 'seed': 42, 'k': 4, 'optimized': True}
05-30 15:18 girvan_newman INFO     The clustering algorithm: girvan_newman took 4217 seconds
05-30 15:18 girvan_newman INFO     The rand index for the clustering algorithm girvan_newman is 0.26791726871754057
05-30 15:18 girvan_newman DEBUG    The graph was divided in 6
05-30 15:18 girvan_newman DEBUG    The length of the cluster_1 is 22407
05-30 15:18 girvan_newman DEBUG    The length of the cluster_2 is 20
05-30 15:18 girvan_newman DEBUG    The length of the cluster_3 is 14
05-30 15:18 girvan_newman DEBUG    The length of the cluster_4 is 14
05-30 15:18 girvan_newman DEBUG    The length of the cluster_5 is 14
05-30 15:18 girvan_newman DEBUG    The length of the cluster_6 is 1
```

It is possible to see that the algorithm tends to create clusters that are not homogeneous in size.
This can due to the fact that when a cluster becomes bigger, the probability that other clusters will be merged with it
increase.

'''


def girvan_newman(graph, centrality_measure="edges_betweenness_centrality", seed=42, k=4,
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

        # takes all the edges associated with the maximum btw values
        while len(pq) != 0 and round(pq.top()[0], decimal_digits) == -round(btw_dict[edges_to_remove[0]],
                                                                            decimal_digits):
            edges_to_remove.append(pq.pop())

        copy_graph.remove_edges_from(edges_to_remove)

        connected_components = list(nx.connected_components(copy_graph))
        logger.debug(f"The connected components are {len(connected_components)} at the {i} iteration")

        if not optimized:
            btw_dict = CENTRALITY_MEASURES[centrality_measure](copy_graph, seed=seed)
            pq = PriorityQueue()
            for edge, value in btw_dict.items():
                pq.add(edge, -value)
        i += 1

    return connected_components


'''

The spectral algorithm has not been optimized because the execution time was already feasible, in the order of seconds.
The only constraint of this implementation is that at each iteration the algorithm splits each cluster in two sub 
clusters.
At the beginning the whole graph is split in two clusters using the `spectral_one_iteration` function:

1. The laplacian matrix of the cluster passed as input is computed.
2. The the greatest eigenvalue and the associated eigenvector is computed.
3. The cluster is split in two sub clusters following this rule:
    - The nodes associated with positive values in the eigenvector are assigned to the first cluster
    - The nodes associated with negative values in the eigenvector are assigned to the second cluster

This function is repeated until the desired number of cluster (that must be a power of 2) is reached.

```
05-30 15:18 spectral     INFO     Evaluating spectral algorithm, with these arguments : {'k': 4}
05-30 15:18 spectral     INFO     The clustering algorithm: spectral took 0.9156270999992557 seconds
05-30 15:18 spectral     INFO     The rand index for the clustering algorithm spectral is 0.617678996506149
05-30 15:18 spectral     DEBUG    The graph was divided in 4
05-30 15:18 spectral     DEBUG    The length of the cluster_1 is 5102
05-30 15:18 spectral     DEBUG    The length of the cluster_2 is 5259
05-30 15:18 spectral     DEBUG    The length of the cluster_3 is 6065
05-30 15:18 spectral     DEBUG    The length of the cluster_4 is 6044
```

The obtained rand index using this algorithm is one of the best, although the output of this algorithm is hard
to interpret.

'''


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
