import math
from collections import defaultdict
from joblib import Parallel, delayed
import networkx as nx
import utils
from utils import check_hits_convergence
from tqdm import tqdm
import numpy as np
import logging
import logging_configuration
from pytest import approx

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
2021-06-13 12:13:01,319 __evaluate      INFO     Evaluating parallel_closeness_centrality algorithm, with these arguments : {'n_jobs': 8}
2021-06-13 12:17:39,897 __evaluate      INFO     The centrality algorithm: parallel_closeness_centrality took 278.57 seconds
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

The `basic_page_rank` is the iterative implementation of the basic page rank algorithm. It was decided in this version
to not consider the problems due to dead ends and spider traps, because the graph in analysis is undirected and so there
are no dead ends and spider traps.

The implementation of the iterative loop was optimized because, instead of visiting, for each node, all its neighbours,
updating their next page rank (O(n + m)), it goes through all the edges of the graph once (O(m)) and updates the page
rank of each end point of the edge.
This update is performed using two dictionaries, and so in constant time.

The algorithm uses these data structures:

- current_node_to_rank: it is a dictionary mapping each node to its current page rank 
- next_node_to_rank: it is a dictionary used to update the page rank at each iteration

The algorithm follows these steps:

1. The current node to rank is initialized for each node to 1/n, while the next node to rank to 0.
2. For each edge of the graph:
    - Its endpoint are considered
    - Updates next rank of one endpoint with the current rank of the other over its degree
    
```
next_rank[first_endpoint] += (current_rank[second_endpoint] / degree(second_endpoint))
next_rank[second_endpoint] += (current_rank[first_endpoint] / degree(first_endpoint)) 
```

3. If the convergence is reached, then the algorithm ends. 
The algorithm reaches the convergence if the difference between two consecutive ranks for all nodes is less than a
specific threshold that is not absolute, but is relative to the order of magnitude associated to the rank value.
4. If the convergence is not reached, then the next rank is inserted in the current rank and reinitialized to 0.
So repeat the steps from 2.

```
2021-06-13 15:25:48,120 __evaluate                     INFO     Evaluating basic_page_rank algorithm: {'max_iterations': 10000, 'delta_rel': 0.2}
2021-06-13 15:27:03,518 basic_page_rank                INFO     The algorithm has reached convergence at iteration 161
2021-06-13 15:27:03,523 __evaluate                     INFO     The centrality algorithm: basic_page_rank took 75.40 seconds

2021-06-13 16:06:28,041 __evaluate                     INFO     Evaluating basic_page_rank algorithm: {'max_iterations': 10000, 'delta_rel': 0.4}
2021-06-13 16:07:33,329 basic_page_rank                INFO     The algorithm has reached convergence at iteration 135
2021-06-13 16:07:33,331 __evaluate                     INFO     The centrality algorithm: basic_page_rank took 65.289 seconds

2021-06-13 16:10:47,365 __evaluate                     INFO     Evaluating basic_page_rank algorithm: {'max_iterations': 10000, 'delta_rel': 0.6}
2021-06-13 16:11:45,624 basic_page_rank                INFO     The algorithm has reached convergence at iteration 121
2021-06-13 16:11:45,625 __evaluate                     INFO     The centrality algorithm: basic_page_rank took 58.260 seconds

2021-06-13 16:14:37,236 __evaluate                     INFO     Evaluating basic_page_rank algorithm: {'max_iterations': 10000, 'delta_rel': 0.8}
2021-06-13 16:15:28,762 basic_page_rank                INFO     The algorithm has reached convergence at iteration 109
2021-06-13 16:15:28,762 __evaluate                     INFO     The centrality algorithm: basic_page_rank took 51.526 seconds

2021-06-13 16:18:03,536 __evaluate                     INFO     Evaluating basic_page_rank algorithm: {'max_iterations': 10000, 'delta_rel': 1}
2021-06-13 16:18:51,097 basic_page_rank                INFO     The algorithm has reached convergence at iteration 101
2021-06-13 16:18:51,097 __evaluate                     INFO     The centrality algorithm: basic_page_rank took 47.562 seconds
```

'''


def basic_page_rank(graph, max_iterations=100, delta_rel=None):
    """
        Page rank using iterative method

        Preconditions:
            Undirected graph

        max_iterations: number of max iterations to perform
        delta_rel: tolerance threshold for determining convergence. If given, the algorithm stops when the ranks are
            stabilized within the delta_rel variable (for accounting for float precision).
    """

    def check_convergence(current_ranks, next_ranks, delta_rel):
        if delta_rel is None:
            return False
        for node, rank in current_ranks.items():
            next_rank = next_ranks[node]
            if approx(next_rank, rel=delta_rel) != rank:
                return False
        return True

    # Initialize nodes weights
    current_node_to_rank = {}
    next_node_to_rank = {}
    for node in graph.nodes():
        current_node_to_rank[node] = np.float(1 / len(graph))
        next_node_to_rank[node] = np.float(0)

    with tqdm(total=max_iterations) as pbar:
        for i in range(max_iterations):
            for edge in graph.edges():
                first_endpoint = edge[0]
                second_endpoint = edge[1]

                # There are no dead ends and spider traps, the graph is undirected
                next_node_to_rank[first_endpoint] = next_node_to_rank[first_endpoint] + (
                        current_node_to_rank[second_endpoint] * np.float((1 / graph.degree(second_endpoint))))
                next_node_to_rank[second_endpoint] = next_node_to_rank[second_endpoint] + (
                        current_node_to_rank[first_endpoint] * np.float((1 / graph.degree(first_endpoint))))

            if check_convergence(current_node_to_rank, next_node_to_rank, delta_rel):
                logger.info(f"The algorithm has reached convergence at iteration {i}")
                return next_node_to_rank

            for node, rank in next_node_to_rank.items():
                current_node_to_rank[node] = rank
                next_node_to_rank[node] = np.float(0)

            pbar.update(1)

    return current_node_to_rank


'''

The `algebraic_page_rank` is the algebraic implementation of the page rank algorithm.

The algorithm follows these steps:
1. Initialize the rank vector v with the value 1/n 
2. Compute the transition matrix
3. At each iteration:
    - The rank vector is updated with its product with the transition matrix
    - If the convergence is reached, then the algorithm ends, else repeat from 3.

4. If the algorithm does not reaches the convergence in max iterations then it ends.

The check of the convergence is the same used in the basic_page_rank.

```
2021-06-13 15:53:15,668 __evaluate                     INFO     Evaluating algebraic_page_rank algorithm: {'max_iterations': 10000, 'alpha': 1, 'delta_rel': 0.2}
2021-06-13 15:54:22,941 algebraic_page_rank            INFO     The algorithm has reached convergence at iteration 161.
2021-06-13 15:54:23,262 __evaluate                     INFO     The centrality algorithm: algebraic_page_rank took 67.59438390000003 seconds

2021-06-13 16:09:55,159 __evaluate                     INFO     Evaluating algebraic_page_rank algorithm: {'max_iterations': 10000, 'alpha': 1, 'delta_rel': 0.4}
2021-06-13 16:10:47,003 algebraic_page_rank            INFO     The algorithm has reached convergence at iteration 135.
2021-06-13 16:10:47,309 __evaluate                     INFO     The centrality algorithm: algebraic_page_rank took 52.149115600000016 seconds

2021-06-13 16:13:45,160 __evaluate                     INFO     Evaluating algebraic_page_rank algorithm: {'max_iterations': 10000, 'alpha': 1, 'delta_rel': 0.6}
2021-06-13 16:14:36,877 algebraic_page_rank            INFO     The algorithm has reached convergence at iteration 121.
2021-06-13 16:14:37,178 __evaluate                     INFO     The centrality algorithm: algebraic_page_rank took 52.017438700000014 seconds

2021-06-13 16:17:18,057 __evaluate                     INFO     Evaluating algebraic_page_rank algorithm: {'max_iterations': 10000, 'alpha': 1, 'delta_rel': 0.8}
2021-06-13 16:18:03,043 algebraic_page_rank            INFO     The algorithm has reached convergence at iteration 109.
2021-06-13 16:18:03,371 __evaluate                     INFO     The centrality algorithm: algebraic_page_rank took 45.31399070000009 seconds

2021-06-13 16:20:34,706 __evaluate                     INFO     Evaluating algebraic_page_rank algorithm: {'max_iterations': 10000, 'alpha': 1, 'delta_rel': 1}
2021-06-13 16:21:20,211 algebraic_page_rank            INFO     The algorithm has reached convergence at iteration 101.
2021-06-13 16:21:20,529 __evaluate                     INFO     The centrality algorithm: algebraic_page_rank took 45.822299099999896 seconds
```

'''


def algebraic_page_rank(graph, alpha=0.85, max_iterations=100, delta_rel=None):
    def check_convergence(current_ranks, next_ranks, delta_rel):
        if delta_rel is None:
            return False

        for current_rank, next_rank in zip(current_ranks.flat, next_ranks.flat):
            if approx(next_rank, rel=delta_rel) != current_rank:
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
            if check_convergence(current_v, next_v, delta_rel):
                logger.info(f"The algorithm has reached convergence at iteration {i}.")
                current_v = next_v
                break
            current_v = next_v

    node_to_rank = {}
    # Luckily the array index i corresponds to the node index in the graph
    # This happens because all the nodes in the graph are numbered from 0 to len(graph)
    for i in range(len(graph)):
        node_to_rank[i] = current_v.item(i)

    return node_to_rank


'''

The parallel_basic_page_rank is the parallel implementation of the basic_page_rank algorithm.
In particular it is exactly the same algorithm, but the edges of the graph are split in non overlapping chunks 
that are given in input to different jobs.

Each job, computes a partial next rank for the endpoints of the given edges, considering
only the rank coming from the edges given as input.
 
The aggregation phase is very simple, because it consists in the sum of all the partial
results computed by the jobs.

```
2021-06-13 15:50:27,804 __evaluate                     INFO     Evaluating parallel_basic_page_rank algorithm: {'max_iterations': 10000, 'delta_rel': 0.2, 'jobs': 8}
2021-06-13 15:53:15,617 parallel_basic_page_rank       INFO     The algorithm has reached convergence at iteration 161.
2021-06-13 15:53:15,624 __evaluate                     INFO     The centrality algorithm: parallel_basic_page_rank took 167.82 seconds

2021-06-13 16:07:33,403 __evaluate                     INFO     Evaluating parallel_basic_page_rank algorithm: {'max_iterations': 10000, 'delta_rel': 0.4, 'jobs': 8}
2021-06-13 16:09:55,129 parallel_basic_page_rank       INFO     The algorithm has reached convergence at iteration 135.
2021-06-13 16:09:55,137 __evaluate                     INFO     The centrality algorithm: parallel_basic_page_rank took 141.73seconds

2021-06-13 16:11:45,796 __evaluate                     INFO     Evaluating parallel_basic_page_rank algorithm: {'max_iterations': 10000, 'delta_rel': 0.6, 'jobs': 8}
2021-06-13 16:13:45,033 parallel_basic_page_rank       INFO     The algorithm has reached convergence at iteration 121.
2021-06-13 16:13:45,040 __evaluate                     INFO     The centrality algorithm: parallel_basic_page_rank took 119.244 seconds

2021-06-13 16:15:28,789 __evaluate                     INFO     Evaluating parallel_basic_page_rank algorithm: {'max_iterations': 10000, 'delta_rel': 0.8, 'jobs': 8}
2021-06-13 16:17:18,017 parallel_basic_page_rank       INFO     The algorithm has reached convergence at iteration 109.
2021-06-13 16:17:18,024 __evaluate                     INFO     The centrality algorithm: parallel_basic_page_rank took 109.236 seconds

2021-06-13 16:18:51,124 __evaluate                     INFO     Evaluating parallel_basic_page_rank algorithm: {'max_iterations': 10000, 'delta_rel': 1, 'jobs': 8}
2021-06-13 16:20:34,558 parallel_basic_page_rank       INFO     The algorithm has reached convergence at iteration 101.
2021-06-13 16:20:34,568 __evaluate                     INFO     The centrality algorithm: parallel_basic_page_rank took 103.4429374 seconds
```


Comparison between the different implementation of the page rank algorithm:

As it is evident from the results, the higher the parameter delta_rel is, the lower is the number of iterations and consequently the running
time of the algorithm.

The results obtained from the parallel version are exactly the same to the ones obtained with the basic one. Instead the results
obtained with the algebraic version are almost equal, except for small variations in the lasts decimal digits.

The parallel version is the slowest, this can due to the fact that the algorithm itself is fast, and so the time overhead
due to the creation of the different jobs, the creation of the chunks and the aggregation phase, is greater than the speed up of the algorithm itself.

The algebraic version of the algorithm is the fastest, but it is very memory expensive due to the computation of the transition matrix, infact this 
matrix is in the order of n^2 in memory, that for the given network is more or less 12 GB. 

The basic version is almost as fast as the algebraic one, but it is not as memory expensive as the algebraic one.

'''


def parallel_basic_page_rank(graph, max_iterations=100, jobs=8, delta_rel=None):
    def check_convergence(current_ranks, next_ranks, delta_rel):
        if delta_rel is None:
            return False
        for node, rank in current_ranks.items():
            next_rank = next_ranks[node]
            if approx(next_rank, rel=delta_rel) != rank:
                return False
        return True

    def chunked_page_rank_step(edges, current_node_to_rank):
        next_node_to_rank = {node: 0 for node in current_node_to_rank.keys()}
        for edge in edges:
            first_endpoint = edge[0]
            second_endpoint = edge[1]

            next_node_to_rank[first_endpoint] = next_node_to_rank[first_endpoint] + (
                    current_node_to_rank[second_endpoint] * np.float((1 / graph.degree(second_endpoint))))
            next_node_to_rank[second_endpoint] = next_node_to_rank[second_endpoint] + (
                    current_node_to_rank[first_endpoint] * np.float((1 / graph.degree(first_endpoint))))
        return next_node_to_rank

    def aggregate_results(results):
        aggregated = defaultdict(int)  # If the key is not already in the dict, then it is initialized with 0
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

            for i in range(max_iterations):
                results = parallel(
                    delayed(chunked_page_rank_step)(edges_chunk, current_node_to_rank) for edges_chunk in edges_chunks)
                next_node_to_rank = aggregate_results(results)
                if check_convergence(current_node_to_rank, next_node_to_rank, delta_rel):
                    logger.info(f"The algorithm has reached convergence at iteration {i}.")
                    return next_node_to_rank

                current_node_to_rank = next_node_to_rank
                pbar.update(1)

    return current_node_to_rank


'''

The `naive_edge_hits` is one of the four implementation of the hubs and authorities (HITS) algorithm.
The implementation of this algorithm has been done with the purpose of optimizing the main iterative loop. In fact,
instead of visiting the neighbourhood of each node, the loop is performed on the edges, in order to have
a computational time in the order of m instead of (m + n).
Unfortunately, using this strategy, the algorithm becomes less readable and the running time slightly increases. 
This is due to the fact that there is the need of initializing to zero the authority of each node at each iteration. 

The algorithm uses these data structures:

- node_to_authorities: it is a dictionary mapping each node to its current authority value
- node_to_hubs: it is a dictionary mapping each node to its current hub value
- last_node_to_hubs: this is the previous iteration node_to_hubs dictionary, used to check the convergence

The algorithm follows these steps:

1. The current node to hubs are initialized, for each node, to 1.
2. For each node the authorities are initialized to 0
3. For each edge of the graph:
    - Its endpoint are considered
    - Updates the authority value of one endpoint with the current hub value of the other
4. All the authority values are normalized
5. All the hubs are re initialized to 0
6. For each edge of the graph:
    - Its endpoint are considered
    - Updates the hub value of one endpoint with the current authority value of the other
7. All the hub values are normalized
8. If the convergence or the maximum number of iteration has been reached, the algorithm ends and returns the two dictionaries
The algorithm reaches the convergence if the sum of the absolute value of the differences between two consecutive hub
values is lower than the specified threshold.
9. Else repeat from step 2.

```

2021-06-14 11:51:22,728 __evaluate          INFO     Evaluating naive_edge_hits algorithm, with these arguments : {'max_iterations': 100}
2021-06-14 11:51:26,834 naive_edge_hits     INFO     The algorithm has reached convergence at iteration 22.
2021-06-14 11:51:26,836 __evaluate          INFO     The centrality algorithm: naive_edge_hits took 4.10 seconds
 
```

'''


def naive_edge_hits(graph, max_iterations=100, tol=1e-8):
    node_to_authorities = {}
    node_to_hubs = {}
    last_node_to_hubs = {}

    for node in graph.nodes():
        node_to_hubs[node] = 1

    with tqdm(total=max_iterations) as pbar:
        for i in range(max_iterations):

            for node in graph.nodes():
                node_to_authorities[node] = 0

            for edge in graph.edges():
                first_endpoint = edge[0]
                second_endpoint = edge[1]

                node_to_authorities[first_endpoint] += node_to_hubs[second_endpoint]
                node_to_authorities[second_endpoint] += node_to_hubs[first_endpoint]

            sum_of_authorities = sum(node_to_authorities.values())

            for node in graph.nodes():
                node_to_authorities[node] = node_to_authorities[node] / sum_of_authorities
                node_to_hubs[node] = 0

            for edge in graph.edges():
                first_endpoint = edge[0]
                second_endpoint = edge[1]

                node_to_hubs[first_endpoint] += node_to_authorities[second_endpoint]
                node_to_hubs[second_endpoint] += node_to_authorities[first_endpoint]

            sum_of_hubs = sum(node_to_hubs.values())

            for node in graph.nodes():
                node_to_hubs[node] = node_to_hubs[node] / sum_of_hubs

            if i > 0 and check_hits_convergence(graph, node_to_hubs, last_node_to_hubs, tol):
                logger.info(f"The algorithm has reached convergence at iteration {i}.")
                break

            last_node_to_hubs = node_to_hubs.copy()
            pbar.update(1)

    return node_to_hubs, node_to_authorities


'''

The `parallel_edge_hits` is the parallel implementation of the `naive_edge_hits` algorithm.
In particular it is exactly the same algorithm, but the edges of the graph are split in non overlapping chunks 
that are given in input to different jobs.

1. Each job, firstly computes a partial authority value for the endpoints of the given edges, considering 
only the hub values coming from the edges given as input.
2. The partial authorities are aggregated in order to compute the authority of each node.
3. Each job, then computes a partial hub value for the endpoints of the given edges, considering 
only the authority values coming from the edges given as input.
4. The partial hubs are aggregated in order to compute the hub of each node.

```

2021-06-14 11:51:26,935 __evaluate              INFO     Evaluating parallel_edge_hits algorithm, with these arguments : {'max_iterations': 100}
2021-06-14 11:52:11,642 parallel_edge_hits      INFO     The algorithm has reached convergence at iteration 22.
2021-06-14 11:52:11,652 __evaluate              INFO     The centrality algorithm: parallel_edge_hits took 44.71 seconds

```

The results obtained from the parallel version are exactly the same to the ones obtained with the non parallel one.

The parallel version is the slowest, this can due to the fact that the algorithm itself is fast, 
and so the time overhead due to the creation of the different jobs, the creation of the chunks and
the aggregation phase, that are performed twice, is greater than the speed up of the algorithm itself.

The non parallel version is faster than the one taken from networkx library, as you can see from the results below:

```

2021-06-14 12:54:22,313 __evaluate      INFO     Evaluating hits algorithm, with these arguments : {'max_iter': 100}
2021-06-14 12:54:40,418 __evaluate      INFO     The centrality algorithm: hits took 18.104869 seconds

```

'''


def parallel_edge_hits(graph, max_iterations=100, jobs=8, tol=1.0e-8):
    def chunked_update_step(edges, node_to_value):
        partial_node_to_value = {node: 0 for node in graph.nodes()}
        for edge in edges:
            first_endpoint = edge[0]
            second_endpoint = edge[1]

            partial_node_to_value[first_endpoint] += node_to_value[second_endpoint]
            partial_node_to_value[second_endpoint] += node_to_value[first_endpoint]

        return partial_node_to_value

    node_to_authorities = {}
    node_to_hubs = {}
    last_node_to_hubs = {}

    for node in graph.nodes():
        node_to_hubs[node] = 1

    with tqdm(total=max_iterations) as pbar:
        with Parallel(n_jobs=jobs) as parallel:
            edges_chunks = []
            chunk_size = math.ceil(len(graph.edges) / jobs)
            for i in range(jobs):
                edges_chunks.append(list(graph.edges())[i * chunk_size: (i + 1) * chunk_size])

            for i in range(max_iterations):

                for node in graph.nodes():
                    node_to_authorities[node] = 0

                authorities_results = parallel(
                    delayed(chunked_update_step)(edges_chunk, node_to_hubs) for edges_chunk in edges_chunks)

                for authority_results in authorities_results:
                    for node, authority in authority_results.items():
                        node_to_authorities[node] += authority

                sum_of_authorities = sum(node_to_authorities.values())
                for node in graph.nodes():
                    node_to_authorities[node] = node_to_authorities[node] / sum_of_authorities
                    node_to_hubs[node] = 0

                hubs_results = parallel(
                    delayed(chunked_update_step)(edges_chunk, node_to_authorities) for edges_chunk in edges_chunks)

                for hub_results in hubs_results:
                    for node, hub in hub_results.items():
                        node_to_hubs[node] += hub

                sum_of_hubs = sum(node_to_hubs.values())

                for node in graph.nodes():
                    node_to_hubs[node] = node_to_hubs[node] / sum_of_hubs

                if i > 0 and check_hits_convergence(graph, node_to_hubs, last_node_to_hubs, tol):
                    logger.info(f"The algorithm has reached convergence at iteration {i}.")
                    break

                last_node_to_hubs = node_to_hubs.copy()
                pbar.update(1)

    return node_to_hubs, node_to_authorities


'''

The `naive_hits` is the third of the four implementation of the hubs and authorities (HITS) algorithm.
The implementation of the algorithm follows the classical theoretical algorithm. In fact it iterates over all the nodes 
and updates their authorities and hubs using their neighbours.

The algorithm uses these data structures:

- node_to_authorities: it is a dictionary mapping each node to its current authority value
- node_to_hubs: it is a dictionary mapping each node to its current hub value
- last_node_to_hubs: this is the previous iteration node_to_hubs dictionary, used to check the convergence

The algorithm follows these steps:

1. The current node to hubs are initialized, for each node, to 1.
2. For each node:
    - Its authority value is updated with the sum of the hub values of its neighbours
3. All the authority values are normalized
4. For each node:
    - Its hub value is updated with the sum of the authority values of its neighbours
5. All the hub values are normalized
6. If the convergence or the maximum number of iteration has been reached, the algorithm ends and returns the two dictionaries
The algorithm reaches the convergence if the sum of the absolute value of the differences between two consecutive hub
values is lower than the specified threshold.
7. Else repeat from step 2.

```

2021-06-14 11:52:11,763 __evaluate      INFO     Evaluating naive_hits algorithm, with these arguments : {'max_iterations': 100}
2021-06-14 11:52:14,609 naive_hits      INFO     The algorithm has reached convergence at iteration 22.
2021-06-14 11:52:14,610 __evaluate      INFO     The centrality algorithm: naive_hits took 2.84 seconds

```

'''


def naive_hits(graph, max_iterations=100, tol=1e-8):
    node_to_authorities = {}
    node_to_hubs = {}
    last_node_to_hubs = {}

    for node in graph.nodes():
        node_to_hubs[node] = 1

    with tqdm(total=max_iterations) as pbar:
        for i in range(max_iterations):
            for node in graph.nodes():
                node_to_authorities[node] = sum([node_to_hubs[neighbour] for neighbour in graph[node]])

            sum_of_authorities = sum(node_to_authorities.values())

            for node in graph.nodes():
                node_to_authorities[node] = node_to_authorities[node] / sum_of_authorities

            for node in graph.nodes():
                node_to_hubs[node] = sum([node_to_authorities[neighbour] for neighbour in graph[node]])

            sum_of_hubs = sum(node_to_hubs.values())

            for node in graph.nodes():
                node_to_hubs[node] = node_to_hubs[node] / sum_of_hubs

            if i > 0 and check_hits_convergence(graph, node_to_hubs, last_node_to_hubs, tol):
                logger.info(f"The algorithm has reached convergence at iteration {i}.")
                break

            last_node_to_hubs = node_to_hubs.copy()
            pbar.update(1)

    return node_to_hubs, node_to_authorities


'''

The `parallel_naive_hits` is the parallel implementation of the `naive_hits` algorithm.
In particular, the nodes of the graph are split in non overlapping chunks that are given in input to different jobs.

1. Each job for the each node of the given chunk:
    - Firstly computes the authority value.
    - Sum the authority value computed in the previous step, to the hub value of its neighbours, 
    computing a partial hub value for them.
2. In the aggregation phase:
    - The authority values are merged in a single structure.
    - The partial hub values are sum together in order to obtain the final hub value for each node.
3. In this algorithm the normalization phase is performed at the end of each iteration.

The convergence check is the same of the other hits implementation.

```

2021-06-14 11:52:14,656 __evaluate              INFO     Evaluating parallel_naive_hits algorithm, with these arguments : {'max_iterations': 100}
2021-06-14 11:52:36,253 parallel_naive_hits     INFO     The algorithm has reached convergence at iteration 22.
2021-06-14 11:52:36,254 __evaluate              INFO     The centrality algorithm: parallel_naive_hits took 21.59 seconds

```

The results obtained from the parallel version are exactly the same to the ones obtained with the non parallel one.

The parallel version is slower than the non parallel one but is faster than the `parallel_edge_hits`. This is due to
the fact that in this parallel version the parallelization and the aggregation phases are performed only once at each
iteration.

Final considerations:

All the four implementations return the same results and, in fact, they all reach the convergence at the same iteration.
The fastest implementation is the `naive_hits` that only takes 2.84 seconds (on our computer).

'''


def parallel_naive_hits(graph, max_iterations=100, jobs=8, tol=1.0e-8):
    def chunked_update_step(nodes, node_to_hubs):
        chunked_node_to_authoritiy = {node: 0 for node in nodes}
        partial_node_to_hub = {node: 0 for node in graph.nodes()}
        for node in nodes:
            chunked_node_to_authoritiy[node] = sum([node_to_hubs[neighbour] for neighbour in graph[node]])
            for neighbour in graph[node]:
                partial_node_to_hub[neighbour] += chunked_node_to_authoritiy[node]

        return partial_node_to_hub, chunked_node_to_authoritiy

    node_to_authorities = {}
    node_to_hubs = {}
    last_node_to_hubs = {}

    for node in graph.nodes():
        node_to_authorities[node] = 0
        node_to_hubs[node] = 1

    with tqdm(total=max_iterations) as pbar:
        with Parallel(n_jobs=jobs) as parallel:
            for i in range(max_iterations):
                partial_results = parallel(delayed(chunked_update_step)(chunk, node_to_hubs) for chunk in
                                           utils.chunks(graph.nodes(), math.ceil(len(graph.nodes()) / jobs)))

                for node in graph.nodes():
                    node_to_hubs[node] = 0

                for job_result in partial_results:
                    for node in graph.nodes():
                        node_to_hubs[node] += job_result[0][node]
                        try:
                            node_to_authorities[node] = job_result[1][node]
                        except KeyError:
                            continue

                sum_of_hubs = sum(node_to_hubs.values())
                sum_of_authorities = sum(node_to_authorities.values())
                for node in graph.nodes():
                    node_to_hubs[node] = node_to_hubs[node] / sum_of_hubs
                    node_to_authorities[node] = node_to_authorities[node] / sum_of_authorities

                if i > 0 and check_hits_convergence(graph, node_to_hubs, last_node_to_hubs, tol):
                    logger.info(f"The algorithm has reached convergence at iteration {i}.")
                    break

                last_node_to_hubs = node_to_hubs.copy()
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


def parallel_edges_betweenness_centrality(graph, **kwargs):
    edge_btw, _ = parallel_betweenness_centrality(graph, n_jobs=16)
    return edge_btw
