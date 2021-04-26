from itertools import combinations, groupby
from pathlib import Path
import networkx as nx
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import rand_score
import random

COLORS = list(mcolors.BASE_COLORS.keys())
ALL_COLORS = list(mcolors.CSS4_COLORS.keys())

CENTRALITY_MEASURES = {
    "degree_centrality": nx.degree_centrality,
    "closeness_centrality": nx.closeness_centrality,
    "nodes_betweenness_centrality": nx.betweenness_centrality,
    "edges_betweenness_centrality" : nx.edge_betweenness_centrality,
    "pagerank": nx.pagerank
}


class Clustering:
    def __init__(self, clustering_algorithms_with_kwargs, graph, true_clusters, seed=42, draw_graph=False,
                 verbose=False):
        self.clustering_algorithms_with_kwargs = clustering_algorithms_with_kwargs
        self.draw_graph = draw_graph
        self.verbose = verbose
        self.graph = graph
        self.true_clusters = true_clusters
        self.seed = seed

    def __evaluate(self, clustering_algorithm, kwargs):
        print(f"Evaluating {clustering_algorithm.__name__} algorithm:")
        start = time.perf_counter()
        clusters = clustering_algorithm(self.graph, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"The clustering algorithm took {elapsed} seconds")
        cluster_similarity = rand_index(self.graph, clusters, self.true_clusters)
        print(f"The rand index for the clustering algorithm {clustering_algorithm.__name__} is {cluster_similarity}")
        if self.verbose:
            print(f"The graph was divided in {len(clusters)}")
            for i, cluster in enumerate(clusters):
                print(f"The length of the cluster_{i + 1} is {len(cluster)}")
        if self.draw_graph:
            draw_clusters(self.graph, clusters, self.seed)

        return clusters, len(clusters), elapsed, cluster_similarity

    def evaluate_all(self):
        results_dict = {}
        for algorithm, kwargs in self.clustering_algorithms_with_kwargs:
            clusters, num_clusters, time_elapsed, cluster_similarity = self.__evaluate(algorithm, kwargs)
            results_dict[algorithm.__name__] = [clusters, num_clusters, time_elapsed]

        # results = {k: v for k, v in sorted(results_dict.items(), key=lambda item: item[2])}


def build_random_graph(num_of_nodes, probability_of_edge, num_of_clusters, seed=42):
    graph = nx.fast_gnp_random_graph(num_of_nodes, probability_of_edge, seed=seed)
    clusters = generate_random_clusters(graph, num_of_clusters, num_of_nodes)
    return graph, clusters


def build_random_connected_graph(num_of_nodes, probability_of_edge, num_of_clusters, seed=42):
    edges = combinations(range(num_of_nodes), 2)
    graph = nx.Graph()
    random.seed(seed)
    graph.add_nodes_from(range(num_of_nodes))
    if probability_of_edge <= 0:
        return graph
    if probability_of_edge >= 1:
        return nx.complete_graph(num_of_nodes, create_using=graph)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        graph.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < probability_of_edge:
                graph.add_edge(*e)

    clusters = generate_random_clusters(graph, num_of_clusters, num_of_nodes)
    return graph, clusters


def generate_random_clusters(graph, num_of_clusters, num_of_nodes):
    clusters = [[] for _ in range(num_of_clusters)]
    i = 0
    j = 0
    for node in graph.nodes():
        clusters[j].append(node)
        if i > round(num_of_nodes / num_of_clusters):
            i = 0
            j += 1
        i += 1
    return clusters


def load_graph(nodes_path, edges_path, num_of_clusters=4):
    nodes_path = Path(nodes_path).absolute()
    edges_path = Path(edges_path).absolute()

    graph = nx.Graph()

    current_max_index = 0
    label_to_index = {}
    clusters = [[] for _ in range(num_of_clusters)]
    with open(nodes_path, "r", encoding="utf-8") as nodes_file, open(edges_path, "r", encoding="utf-8") as edges_file:
        node_lines = nodes_file.readlines()
        edge_lines = edges_file.readlines()

        for edge in edge_lines[1:]:
            id_1 = edge.split(",")[0].strip()
            id_2 = edge.split(",")[1].strip()
            graph.add_edge(int(id_1), int(id_2))

        for node in node_lines[1:]:
            line = node.split(",")
            identifier = int(line[0].strip())
            label = line[-1].strip()
            try:
                index = label_to_index[label]
            except KeyError:
                label_to_index[label] = current_max_index
                index = current_max_index
                current_max_index += 1

            clusters[index].append(identifier)

    print(f"There are {len(graph)} nodes in the graph")

    return graph, clusters


def clusters_dict_representation(clusters):
    i = 0
    node_to_cluster = {}
    for cluster in clusters:
        for node in cluster:
            node_to_cluster[node] = i
        i += 1
    return node_to_cluster


def rand_index(graph, predicted_clusters, true_clusters):
    node_to_cluster = clusters_dict_representation(predicted_clusters)
    node_to_labels = clusters_dict_representation(true_clusters)

    predicted_list = []
    label_list = []
    for i in range(len(graph)):
        predicted_list.append(node_to_cluster[i])
        label_list.append(node_to_labels[i])

    return rand_score(label_list, predicted_list)


def draw_clusters(graph, clusters, seed):
    pos = nx.spring_layout(graph, seed=seed)
    if len(clusters) > len(COLORS):
        colors = ALL_COLORS
    else:
        colors = COLORS

    for i, cluster in enumerate(clusters):
        nx.draw_networkx_nodes(graph, pos, nodelist=list(cluster), node_color=colors[i])

    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
    plt.show()
