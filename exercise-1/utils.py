from pathlib import Path
import networkx as nx
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
from sklearn.metrics import rand_score

COLORS = list(mcolors.BASE_COLORS.keys())
ALL_COLORS = list(mcolors.CSS4_COLORS.keys())


def build_random_graph(num_of_nodes, probability_of_edge, num_of_clusters, seed=42):
    graph = nx.fast_gnp_random_graph(num_of_nodes, probability_of_edge, seed=seed)

    node_to_labels = {}

    i = 0
    j = 0
    for node in graph.nodes():
        node_to_labels[node] = j
        if i >= math.ceil(num_of_nodes / num_of_clusters):
            i = 0
            j += 1
        i += 1

    return graph, node_to_labels


def load_graph(nodes_path, edges_path):
    nodes_path = Path(nodes_path).absolute()
    edges_path = Path(edges_path).absolute()

    graph = nx.Graph()

    node_to_label = {}
    with open(nodes_path, "r", encoding="utf-8") as nodes_file, open(edges_path, "r", encoding="utf-8") as edges_file:
        node_lines = nodes_file.readlines()
        edge_lines = edges_file.readlines()

        for edge in edge_lines[1:]:
            id_1 = edge.split(",")[0].strip()
            id_2 = edge.split(",")[1].strip()
            graph.add_edge(id_1, id_2)

        for node in node_lines[1:]:
            line = node.split(",")
            identifier = line[0].strip()
            label = line[-1].strip()

            node_to_label[identifier] = label

    print(f"There are {len(node_to_label)} in the dictionary")
    print(f"There are {len(graph)} nodes in the graph")

    return graph, node_to_label


class Clustering:
    def __init__(self, clustering_algorithm_list, graph, node_to_labels, draw_graph=False, verbose=False):
        self.clustering_algorithm_list = clustering_algorithm_list
        self.draw_graph = draw_graph
        self.verbose = verbose
        self.graph = graph
        self.node_to_labels = node_to_labels

    def __evaluate(self, clustering_algorithm):
        print(f"Evaluating {clustering_algorithm.__name__} algorithm:")
        start = time.perf_counter()
        clusters = clustering_algorithm(self.graph)
        end = time.perf_counter()
        elapsed = end - start
        print(f"The clustering algorithm took {elapsed} seconds")
        rand_index = self.__rand_index(clusters)
        print(f"The rand index for the clustering algorithm {clustering_algorithm.__name__} is {rand_index}")
        if self.verbose:
            print(f"The graph was divided in {len(clusters)} clusters that are : {clusters}")
        if self.draw_graph:
            self.__draw_clusters(clusters)

        return clusters, len(clusters), elapsed, rand_index

    def __rand_index(self, clusters):
        i = 0
        node_to_cluster = {}
        for cluster in list(clusters):
            cluster = list(cluster)
            for node in cluster:
                node_to_cluster[node] = i
            i += 1

        predicted_list = []
        label_list = []
        for i in range(len(self.graph)):
            predicted_list.append(node_to_cluster[i])
            label_list.append(self.node_to_labels[i])

        rand_index = rand_score(label_list, predicted_list)

        return rand_index

    def __draw_clusters(self, clusters):
        pos = nx.spring_layout(self.graph)
        if len(clusters) > len(COLORS):
            colors = ALL_COLORS
        else:
            colors = COLORS
        i = 0
        for cluster in clusters:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=list(cluster), node_color=colors[i])
            i += 1

        nx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.5)
        plt.show()

    def evaluate_all(self):
        results_dict = {}
        for algorithm in self.clustering_algorithm_list:
            clusters, num_clusters, time_elapsed, rand_index = self.__evaluate(algorithm)
            results_dict[algorithm.__name__] = [clusters, num_clusters, time_elapsed]

        # results = {k: v for k, v in sorted(results_dict.items(), key=lambda item: item[2])}

        print(results_dict)
