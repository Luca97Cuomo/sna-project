import networkx as nx
from itertools import combinations, groupby
import random
from pathlib import Path


def build_random_graph(num_of_nodes, probability_of_edge, seed=42):
    graph = nx.fast_gnp_random_graph(num_of_nodes, probability_of_edge, seed=seed)
    # clusters = generate_random_clusters(graph, num_of_clusters, num_of_nodes)
    return graph


def build_random_connected_graph(num_of_nodes, probability_of_edge, seed=42):
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

    # clusters = generate_random_clusters(graph, num_of_clusters, num_of_nodes)
    return graph


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


def bfs(graph, node):
    visited = set()
    visited.add(node)
    queue = [node]
    node_distances = {node: 0}

    while len(queue) > 0:
        current_node = queue.pop(0)
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                node_distances[neighbor] = node_distances[current_node] + 1

    return node_distances
