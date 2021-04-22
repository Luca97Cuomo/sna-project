from pathlib import Path
import networkx as nx


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

    return graph, node_to_label
