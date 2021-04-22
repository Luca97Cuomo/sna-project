import utils
import clustering_algorithms
from time import sleep


def main():
    graph, node_to_labels = utils.load_graph("facebook_large\\musae_facebook_target.csv",
                                             "facebook_large\\musae_facebook_edges.csv")
    print(f"Ci sono {len(node_to_labels)} nodi nel dizionario\n")
    print(node_to_labels)
    print("\n")

    print(f"Ci sono {len(graph)} nodi nel grafo\n")
    print(graph)


if __name__ == "__main__":
    main()
