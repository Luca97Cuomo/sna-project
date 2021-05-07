import sys
sys.path.append("../")
import utils
from tqdm import tqdm

def degree_centrality(graph):
    node_to_degree = {}
    with tqdm(total=len(graph)) as pbar:
        for node in graph.nodes():
            node_to_degree[node] = graph.degree(node)
            pbar.update(1)
        return node_to_degree


def closeness_centrality(graph):
    node_to_closeness = {}
    with tqdm(total=len(graph)) as pbar:
        for node in graph.nodes():
            distances_from_node = utils.bfs(graph, node)

            node_to_closeness[node] = (len(graph.nodes) - 1) / (sum(distances_from_node.values()))
            pbar.update(1)

    return node_to_closeness


def betweenness_centrality(G):
    edge_btw = {frozenset(e): 0 for e in G.edges()}
    node_btw = {i: 0 for i in G.nodes()}
    with tqdm(total=len(G)) as pbar:
        for s in G.nodes():
            # Compute the number of shortest paths from s to every other node
            tree = []  # it lists the nodes in the order in which they are visited
            spnum = {i: 0 for i in G.nodes()}  # it saves the number of shortest paths from s to i
            parents = {i: [] for i in G.nodes()}  # it saves the parents of i in each of the shortest paths from s to i
            distance = {i: -1 for i in G.nodes()}  # it saves the distance of i from s
            eflow = {frozenset(e): 0 for e in G.edges()}  # the number of shortest paths starting from s that use the edge e
            vflow = {i: 1 for i in
                     G.nodes()}  # the number of shortest paths starting from s that use the vertex i. It is initialized to 1 because the shortest path from s to i is assumed to uses that vertex once.

            # BFS
            queue = [s]
            spnum[s] = 1
            distance[s] = 0
            while queue != []:
                c = queue.pop(0)
                tree.append(c)
                for i in G[c]:
                    if distance[i] == -1:  # if vertex i has not been visited
                        queue.append(i)
                        distance[i] = distance[c] + 1
                    if distance[i] == distance[c] + 1:  # if we have just found another shortest path from s to i
                        spnum[i] += spnum[c]
                        parents[i].append(c)

            # BOTTOM-UP PHASE
            while tree != []:
                c = tree.pop()
                for i in parents[c]:
                    eflow[frozenset({c, i})] += vflow[c] * (spnum[i] / spnum[
                        c])  # the number of shortest paths using vertex c is split among the edges towards its parents proportionally to the number of shortest paths that the parents contributes
                    vflow[i] += eflow[frozenset({c,
                                                 i})]  # each shortest path that use an edge (i,c) where i is closest to s than c must use also vertex i
                    edge_btw[frozenset({c, i})] += eflow[frozenset({c,
                                                                    i})]  # betweenness of an edge is the sum over all s of the number of shortest paths from s to other nodes using that edge
                if c != s:
                    node_btw[c] += vflow[
                        c]  # betweenness of a vertex is the sum over all s of the number of shortest paths from s to other nodes using that vertex
            pbar.update(1)
    return edge_btw, node_btw
