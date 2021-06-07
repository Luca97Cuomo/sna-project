from tqdm import tqdm
import sys
sys.path.append("../")

# Node betweenness
# The betweenness centrality of a vertex v is defined as the number of shortest paths that pass through the vertex.

# This is the algorithm implemented  during the lesson
def lecture_betweenness_centrality(G):
    edge_btw = {frozenset(e): 0 for e in G.edges()}
    node_btw = {i: 0 for i in G.nodes()}
    with tqdm(total=len(G)) as pbar:
        for s in G.nodes():
            # Compute the number of shortest paths from s to every other node
            tree = []  # it lists the nodes in the order in which they are visited
            spnum = {i: 0 for i in G.nodes()}  # it saves the number of shortest paths from s to i
            parents = {i: [] for i in G.nodes()}  # it saves the parents of i in each of the shortest paths from s to i
            distance = {i: -1 for i in G.nodes()}  # it saves the distance of i from s
            eflow = {frozenset(e): 0 for e in
                     G.edges()}  # the number of shortest paths starting from s that use the edge e
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


def naive_betweenness_centrality(graph):
    """
    Algorithm
    1) Define a dict that associates each node to the corresponding betweenness value
    2) Run the bfs algorithm for each pair of nodes and update the betweenness value
       of each node encountered for each shortest path. Be careful not to count the same path several times.
    3) Return the dict
    """

    node_to_rank = {}
    for node in graph.nodes():
        node_to_rank[node] = 0

    # I define a support list to more easily calculate the combinations (without repetition)
    node_list = []
    for node in graph.nodes():
        node_list.append(node)

    # compute bfs between each pair of nodes (without repetition)
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            source = node_list[i]
            dest = node_list[j]

            naive_betweenness_bfs(graph, source, dest, node_to_rank)

    return node_to_rank


def naive_betweenness_bfs(graph, source_node, dest_node, node_to_rank):
    """
    It assumes that exists at least one shortest path between source and dest
    """

    visited = {}
    fathers = {}
    for node in graph.nodes():
        visited[node] = False
        fathers[node] = []

    queue = [source_node]

    while len(queue) > 0:
        current_node = queue.pop(0)
        visited[current_node] = True

        if current_node == dest_node:
            # A shortest path has been found
        else:
            for neighbour in graph.neighbors(current_node):
                if not visited[neighbour]:
                    queue.append(neighbour)
                    fathers[neighbour].append(current_node)

    # update node to rank
    for father in fathers[dest_node]:
        # The rank for each node is equal to the number of ways it can be reached by his fathers
        # multiplied by the number of ways he can reach the destination node

