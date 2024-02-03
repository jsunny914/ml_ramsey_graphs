import networkx as nx
import matplotlib.pyplot as plt

def print_graph(adj_mat):
    size = len(adj_mat)
    G = nx.complete_graph(size)
    for i in range(size):
        G.add_node(i)

    red_edges = []
    blue_edges = []
    blank_edges = []
    for i in range(size):
        for j in range(i, size):
            if adj_mat[i][j] == -1:
                G.add_edge(i, j, color='red', weight=5)
                red_edges.append((i,j))
            elif adj_mat[i][j] == 0:
                G.add_edge(i, j, color='gray', weight=5)
                if i != j:
                    blank_edges.append((i,j))
            elif adj_mat[i][j] == 1:
                G.add_edge(i, j, color='blue', weight=5)
                blue_edges.append((i,j))

    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='black', node_size=300)
    nx.draw_networkx_labels(G, pos, font_color='white')
    nx.draw_networkx_edges(G, pos, edgelist=blank_edges, edge_color='lightgray', width=3)
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='red', width=3)
    nx.draw_networkx_edges(G, pos, edgelist=blue_edges, edge_color='blue', width=3)
    
    plt.show(block=False)
    plt.savefig('graph.png')
    plt.pause(0.1)

def filled(adj_mat):
    size = len(adj_mat)
    for i in range(size):
        for j in range(i+1, size):
            if adj_mat[i][j] == 0:
                return False
    return True

def clique_3(adj_mat, new_edge, player):
    new_edge_v1 = new_edge[0]
    new_edge_v2 = new_edge[1]
    size = len(adj_mat)

    for i in range(size):
        if adj_mat[new_edge_v1][i] == player and adj_mat[new_edge_v2][i] == player:
            #print(new_edge_v1, new_edge_v2, i)
            return True
    return False

def clique_4(adj_mat, new_edge, player):
    new_edge_v1 = new_edge[0]
    new_edge_v2 = new_edge[1]
    size = len(adj_mat)

    possible_nodes = []
    for i in range(size):
        if adj_mat[new_edge_v1][i] == player and adj_mat[new_edge_v2][i] == player:
            possible_nodes.append(i)

    for i in range(len(possible_nodes)-1):
        for j in range(i+1, len(possible_nodes)):
            if adj_mat[possible_nodes[i]][possible_nodes[j]] == player:
                #print(new_edge_v1, new_edge_v2, possible_nodes[i], possible_nodes[j])
                return True

    return False