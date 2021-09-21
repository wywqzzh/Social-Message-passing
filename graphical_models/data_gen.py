"""

Graphical model generators
Authors: kkorovin@cs.cmu.edu

"""

import numpy as np
import networkx as nx

from graphical_models.data_structs import BinaryMRF
import matplotlib.pyplot as plt

struct_names = ["star", "random_tree", "powerlaw_tree", "path",
                "cycle", "ladder", "grid",
                "circ_ladder", "barbell", "loll", "wheel",
                "bipart", "tripart", "fc"]


def show_graph_with_labels(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    return gr


def generate_struct_mask(struct, n_nodes, shuffle_nodes, adjacency_matrix=None):
    # a horrible collection of ifs due to args in nx constructors
    if adjacency_matrix is not None:
        return np.asmatrix(adjacency_matrix.astype(int))
    elif struct == "star":
        g = nx.star_graph(n_nodes)
    elif struct == "random_tree":
        g = nx.random_tree(n_nodes)
    elif struct == "powerlaw_tree":
        g = nx.random_powerlaw_tree(n_nodes, gamma=3, seed=None)
    elif struct == "binary_tree":
        raise NotImplementedError("Implement a binary tree.")
    elif struct == "path":
        g = nx.path_graph(n_nodes)
    elif struct == "cycle":
        g = nx.cycle_graph(n_nodes)
    elif struct == "ladder":
        g = nx.ladder_graph(n_nodes)
    elif struct == "grid":
        m = np.random.choice(range(1, n_nodes + 1))
        n = n_nodes // m
        g = nx.grid_2d_graph(m, n)
    elif struct == "circ_ladder":
        g = nx.circular_ladder_graph(n_nodes)
    elif struct == "barbell":
        assert n_nodes >= 4
        m = np.random.choice(range(2, n_nodes - 1))
        blocks = (m, n_nodes - m)
        g = nx.barbell_graph(*blocks)
    elif struct == "loll":
        assert n_nodes >= 2
        m = np.random.choice(range(2, n_nodes + 1))
        g = nx.lollipop_graph(m, n_nodes - m)
    elif struct == "wheel":
        g = nx.wheel_graph(n_nodes)
    elif struct == "bipart":
        m = np.random.choice(range(n_nodes))
        blocks = (m, n_nodes - m)
        g = nx.complete_multipartite_graph(*blocks)
    elif struct == "tripart":
        # allowed to be zero
        m, M = np.random.choice(range(n_nodes), size=2)
        if m > M:
            m, M = M, m
        blocks = (m, M - m, n_nodes - M)
        g = nx.complete_multipartite_graph(*blocks)
    elif struct == "fc":
        g = nx.complete_graph(n_nodes)
    else:
        raise NotImplementedError("Structure {} not implemented yet.".format(struct))

    if struct == "ladder":
        node_order = list(range(n_nodes * 2))
    else:
        node_order = list(range(n_nodes))

    if shuffle_nodes:
        np.random.shuffle(node_order)

    # a weird subclass by default; raises a deprecation warning
    # with a new update of networkx, this should be updated to
    # nx.convert_matrix.to_numpy_array
    np_arr_g = nx.to_numpy_matrix(g, nodelist=node_order)

    return np_arr_g.astype(int)


def construct_binary_mrf(struct, n_nodes, shuffle_nodes=True, read_J=None, adjacency_matrix=None, w=None):
    """Construct one binary MRF graphical model

    Arguments:
        struct {string} -- structure of the graph
        (on of "path", "ladder", ...)
        n_nodes {int} -- number of nodes in the graph
        shuffle_nodes {bool} -- whether to permute node labelings
                                uniformly at random
    Returns:
        BinaryMRF object
    """
    if read_J is None:
        # W = np.random.normal(0, 1, (n_nodes, n_nodes))
        # W = np.random.uniform(0, 1, (n_nodes, n_nodes))
        J = np.ones((n_nodes, n_nodes))
        J = (J + J.T) / 2
    else:
        J = read_J.copy()
    # w = np.ones(n_nodes)
    if w is None:
        w = np.random.normal(0, 1, n_nodes)
    # w = np.random.normal(0., 0.25, n_nodes)


    # add extreme value for W, in column/row connected to center
    # center_node = list((W != 0).sum(1)).index(n_nodes - 1)
    # rest_nodes = list(range(n_nodes))
    # rest_nodes.pop(center_node)
    # change_nodes = np.random.choice(rest_nodes, 2)
    # W[center_node, change_nodes[0]] = W[change_nodes[0], center_node] = W[change_nodes[0], center_node] * 10
    # W[center_node, change_nodes[1]] = W[change_nodes[1], center_node] = W[change_nodes[1], center_node] * (-10)

    return BinaryMRF(J, w, struct=struct)


if __name__ == "__main__":
    print("Testing all structures:")
    n_nodes = 5
    for struct in struct_names:
        print(struct, end=": ")
        graph = construct_binary_mrf(struct, n_nodes)
        print("ok")
        print(graph.W, graph.b)

    print("Nodes not shuffled:")
    graph = construct_binary_mrf("wheel", n_nodes, False)
    print(graph.W, graph.b)

    print("Nodes shuffled:")
    graph = construct_binary_mrf("wheel", 5)
    print(graph.W, graph.b)

    try:
        graph = construct_binary_mrf("fully_conn", 3)
    except NotImplementedError:
        pass
