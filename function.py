import copy
import torch
import random
import argparse
import numpy as np
import networkx as nx
from inference import get_algorithm
from networkx.generators.classic import complete_graph
from networkx.linalg.graphmatrix import adjacency_matrix
from graphical_models import construct_binary_mrf
from graphical_models.data_gen import generate_struct_mask
import itertools


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def momentum(gradient, lr, gamma=0.9, last_v=1):
    return gamma * last_v + lr * gradient


def rmsprop(gradient, lr, cache, gamma=0.2, eps=1e-8):
    cache = gamma * cache + (1 - gamma) * (gradient ** 2)
    return cache, lr * gradient / (np.sqrt(cache) + eps)


def sample_marginal_prob2(res, J):
    row_index, col_index = np.where(J)
    binary_matrix = np.zeros_like(J)
    for j in range(len(res)):
        for i in range(len(col_index)):
            binary_matrix[row_index[i], col_index[i]] += (res[j][:, row_index[i]] * res[j][:, col_index[i]]).mean()
    for i in range(len(col_index)):
        binary_matrix[row_index[i], col_index[i]] = binary_matrix[row_index[i], col_index[i]] / len(res)
    return binary_matrix


def compare_graphs(graph_num):
    if graph_num == 1:
        a = complete_graph(5)
        b = complete_graph(5)
        c = complete_graph(5)

        d = nx.union(a, b, rename=('a-', 'b-'))
        d.add_edge('a-0', 'b-0')

        e = nx.union(d, c, rename=('', 'c-'))
        e.add_edge('a-1', 'c-0')
        e.add_edge('b-1', 'c-1')
        adj = adjacency_matrix(e).todense()
        node_order = list(e.nodes())
        for i in ['a', 'b', 'c']:
            adj[node_order.index(i + '-0'), node_order.index(i + '-1')] = 0
            adj[node_order.index(i + '-1'), node_order.index(i + '-0')] = 0

        e = nx.from_numpy_matrix(adj)
    elif graph_num == 3:
        G = nx.cycle_graph(15)
        adj_m = adjacency_matrix(G).todense()
        for i in range(15):
            cols = list(G.neighbors(i))
            adj_m[cols[0], cols[1]] = 1
            adj_m[cols[1], cols[0]] = 1
        e = nx.from_numpy_matrix(adj_m)
    elif graph_num == 2:
        a = complete_graph(4)
        a.add_node(4)
        a.add_edge(0, 4)
        a.add_edge(1, 4)

        b = complete_graph(4)
        b.add_node(4)
        b.add_edge(0, 4)
        b.add_edge(1, 4)

        c = complete_graph(4)
        c.add_node(4)
        c.add_edge(0, 4)
        c.add_edge(1, 4)

        d = nx.union(a, b, rename=('a-', 'b-'))
        d.add_edge('a-4', 'b-3')
        d.add_edge('a-4', 'b-2')

        e = nx.union(d, c, rename=('', 'c-'))
        e.add_edge('b-4', 'c-3')
        e.add_edge('b-4', 'c-2')
        e.add_edge('c-4', 'a-3')
        e.add_edge('c-4', 'a-2')
        adj = adjacency_matrix(e).todense()
        e = nx.from_numpy_matrix(adj)
    return e


def extended_star(neighbor_num):
    G = nx.Graph()
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_node(4)
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(0, 3)
    G.add_edge(0, 4)
    n = 5
    num = [int(i) for i in neighbor_num]
    for i in range(len(num)):
        for j in range(num[i]):
            G.add_node(n)
            G.add_edge(i + 1, n)
            n += 1
    adj = adjacency_matrix(G).todense()
    e = nx.from_numpy_matrix(adj)
    return e


# 获取四个节点的环形涂
def cycle():
    G = nx.Graph()
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(1, 3)
    G.add_edge(2, 3)
    adj = adjacency_matrix(G).todense()
    e = nx.from_numpy_matrix(adj)
    return e


def star_s():
    G = nx.Graph()
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_node(4)
    G.add_node(5)
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(0, 3)
    G.add_edge(0, 4)
    G.add_edge(4, 5)
    G.add_edge(3, 5)
    adj = adjacency_matrix(G).todense()
    e = nx.from_numpy_matrix(adj)
    return e


def wheel():
    G = nx.Graph()
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(0, 3)

    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 1)
    adj = adjacency_matrix(G).todense()
    e = nx.from_numpy_matrix(adj)
    return e


def fill_graph(graph, mask, node_idx):
    complete_to_sub_mapping = dict(zip(node_idx, range(graph.n_nodes)))
    complete_g = nx.from_numpy_matrix(mask)
    nodes_comb = list(itertools.combinations(node_idx, 2))
    for i, j in nodes_comb:
        shortest_paths = nx.all_shortest_paths(complete_g, i, j)
        for shortest_path in shortest_paths:
            if len(shortest_path[1:-1]) == 0:
                continue
            if len(set(node_idx) & set(shortest_path[1:-1])) == 0:
                graph.J[complete_to_sub_mapping[i], complete_to_sub_mapping[j]] = graph.J[graph.J != 0][0]
                graph.J[complete_to_sub_mapping[j], complete_to_sub_mapping[i]] = graph.J[graph.J != 0][0]
                break
    return graph


# 获取信号强度
def get_singal(args):
    if args.continues == 1:
        x = np.linspace(1, args.iteration_num, args.iteration_num)
        singal = (-np.sin(x * np.pi / args.iteration_num) + 1) / 2
        singal = np.tile(singal, 2 * args.block_num)
    else:
        # singal = np.repeat(args.signal_1, args.iteration_num * 2 * args.block_num)
        s1 = list(np.repeat(args.signal_1, args.iteration_num))
        s2 = list(np.repeat(1 - args.signal_1, args.iteration_num))
        singal = (s1 + s2) * int(args.block_num / 2)
    return singal


def get_signal_uniform(args):
    signal = []
    for i in range(args.block_num):
        temp_signal = np.random.uniform(-10, 10, 1)
        s1 = list(np.repeat(temp_signal, args.iteration_num))

        signal += s1
    return signal


# 初始化w
def initialize_w(log_prob, args, signal, ball):
    if args.graph_struct != "cycle" or args.continues == 1:
        w = np.repeat(-1, args.n_nodes) * log_prob
        # w = np.array(random.choices([-1, 1], [signal, 1-signal], k=args.n_nodes)) * log_prob
        w = np.random.choice([-1, 1], p=[signal, 1 - signal], size=args.n_nodes) * log_prob
    elif args.graph_struct == "cycle":
        if args.case == 1:
            w = np.array([1, 1, -1, -1]) * log_prob
        elif args.case == 2:
            w = np.array([1, -1, 1, -1]) * log_prob
        elif args.case == 3:
            w = np.array([-1, -1, -1, 1]) * log_prob
        elif args.case == 4:
            w = np.array([-1, -1, -1, -1]) * log_prob
        elif args.case == 5:
            w = np.array([1, 1, 1, 1]) * log_prob
    return w


def get_graph_struct(args, log_prob):
    if args.graph_struct == 'extended_star':
        G = extended_star(args.n_extended)
        mask = nx.to_numpy_matrix(G) * log_prob
    elif args.graph_struct == 'self_defined':
        G = compare_graphs(args.which_graph)
        mask = nx.to_numpy_matrix(G)
    elif args.graph_struct == 'wheel':
        G = wheel()
        mask = nx.to_numpy_matrix(G)
    elif args.graph_struct == 'star':
        mask = generate_struct_mask(args.graph_struct, args.n_nodes, shuffle_nodes=False)
    else:
        if args.n_nodes == 4:
            G = cycle()
            mask = nx.to_numpy_matrix(G)
        else:
            mask = generate_struct_mask(args.graph_struct, args.n_nodes, shuffle_nodes=False)
    return mask


# 获取子图
def get_graphs(args, w, J, mask):
    algo_obj = None
    graphs_all = construct_binary_mrf(args.graph_struct, args.n_nodes, shuffle_nodes=False, read_J=J.copy(),
                                      w=w.copy())
    if args.get_subgraph == 'single_node':
        nodes_list = np.arange(args.n_nodes).reshape(-1, 1).tolist()
        graphs = [graphs_all.get_subgraph_on_nodes(i) for i in nodes_list]
        for i in range(len(graphs)):
            graphs[i].adj = copy.deepcopy(graphs[i].J)
        list_of_res = {idx: ([np.exp(-g.w), np.exp(g.w)] / (np.exp(-g.w) + np.exp(g.w))).ravel() for idx, g in
                       enumerate(graphs)}
    if args.get_subgraph == 'neighbor_node':
        graphs = []
        nodes_list = []
        for i in range(args.n_nodes):
            cur_nodes = sorted([i] + np.where(mask[i, :])[1].tolist())
            nodes_list.append(cur_nodes)
            graphs.append(graphs_all.get_subgraph_on_nodes(cur_nodes))
            graphs[i].adj = copy.deepcopy(graphs[i].J)
        algo_obj = get_algorithm(args.algo)(args.mode)
        list_of_res = algo_obj.run(graphs, verbose=args.verbose, use_binary=False)
    if args.get_subgraph == 'interpolation_node':
        graphs = []
        nodes_list = []
        for i in range(args.n_nodes):
            cur_nodes = sorted([i] + np.where(mask[i, :])[1].tolist())
            nodes_list.append(cur_nodes)
            initial_graph = graphs_all.get_subgraph_on_nodes(cur_nodes)
            revised_graph = fill_graph(initial_graph, mask, cur_nodes)
            graphs.append(revised_graph)
        algo_obj = get_algorithm(args.algo)(args.mode)
        list_of_res = algo_obj.run(graphs, verbose=args.verbose, use_binary=False)

    return list_of_res, nodes_list, graphs, algo_obj


def assign_w(args, idx, nodes_list, graphs, observation_history, batch_history, log_prob):
    if args.continues == 0:
        for i in range(len(nodes_list[idx])):
            if nodes_list[idx][i] == idx:
                observation = observation_history[nodes_list[idx][i]][-1]
            else:
                observation = np.array(batch_history[idx][-1])[:, i].reshape(1, -1)
            graphs[idx].w[i] = observation * log_prob
    else:
        for i in range(len(nodes_list[idx])):
            if nodes_list[idx][i] == idx:
                observations = observation_history[nodes_list[idx][i]][
                               max(0, len(observation_history[nodes_list[idx][i]]) - args.w_T_self):]
            else:
                observations = np.array(batch_history[idx][max(0, len(batch_history[idx]) - args.w_T_other):])[:, :,
                               i].reshape(1, -1)
                observations = observations[0]
            temp_mean = np.array(observations).mean()
            if temp_mean == 1:
                temp_mean = (args.w_T_self - 0.5) / args.w_T_self if nodes_list[idx][i] == idx else (
                                                                                                            args.w_T_other - 0.5) / args.w_T_other
            elif temp_mean == -1:
                temp_mean = -(args.w_T_self - 0.5) / args.w_T_self if nodes_list[idx][i] == idx else -(
                        args.w_T_other - 0.5) / args.w_T_other
            log_prob_self = 1 / 2 * np.log((1 + temp_mean) / (1 - temp_mean))
            graphs[idx].w[i] = log_prob_self
    return graphs[idx].w


# def assign_w(args, idx, nodes_list, graphs, observation_history, batch_history):
#     for i in range(len(nodes_list[idx])):
#         if nodes_list[idx][i] == idx:
#             observations = observation_history[nodes_list[idx][i]][
#                            max(0, len(observation_history[nodes_list[idx][i]]) - args.w_T_self):]
#         else:
#             observations = np.array(batch_history[idx][max(0, len(batch_history[idx]) - args.w_T_other):])[:, :,
#                            i].reshape(1, -1)
#             observations = observations[0]
#         temp_mean = np.array(observations).mean()
#         if temp_mean == 1:
#             temp_mean = (args.w_T_self - 0.5) / args.w_T_self if nodes_list[idx][i] == idx else (
#                                                                                                         args.w_T_other - 0.5) / args.w_T_other
#         elif temp_mean == -1:
#             temp_mean = -(args.w_T_self - 0.5) / args.w_T_self if nodes_list[idx][i] == idx else -(
#                     args.w_T_other - 0.5) / args.w_T_other
#         log_prob_self = 1 / 2 * np.log((1 + temp_mean) / (1 - temp_mean))
#         graphs[idx].w[i] = log_prob_self
#     return graphs[idx].w

def fit_J(args, idx, graphs, batch_T, sub_graph_res, mask):
    gradient_front = []
    gradient = []
    if args.get_subgraph != 'single_node':
        if args.optimize == 'None':
            last_v_J = np.array(
                np.multiply(
                    args.lr_J * (sample_marginal_prob2(batch_T, graphs[idx].adj) - sub_graph_res[
                        'binary_matrix']),
                    mask))
            gradient_front.append(sample_marginal_prob2(batch_T, graphs[idx].adj))
            gradient.append(sample_marginal_prob2(batch_T, graphs[idx].adj) - sub_graph_res[
                'binary_matrix'])
            # update people to position
            graphs[idx].J += last_v_J
            # graphs[idx].J=np.max(0, graphs[idx].J)

            row_index, col_index = np.where(graphs[idx].J < 0)
            for i in range(len(row_index)):
                graphs[idx].J[row_index[i], col_index[i]] = 0
    return graphs[idx].J, gradient_front, gradient


def get_finall_result(list_of_res_history, nodes_list, graphs_list, gradient_front, gradient, args, simulated_history):
    history_choices = []
    binary_choices = []
    # list_of_res_history: 4 iterations, 7 nodes, 'belief', converge path
    for i in range(len(simulated_history)):
        binary_choices.append(list(simulated_history[i]))
    for curr in list_of_res_history:  # here to output choice (-1 or 1)
        if args.get_subgraph == 'single_node':
            choice = np.array(list(curr.values()))[:, 1]
        else:
            choice = [node['belief'][-1][np.equal(nodes_list[idx], idx), :].ravel()[1] for idx, node in
                      enumerate(curr)]
        history_choices.append(choice)

    result_dict = {'history_choices': history_choices, "binary_choices": binary_choices, 'nodes_list': nodes_list,
                   'graphs_list': graphs_list,
                   "graph_struct": args.graph_struct + str(args.which_graph), "iteration_num": args.iteration_num,
                   "block_num": args.block_num,
                   "gradient_front": gradient_front, "gradient": gradient}
    return result_dict


def parse_dataset_args():
    parser = argparse.ArgumentParser()
    # crucial arguments
    parser.add_argument('--size_range', default="5_5", type=str,
                        help='range of sizes, in the form "10_20"')
    parser.add_argument('--verbose', default=False, type=str2bool,
                        help='whether to display dataset statistics')
    parser.add_argument('--algo', default='bp', type=str,
                        help='Algorithm to use for labeling. Can be exact/bp/mcmc,\
                                 label_prop for label propagation, or label_sg for subgraph labeling')
    parser.add_argument('--add_initial', default=False, type=str2bool,
                        help='if add initial guess')

    parser.add_argument('--output_format', default='pro', type=str,
                        help='if output format is choice, then results are 0/1;'
                             'if output format is prob, then results are probabilities')

    # 图结构与模型结构
    parser.add_argument('--graph_struct', default="star", type=str,
                        help='type of graph structure, such as star or fc')
    parser.add_argument('--n_extended', default="1234", type=str,
                        help='拓展星形每个点的节点个数')
    parser.add_argument('--n_nodes', default=4, type=int,
                        help='number of nodes of a graph')
    parser.add_argument('--get_subgraph', default='neighbor_node', type=str,
                        help='indicate the method to get the subgraph of the whole graph')
    parser.add_argument('--which_graph', default=2, type=int,
                        help='when testing with 3 graphs having all nodes with same degree, this parameter works')
    parser.add_argument('--case', default=1, type=int,
                        help='when the graph struct is cycle,the choice of init')

    # 方法参数
    parser.add_argument('--simulation_mode', default='simulation', type=str,
                        help='this experiment is simulation or fitting')
    parser.add_argument('--mode', default='marginal', type=str,
                        help='type of inference to perform')
    parser.add_argument('--optimize', default='None', type=str,
                        help='choose optimization method from rmsprop, momentum and None')
    # deterministic stochastic
    parser.add_argument('--policy', default="stochastic", type=str,
                        help='the type of policy')

    # 参数初始化
    parser.add_argument('--lr_w', default=0.1, type=float,
                        help='choose learning rate for w')
    parser.add_argument('--lr_J', default=0.1, type=float,
                        help='choose learning rate for J')
    parser.add_argument('--J_times', default=0.1, type=float,
                        help='multiplier for J')
    parser.add_argument('--inverse_tau', default=10, type=int,
                        help='softmax中每个belief的共同系数')
    parser.add_argument('--signal_1', default=8 / 10, type=float,
                        help='softmax中每个belief的共同系数')
    # 每个block的长度与block数
    parser.add_argument('--iteration_num', default=5, type=int,
                        help='number of iterations')
    parser.add_argument('--block_num', default=100, type=int,
                        help='block number')

    # 控制参数
    parser.add_argument('--continues', default=0, type=int,
                        help='if the singal is cintinue')
    parser.add_argument('--J_T', default=1, type=int,
                        help='每次使用T个时间长度的数据进行参数J更新')
    parser.add_argument('--w_T_self', default=5, type=int,
                        help='每次使用T个时间长度的观测数据进行自身w参数assign')
    parser.add_argument('--w_T_other', default=5, type=int,
                        help='每次使用T个时间长度的数据进行邻居w参数assign')
    parser.add_argument('--learn_self_w', default=False, type=bool,
                        help='是否学习自己的w')
    parser.add_argument('--self_batch_simulated', default=False, type=bool,
                        help='自己的batch是否为上一回合的猜测')
    parser.add_argument('--skip_observe_t', default=1, type=int,
                        help='间隔多久得到一次观测')

    return parser.parse_args()
