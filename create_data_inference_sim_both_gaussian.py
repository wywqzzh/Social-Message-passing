"""
Data creation helpers and action.

For creating data labels, one can use exact or approximate inference algorithms,
    as well as scalable alternatives such as subgraph labeling and label propagation
    (see ./labeling/ directory for details).

If variable size range is supplied, each generated graph
    has randomly chosen size in range.

@authors: kkorovin@cs.cmu.edu

TODO:
* add random seeds
"""
import itertools
import os
import argparse
import numpy as np
import random
import pandas as pd
import pickle
from time import time
import matplotlib.pyplot as plt

from graphical_models import construct_binary_mrf, BinaryMRF
from inference import get_algorithm
# from labeling import LabelProp, LabelSG, LabelTree
from numpy.random import uniform
from graphical_models.data_gen import generate_struct_mask


def parse_dataset_args():
    parser = argparse.ArgumentParser()

    # crucial arguments
    parser.add_argument('--graph_struct', default="star", type=str,
                        help='type of graph structure, such as star or fc')
    parser.add_argument('--size_range', default="5_5", type=str,
                        help='range of sizes, in the form "10_20"')
    parser.add_argument('--n_nodes', default=4, type=int,
                        help='number of nodes of a graph')
    parser.add_argument('--fit_order', default="w_J", type=str,
                        help='order of fitting')
    parser.add_argument('--num', default=10000, type=int,
                        help='number of graphs to generate')
    parser.add_argument('--simulation_num', default=2, type=int,
                        help='number of graphs to generate in simulation stage')
    # manage unlabeled/labeled data
    parser.add_argument('--unlab_graphs_path', default='none',
                        type=str, help='whether to use previously created unlabeled graphs.\
                            If `none`, creates new graphs. \
                            If non-`none`, should be a path from base_data_dir')
    # should be used for train-test split
    parser.add_argument('--data_mode', default='train',
                        type=str, help='use train/val/test subdirectory of base_data_dir')

    parser.add_argument('--mode', default='marginal', type=str,
                        help='type of inference to perform')
    parser.add_argument('--algo', default='bp', type=str,
                        help='Algorithm to use for labeling. Can be exact/bp/mcmc,\
                        label_prop for label propagation, or label_sg for subgraph labeling')

    # no need to change the following arguments
    parser.add_argument('--base_data_dir', default='./graphical_models/datasets/',
                        type=str, help='directory to save a generated dataset')
    parser.add_argument('--verbose', default=False, type=bool,
                        help='whether to display dataset statistics')
    parser.add_argument('--input_map', default='network_learning_data/chosen_combinations_fMRI.mat', type=str,
                        help='path of the adjacency matrix')
    parser.add_argument('--input_data', default='network_learning_data/matlabdata_1_35.csv', type=str,
                        help='path of the experiment data')
    parser.add_argument('--simulation_mode', default='fit', type=str,
                        help='this experiment is simulation or fitting')
    parser.add_argument('--lr', default=1, type=float,
                        help='learning rate in fitting')
    parser.add_argument('--epoch', default=20, type=int,
                        help='learning epoches in fitting')
    parser.add_argument('--optimize', default='None', type=str,
                        help='adaptive learning rate method')
    return parser.parse_args()


# Helpers ---------------------------------------------------------------------
def save_graphs(graphs, labels, args):
    # unlabeled data, save to its temporary address
    if args.algo == 'none':
        path = os.path.join(args.base_data_dir, args.unlab_graphs_path)
        np.save(path + '.npy', graphs, allow_pickle=True)
    # otherwise the data is prepared and should be saved
    else:
        for graph, res in zip(graphs, labels):
            if args.mode == "marginal":
                res_marginal, res_map = res, None
            else:
                res_marginal, res_map = None, res

            directory = os.path.join(args.base_data_dir, args.data_mode,
                                     graph.struct, str(graph.n_nodes))
            os.makedirs(directory, exist_ok=True)
            data = {"W": graph.W, "b": graph.b,
                    "marginal": res_marginal, "map": res_map}
            # pprint(data)

            t = "_".join(str(time()).split("."))
            path_to_graph = os.path.join(directory, t)
            np.save(path_to_graph, data)


def load_graphs(path):
    graphs = np.load(path, allow_pickle=True)
    return graphs


def sample_marginal_prob(res, J):
    row_index, col_index = np.where(J)
    binary_matrix = np.zeros_like(J)
    for i in range(len(col_index)):
        if res.shape[0] == 1:
            cnt = pd.DataFrame(res[:, [row_index[i], col_index[i]]]).groupby([0, 1]).size().unstack(1).reindex(
                [-1, 1]).T.reindex(
                [-1, 1]).T.fillna(0).values
        else:
            cnt = pd.DataFrame(res[:, [row_index[i], col_index[i]]]).groupby([0, 1]).size().unstack(1).fillna(0).values
        prob_norm = cnt / cnt.sum()
        binary_matrix[row_index[i], col_index[i]] = prob_norm.diagonal(0).sum() - np.rot90(prob_norm).diagonal(
            0).sum()
    return binary_matrix


def sample_marginal_prob2(res, J):
    row_index, col_index = np.where(J)
    binary_matrix = np.zeros_like(J)
    for i in range(len(col_index)):
        binary_matrix[row_index[i], col_index[i]] = (res[:, row_index[i]] * res[:, col_index[i]]).mean()
    return binary_matrix


def people_gradient(last_v_J, J_people, people_to_position=None):
    row_index, col_index = np.where(last_v_J)
    for r, c in zip(row_index, col_index):
        if people_to_position is None:
            J_people[r] += last_v_J[r, c]
            J_people[c] += last_v_J[r, c]
        else:
            position_to_people = {value: key for key, value in people_to_position.items()}
            x, y = position_to_people[r], position_to_people[c]
            J_people[x] += last_v_J[r, c]
            J_people[y] += last_v_J[r, c]
    return J_people


def position_matrix_to_people_matrix(mask, people_to_position=None):
    if people_to_position:
        position_to_people = {value: key for key, value in people_to_position.items()}
        df_mask = pd.DataFrame(mask)
        df_mask.columns = pd.Series(df_mask.columns).replace(position_to_people).values
        df_mask.index = pd.Series(df_mask.index).replace(position_to_people).values
        return df_mask.sort_index().T.sort_index().T.values
    else:
        return mask


def calculate_log_likelihood_sim(w, J, target, mask):
    all_combs = np.array(list(itertools.product([-1, 1], repeat=target.shape[1])))  # based on people, 128 data
    l_total = []
    w_part_simulation = (all_combs @ w.reshape(-1, 1)).ravel()
    row_index, col_index = np.where(mask)
    for idx in range(target.shape[0]):
        w_part_actual = target[idx] @ w
        J_part_simulation = np.zeros(all_combs.shape[0])
        J_part_actual = 0
        for r, c in zip(row_index, col_index):
            J_part_simulation += J[r, c] * np.multiply(all_combs[:, r], all_combs[:, c])
            J_part_actual += J[r, c] * target[idx, r] * target[idx, c]
        Z = np.exp(w_part_simulation + J_part_simulation).sum()
        l_total.append([w_part_actual + J_part_actual - np.log(Z)])

    return np.ravel(l_total).mean()


def compute_cross_expectations(J, w):
    mu = np.linalg.inv(J) @ w.reshape(-1, 1)
    return mu.reshape(-1, 1) * mu.reshape(1, -1) + np.linalg.inv(J)


# Runner ----------------------------------------------------------------------
if __name__ == "__main__":
    # parse arguments and dataset name
    args = parse_dataset_args()
    # low, high = args.size_range.split("_")
    # size_range = np.arange(int(low), int(high) + 1)
    last_v_w = 0.1  # both this row and next row are initialization of gradient
    last_v_J = 0.1
    group_ID = 1
    fit_order = args.fit_order.split('_')
    total_iter = 0
    w_total = []
    J_total = []
    likeli_total = []

    origin_w = np.random.normal(0, 2, args.n_nodes)
    """Gaussian changes: change J matrix, and also make the diagonal mask as 1"""
    origin_J = np.random.uniform(2, 10, (args.n_nodes, args.n_nodes))
    origin_J = (origin_J + origin_J.T) / 2
    mask = generate_struct_mask(args.graph_struct, args.n_nodes, shuffle_nodes=False)
    for i in range(args.n_nodes):
        mask[i, i] = 1
    origin_J *= mask
    target = np.random.multivariate_normal((np.linalg.inv(origin_J) @ origin_w.reshape(-1, 1)).ravel(),
                                           np.linalg.inv(origin_J), size=args.num)
    # random initialize
    w = np.random.normal(0, 2, args.n_nodes)
    J = np.random.uniform(2, 10, (args.n_nodes, args.n_nodes))
    J = (J + J.T) / 2  # J matrix is symmetric
    if len(w_total) > 0:
        w = np.array(w_total)[-1, :]
    if len(J_total) > 0:
        J = np.array(J_total)[-1, :, :]
    while total_iter <= 30:
        for idx, fo in enumerate(fit_order):
            # construct graphical models
            # either new-data-generation or data labeling scenario
            for _ in range(args.epoch):
                # position: key is people, value is position (in J matrix)
                J *= mask
                if args.unlab_graphs_path == 'none' or args.algo == 'none':
                    # create new graphs
                    graphs = []
                    # W = np.ones((n_nodes, n_nodes))/10
                    for idx, graph_num in enumerate(range(args.simulation_num)):
                        # sample n_nodes from range
                        graphs.append(
                            construct_binary_mrf(args.graph_struct, args.n_nodes, shuffle_nodes=False, read_J=J.copy(),
                                                 w=w.copy()))
                else:  # both are non-None: need to load data and label it
                    path = os.path.join(args.base_data_dir, args.unlab_graphs_path)
                    graphs = load_graphs(path + '.npy')

                # label them using a chosen algorithm
                if args.algo in ['exact', 'bp', 'mcmc']:
                    algo_obj = get_algorithm(args.algo)(args.mode)
                    if args.simulation_mode == 'simulation':
                        list_of_res = algo_obj.run(graphs, verbose=args.verbose, use_binary=False)
                        break
                    else:
                        list_of_res = algo_obj.run(graphs, verbose=args.verbose, use_binary=False)

                        # random select a batch of data
                        lis = np.arange(args.num)
                        random.shuffle(lis)
                        batch = target[lis[:100]]

                        if fo == 'w':
                            # fit w
                            """Gaussian changes: pred_w is written as following"""
                            pred_w = np.linalg.inv(J) @ w.reshape(-1, 1)
                            w += args.lr * (batch.mean(0) - pred_w.ravel())
                        else:
                            # adaptive learning rate
                            """Gaussian changes: J is updated as following"""
                            last_v_J = np.array(
                                np.multiply(
                                    (-0.5) * args.lr * (sample_marginal_prob2(batch, J) - compute_cross_expectations(J, w)),
                                    mask))
                            # update people to position
                            J += last_v_J
                        if fo == 'w':
                            w_total.append(w.copy())
                        else:
                            J_total.append(J.copy())

        likeli_total.append(calculate_log_likelihood_sim(w.copy(), J.copy(), target, mask))
        # To Do: converge criterion is defined as below (can be altered)
        if len(likeli_total) > 2:
            if abs(likeli_total[-1] - np.mean(likeli_total[-10:])) < 1e-8:
                break
        total_iter += 1

    colors = ['red', 'blue', 'black', 'green']
    plt.figure()
    for i in range(args.n_nodes):
        plt.plot(np.array(w_total)[:, i], c=colors[i], label='w' + str(i))
        plt.plot([origin_w[i]] * args.epoch * (total_iter + 1), '--', c=colors[i])
    plt.legend()
    plt.title([round(i, 2) for i in origin_w])
    plt.xlabel('iterations')
    plt.ylabel('w')
    # plt.savefig('inference_pics/w_converge_sample_both_iter' + str(total_iter) + '.png')
    plt.show()

    plt.figure()
    for i in range(args.n_nodes - 1):
        plt.plot(np.array(J_total)[:, 0, i+1], c=colors[i], label='J' + str(0) + str(i))
        plt.plot([origin_J[0, i+1]] * args.epoch * (total_iter + 1), '--', c=colors[i])
    plt.legend()
    plt.title([round(origin_J[0, i+1], 2) for i in range(args.n_nodes - 1)])
    plt.xlabel('iterations')
    plt.ylabel('J')
    # plt.savefig('inference_pics/J_converge_sample_both_iter' + str(total_iter) + '.png')
    plt.show()

    plt.figure()
    plt.plot(likeli_total)
    plt.show()
