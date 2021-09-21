"""

Approximate inference using Belief Propagation
Here we can rely on some existing library,
for example https://github.com/mbforbes/py-factorgraph
Authors: lingxiao@cmu.edu
         kkorovin@cs.cmu.edu
"""

import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm
import pandas as pd

from inference.core import Inference
import networkx as nx
import matplotlib.pyplot as plt


def to_binary(np_array, work=True):
    if work:
        return (pd.DataFrame(np_array).apply(lambda x: x == x.max(), 1).astype(int).values - 0.5) * 2
    else:
        return np_array


class BeliefPropagation_single(Inference):
    """
    A special case implementation of BP
    for binary MRFs.
    Exact BP in tree structure only need two passes,
    LBP need multiple passes until convergene. 
    """

    def _safe_norm_exp(self, logit):
        logit -= np.max(logit, axis=1, keepdims=True)
        prob = np.exp(logit)
        prob /= prob.sum(axis=1, keepdims=True)
        return prob

    def _safe_divide(self, a, b):
        '''
        Divies a by b, then turns nans and infs into 0, so all division by 0
        becomes 0.
        '''
        c = a / b
        c[c == np.inf] = 0.0
        c = np.nan_to_num(c)
        return c

    def marginal_normalize(self, n_V, xi, graph, index_bases, use_log, neighbors, messages_psi, initialize=False):
        if not initialize:
            probs = np.zeros([n_V, 2])
            for i in range(n_V):
                probs[i] = graph.w[i] * xi
                if not use_log:
                    probs[i] = np.exp(probs[i])
                for j in neighbors[i]:
                    if use_log:
                        probs[i] += messages_psi[index_bases[j] + neighbors[j].index(i)]
                    else:
                        probs[i] *= messages_psi[index_bases[j] + neighbors[j].index(i)]
        else:
            probs = messages_psi[index_bases]
            if not use_log:
                probs = np.exp(probs)
        # normalize
        if self.mode in ['marginal', 'map']:
            if use_log:
                results = self._safe_norm_exp(probs.copy())
            else:
                results = self._safe_divide(probs.copy(), probs.copy().sum(axis=1, keepdims=True))
        return results

    def calculate_message(self, messages_psi, i, neighbor, use_log, update_result, neighbors
                          , index_bases, degrees):
        in_message_psi_prod = 0 if use_log else 1
        for j in neighbor:
            if use_log:
                in_message_psi_prod += messages_psi[index_bases[j] + neighbors[j].index(i)]
            else:
                in_message_psi_prod *= messages_psi[index_bases[j] + neighbors[j].index(i)]

        # messages_ = messages.copy()
        for k in range(degrees[i]):
            j = neighbor[k]
            if use_log:
                update_result[index_bases[i] + k] = in_message_psi_prod - \
                                                    (messages_psi[index_bases[j] + neighbors[j].index(i)])
            else:
                update_result[index_bases[i] + k] = self._safe_divide(in_message_psi_prod,
                                                                      messages_psi[
                                                                          index_bases[j] + neighbors[j].index(
                                                                              i)])
        return update_result

    def run_one(self, graph, use_log=True, use_binary=True, smooth=0, threshold=None):
        # Asynchronous BP  
        # Sketch of algorithm:
        # -------------------
        # preprocessing:
        # - sort nodes by number of edges
        # Algo:
        # - initialize messages to 1
        # - until convergence or max iters reached:
        #     - for each node in sorted list (fewest edges to most):
        #         - compute outgoing messages to neighbors
        #         - check convergence of messages

        # TODO: check more convergence conditions, like calibration
        if self.mode == "marginal":  # not using log
            sumOp = logsumexp if use_log else np.sum
        else:
            sumOp = np.max
        # storage, W should be symmetric 
        max_iters = 1000
        epsilon = 1e-20  # determines when to stop
        # add networks plots
        # G = nx.from_numpy_matrix(graph.J)
        # nx.draw(G, with_labels=True)
        # plt.show()

        row, col = np.where(graph.J)
        n_V, n_E = len(graph.w), len(row)
        # create index dict
        degrees = np.sum(graph.J != 0, axis=0)
        index_bases = np.zeros(n_V, dtype=np.int64)
        for i in range(1, n_V):
            index_bases[i] = index_bases[i - 1] + degrees[i - 1]

        neighbors = {i: [] for i in range(n_V)}
        for i, j in zip(row, col): neighbors[i].append(j)
        neighbors = {k: sorted(v) for k, v in neighbors.items()}
        # sort nodes by neighbor size 
        ordered_nodes = np.argsort(degrees)
        center_node = ordered_nodes[-1]

        # init messages based on graph structure (E, 2)
        # messages are ordered (out messages)
        # 这里是用伯努利来初始化belief的初始值
        messages_psi = np.ones([n_E, 2])
        # 修改：按照伯努利的概率给每个node赋值

        for i in ordered_nodes:
            for j in neighbors[i]:
                if threshold is not None:
                    x = np.zeros(len(graph.w))
                    x[graph.w == max(graph.w)] = 1
                    messages_psi[index_bases[j] + neighbors[j].index(i)] = [1 - threshold, threshold] if x[
                                                                                                             j] == 1 else [
                        threshold,
                        1 - threshold]
                else:
                    raw_psi = [np.exp(-graph.w[j]), np.exp(graph.w[j])]
                    messages_psi[index_bases[j] + neighbors[j].index(i)] = raw_psi / sum(raw_psi)

        # 添加对b的变化
        if use_log:
            messages_psi = np.log(messages_psi)  # log

        xij = np.array([[1, -1], [-1, 1]])
        xi = np.array([-1, 1])
        temp_results = self.marginal_normalize(n_V, xi, graph, index_bases, use_log, neighbors, messages_psi.copy(),
                                               initialize=True)
        belief_list = [to_binary(temp_results, work=use_binary)]
        messsages_psi_list = [to_binary(messages_psi.copy(), work=use_binary)]
        for iter in range(max_iters):
            # save old message for checking convergence
            old_messages_psi = messages_psi.copy()
            # update messages, here synchronous update
            update_result = np.zeros([n_E, 2])

            for i in ordered_nodes:
                # print("updating message at", i)
                neighbor = neighbors[i]
                Jij = graph.J[i][neighbor]  # vector
                wi = graph.w[i]  # scalar
                # print(Jij, bi)
                local_potential = Jij.reshape(-1, 1, 1) * xij + wi * xi.reshape(-1, 1)
                update_result = self.calculate_message(messages_psi, i, neighbor, use_log, update_result, neighbors
                                                       , index_bases, degrees)
                # update
                update_result[index_bases[i]:index_bases[i] + degrees[i]] = sumOp(
                    update_result[index_bases[i]:index_bases[i] + degrees[i]].reshape(degrees[i], 2,
                                                                                      1) + local_potential,
                    axis=1)

            messages_psi = update_result.copy()

            # plot_list.append([old_message_diff, potential_diff])
            # check convergence
            if use_log:
                error = (self._safe_norm_exp(messages_psi.copy()) - self._safe_norm_exp(old_messages_psi.copy())) ** 2
            else:
                error = (messages_psi.copy() - old_messages_psi.copy()) ** 2

            if len(error):
                error = error.mean()
            else:
                error = 0.

            if error < epsilon:
                # print(error)
                temp_results = self.marginal_normalize(n_V, xi, graph, index_bases, use_log, neighbors,
                                                       messages_psi.copy())
                belief_list.append(
                    to_binary(temp_results, work=use_binary))  # convert belief in each iteration into bool
                messsages_psi_list.append(to_binary(messages_psi.copy(), work=use_binary))
                break
            temp_results = self.marginal_normalize(n_V, xi, graph, index_bases, use_log, neighbors, messages_psi.copy())
            belief_list.append(
                to_binary(temp_results, work=use_binary))  # convert belief in each iteration into bool
            # else:
            #     probability_list.append(temp_results.reshape(-1, 1) * np.array([-1, 1]))
            messsages_psi_list.append(to_binary(messages_psi.copy(), work=use_binary))

        # add joint distribution probabilities
        message_index = np.concatenate([list(zip([key] * len(item), item)) for key, item in neighbors.items()])
        binary_matrix = np.zeros((n_V, n_V))
        for i, neighbor in neighbors.items():
            for n in neighbor:
                Jij = graph.J[i][n]  # here is scaler
                local_potential_2d = np.array([Jij]).reshape(-1, 1, 1) * xij + graph.w[i] * xi.reshape(-1, 1) + graph.w[
                    n] * xi.reshape(1, -1)
                message_comb = local_potential_2d + messages_psi[
                    (message_index[:, 1] == i) & (message_index[:, 0] != n)].sum(0).reshape(-1, 1) + messages_psi[
                                   (message_index[:, 1] == n) & (message_index[:, 0] != i)].sum(0).reshape(1, -1)
                message_comb_norm = np.exp(message_comb.squeeze() - message_comb.squeeze().min()) / np.exp(
                    message_comb.squeeze() - message_comb.squeeze().min()).sum()
                binary_matrix[i, n] = message_comb_norm.diagonal(0).sum() - np.rot90(message_comb_norm).diagonal(
                    0).sum()
        # if iter < 100 or iter == max_iters - 1:
        #     return None
        # plt.figure()
        # plt.plot(np.array(probability_list)[:, :, 1], marker='o')
        # plt.legend(range(n_V))
        # plt.title(['center point:', center_node, 'change nodes:', changed_nodes])
        # plt.show()
        # messages_indexes is message index for message table
        # if self.verbose: print("Is BP converged: {}".format(converged))

        # calculate marginal or map
        # results = self.marginal_normalize(n_V, xi, graph, index_bases, use_log, neighbors, messages)
        # return results
        return {'belief': belief_list, 'message_psi': messsages_psi_list, 'binary_matrix': binary_matrix,
                'node_index': message_index}

    def run(self, graphs, use_log=True, use_binary=True, verbose=False, threshold=None):
        self.verbose = verbose
        res = []
        graph_iterator = tqdm(graphs) if self.verbose else graphs
        for graph in graph_iterator:
            single_res = self.run_one(graph, use_log=use_log, use_binary=use_binary,
                                      threshold=threshold)
            res.append(single_res)
        return res


if __name__ == "__main__":
    bp = BeliefPropagation_single("marginal")
