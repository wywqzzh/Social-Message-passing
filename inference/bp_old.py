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


class BeliefPropagation(Inference):
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

    def marginal_normalize(self, n_V, xi, graph, index_bases, use_log, neighbors, messages, initialize=False):
        if not initialize:
            probs = np.zeros([n_V, 2])
            for i in range(n_V):
                probs[i] = graph.b[i] * xi
                if not use_log:
                    probs[i] = np.exp(probs[i])
                for j in neighbors[i]:
                    if use_log:
                        probs[i] += messages[index_bases[j] + neighbors[j].index(i)]
                    else:
                        probs[i] *= messages[index_bases[j] + neighbors[j].index(i)]
        else:
            probs = messages[index_bases]
            if not use_log:
                probs = np.exp(probs)
        # normalize
        if self.mode in ['marginal', 'map']:
            if use_log:
                results = self._safe_norm_exp(probs.copy())
            else:
                results = self._safe_divide(probs.copy(), probs.copy().sum(axis=1, keepdims=True))

        # if self.mode == 'map':
        #     results = np.argmax(probs.copy(), axis=1)
        #     results[results == 0] = -1
        return results

    def run_one(self, graph, use_log=True, use_binary=True, smooth=0):
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
        G = nx.from_numpy_matrix(graph.W)
        nx.draw(G, with_labels=True)
        # plt.show()

        row, col = np.where(graph.W)
        n_V, n_E = len(graph.b), len(row)
        # create index dict
        degrees = np.sum(graph.W != 0, axis=0)
        index_bases = np.zeros(n_V, dtype=np.int64)
        for i in range(1, n_V):
            index_bases[i] = index_bases[i - 1] + degrees[i - 1]

        neighbors = {i: [] for i in range(n_V)}
        for i, j in zip(row, col): neighbors[i].append(j)
        neighbors = {k: sorted(v) for k, v in neighbors.items()}
        # sort nodes by neighbor size 
        ordered_nodes = np.argsort(degrees)
        center_node = ordered_nodes[-1]
        changed_nodes = [pd.Series(graph.W[center_node]).idxmin(), pd.Series(graph.W[center_node]).idxmax()]

        # init messages based on graph structure (E, 2)
        # messages are ordered (out messages)
        # 这里是用伯努利来初始化belief的初始值
        messages = np.ones([n_E, 2])
        # 修改：按照伯努利的概率给每个node赋值
        # threshold是变化的——一半2/7, 一半5/7
        threshold = np.random.choice([2 / 7, 5 / 7])
        node_beliefs = {i: int(np.random.random() <= threshold) for i in range(n_V)}
        for i in ordered_nodes:
            for j in neighbors[i]:
                messages[index_bases[j] + neighbors[j].index(i)] = [2 / 7, 5 / 7] if node_beliefs[j] == 1 else [5 / 7,
                                                                                                                2 / 7]
        # 添加对b的变化
        graph.b = [5 / 7 if node_beliefs[i] == 1 else 2 / 7 for i in range(n_V)]

        if use_log:
            messages = np.log(messages)  # log

        xij = np.array([[1, -1], [-1, 1]])
        xi = np.array([-1, 1])
        converged = False
        temp_results = self.marginal_normalize(n_V, xi, graph, index_bases, use_log, neighbors, messages.copy(),
                                               initialize=True)
        # if self.mode == 'marginal':
        probability_list = [to_binary(temp_results, work=use_binary)]
        # else:
        #     probability_list = [temp_results.reshape(-1, 1) * np.array([-1, 1])]
        messsages_list = [to_binary(messages.copy(), work=use_binary)]
        for iter in range(max_iters):
            # save old message for checking convergence
            old_messages = messages.copy()
            # update messages
            for i in ordered_nodes:
                # print("updating message at", i)
                neighbor = neighbors[i]
                # print(neighbor)
                # if use_log:
                #     Jij = np.log(graph.W[i][neighbor])
                # else:
                Jij = graph.W[i][neighbor]  # vector
                bi = graph.b[i]  # scalar
                # print(Jij, bi)
                local_potential = Jij.reshape(-1, 1, 1) * xij + bi * xi.reshape(-1, 1)
                # if not use_log:
                #     local_potential = np.exp(local_potential)
                # get in messages product (log)
                in_message_prod = 0 if use_log else 1
                for j in neighbor:
                    if use_log:
                        in_message_prod += messages[index_bases[j] + neighbors[j].index(i)]
                    else:
                        in_message_prod *= messages[index_bases[j] + neighbors[j].index(i)]

                # messages_ = messages.copy()
                for k in range(degrees[i]):
                    j = neighbor[k]
                    if use_log:
                        messages[index_bases[i] + k] = in_message_prod - \
                                                       (messages[index_bases[j] + neighbors[j].index(i)])
                    else:
                        messages[index_bases[i] + k] = self._safe_divide(in_message_prod,
                                                                         messages[
                                                                             index_bases[j] + neighbors[j].index(i)])
                        # update
                messages[index_bases[i]:index_bases[i] + degrees[i]] = sumOp(
                    messages[index_bases[i]:index_bases[i] + degrees[i]].reshape(degrees[i], 2, 1) + local_potential,
                    axis=1)
                # plot_list.append([old_message_diff, potential_diff])
            # check convergence
            if use_log:
                error = (self._safe_norm_exp(messages.copy()) - self._safe_norm_exp(old_messages.copy())) ** 2
            else:
                error = (messages.copy() - old_messages.copy()) ** 2

            if len(error):
                error = error.mean()
            else:
                error = 0.

            if error < epsilon:
                converged = True
                # print(error)
                temp_results = self.marginal_normalize(n_V, xi, graph, index_bases, use_log, neighbors, messages.copy())
                probability_list.append(
                    to_binary(temp_results, work=use_binary))  # convert belief in each iteration into bool
                # else:
                #     probability_list.append(temp_results.reshape(-1, 1) * np.array([-1, 1]))
                messsages_list.append(to_binary(messages.copy(), work=use_binary))
                break
            temp_results = self.marginal_normalize(n_V, xi, graph, index_bases, use_log, neighbors, messages.copy())
            probability_list.append(
                to_binary(temp_results, work=use_binary))  # convert belief in each iteration into bool
            # else:
            #     probability_list.append(temp_results.reshape(-1, 1) * np.array([-1, 1]))
            messsages_list.append(to_binary(messages.copy(), work=use_binary))
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
        return {'belief': probability_list, 'message': messsages_list,
                'node_index': np.concatenate([list(zip([key] * len(item), item)) for key, item in neighbors.items()])}

    def run(self, graphs, use_log=True, use_binary=True, verbose=False):
        self.verbose = verbose
        res = []
        graph_iterator = tqdm(graphs) if self.verbose else graphs
        for graph in graph_iterator:
            res.append(self.run_one(graph, use_log=use_log, use_binary=use_binary))
        return res


if __name__ == "__main__":
    bp = BeliefPropagation("marginal")
