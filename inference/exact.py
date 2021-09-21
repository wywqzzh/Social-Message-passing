"""

Exact inference
Authors: kkorovin@cs.cmu.edu

"""
import itertools
import numpy as np
from tqdm import tqdm
from inference.core import Inference


class ExactInference(Inference):
    """ Special case BinaryMRF implementation """
    def _safe_norm_exp(self, logit):
        logit -= np.max(logit, keepdims=True)
        prob = np.exp(logit)
        prob /= prob.sum(keepdims=True)
        return prob

    def compute_probs(self, W, b, n):
        log_potentials = np.zeros([2]*n)
        for state in itertools.product([0, 1], repeat=n):
            state_ind = np.array(state)
            state_val = 2 * state_ind - 1
            log_potentials[state] = state_val.dot(W.dot(state_val)) + b.dot(state_val)
        # probs = np.exp(log_potentials)
        # probs /= probs.sum()
        probs = self._safe_norm_exp(log_potentials)
        return probs

    def run_one(self, graph):
        W = graph.W
        b = graph.b
        n = graph.n_nodes

        # compute joint probabilities
        # array of shape [2,...,2]
        probs = self.compute_probs(W, b, n)
        # print("M1:", probs[0, :, :].sum(), probs[1, :, :].sum())
        # print("M2:", probs[:, 0, :].sum(), probs[:, 1, :].sum())

        if self.mode == "marginal":
            # select one state and compute marginal:
            marginals = np.zeros((n, 2))  # [i, 0] is P(x_i=0)
            for i in range(n):
                axes = tuple(j for j in range(n) if j != i)
                marginal = probs.sum(axis=axes)
                marginals[i] = marginal
            return marginals

        elif self.mode == "map":
            binary_ind = np.unravel_index(probs.argmax(),
                                          probs.shape)
            return 2 * np.array(binary_ind) - 1

    def run(self, graphs, verbose=False):
        self.verbose = verbose
        res = []
        graph_iterator = tqdm(graphs) if self.verbose else graphs
        for graph in graph_iterator:
            res.append(self.run_one(graph))
        return res
