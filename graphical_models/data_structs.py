"""

Graphical model class
Authors: kkorovin@cs.cmu.edu

TODO:
* MST generation in BinaryMRF
"""

import networkx as nx
import numpy as np
from inference import get_algorithm


dflt_algo = {"marginal": "bp", "map": "bp"}


class GraphicalModel:
    def __init__(self, n_nodes, params=None, default_algo=dflt_algo):
        """Constructor

        Arguments:
            n_nodes {int} - number of vertices in graphical model
            params {dictionary<str,np.array> or None} -- parameters of the model

        Keyword Arguments:
            default_algo {dict} -- default inference methods to use,
            unless they are overriden in corresponding methods
            (default: {dflt_algo})
        """
        self.algo_marginal = default_algo["marginal"]
        self.algo_map = default_algo["map"]

    def set_ground_truth(self, marginal_est=None, map_est=None):
        """ Setting labels:
        To be used when instantiating
        a model from saved parameters
        """
        self.marginal = marginal_est
        self.map = map_est

    # Running inference with/without Inference object
    def get_marginals(self, algo_obj=None, algo=None):
        if algo_obj is None:
            if algo is None:
                algo = self.algo_marginal
            algo_obj = get_algorithm(algo)
        inf_res = algo_obj.run(self, mode="marginal")
        return inf_res

    def get_map(self, algo_obj=None, algo=None):
        if algo_obj is None:
            if algo is None:
                algo = self.algo_map
            algo_obj = get_algorithm(algo)
        inf_res = algo_obj.run(self, mode="map")
        return inf_res

    def __repr__(self):
        return "GraphicalModel:{} on {} nodes".format(
            self.__class__.__name__, self.n_nodes)


class BinaryMRF(GraphicalModel):
    def __init__(self, J, w, struct=None):
        """Constructor of BinaryMRF class.

        Arguments:
            J {np.array} -- (N, N) matrix of pairwise parameters
            w {np.array} -- (N,) vector of unary parameters

        Keyword Arguments:
            struct {string or None} -- description of graph structure
                                       (default: {None})
        """
        self.J = J
        self.w = w
        self.struct = struct
        self.n_nodes = len(J)
        self.default_algo = {"marginal": "bp",
                             "map": "bp"}
        # params = {"W": W, "b": b}
        super(BinaryMRF, self).__init__(
            n_nodes=self.n_nodes,
            default_algo=self.default_algo)

    def get_subgraph_on_nodes(self, node_list):
        """ node_list does not need to be ordered,
            return in the same order
        """
        nx_graph = nx.from_numpy_matrix(self.J)
        sg = nx_graph.subgraph(node_list)
        J_sg = np.array(nx.to_numpy_matrix(sg))
        w_sg = self.w[node_list]  # in the same order
        return BinaryMRF(J_sg, w_sg)

    def get_max_abs_spanning_tree(self):
        nx_graph = nx.from_numpy_matrix(np.abs(self.J))
        tree = nx.minimum_spanning_tree(nx_graph)
        J_abs_tree = np.array(nx.to_numpy_matrix(tree))
        J_mask = np.where(J_abs_tree > 0, 1, 0)
        # zero out unused edges:
        J_tree = J_mask * self.J
        w_tree = self.w
        return BinaryMRF(J_tree, w_tree)


if __name__ == "__main__":
    print(get_algorithm("bp"))
