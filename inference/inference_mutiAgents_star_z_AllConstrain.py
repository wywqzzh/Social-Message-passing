import copy
import torch
import random
import pickle
import pandas as pd
from function import *
from collections import namedtuple
import matplotlib.pyplot as plt
from gmm.gmmFitHtf import *

gaussian = namedtuple('Gaussian', ['mean', 'cov'])
if __name__ == "__main__":

    # np.random.seed(0)
    args = parse_dataset_args()
    singal = get_signal_uniform(args)  # 获取信号强度
    log_prob = 1 / 2 * np.log((1 - 0.4) / 0.4)  # preset probability when blue ball is picked
    # 确定图参数
    w = initialize_w(log_prob, args, 1 - 0.4, np.random.choice([1, -1], p=[0.4, 1 - 0.4]))
    J = np.ones((args.n_nodes, args.n_nodes)) * args.J_times
    mask = get_graph_struct(args, log_prob)  # 获取图结构
    neighbors = dict(pd.DataFrame(np.where(mask)).T.groupby(0).apply(lambda x: x[1].values.tolist()))
    J *= mask
    # 获取子图信息
    list_of_res_history = []
    list_of_res, nodes_list, graphs, algo_obj = get_graphs(args, w, J, mask)
    list_of_res_history.append(list_of_res.copy())
    graphs_list = [copy.deepcopy(graphs)]

    D = 1
    initial_mu = initial_action = 5
    alpha_sigma = 0.1
    alpha_Lambda = 0.1
    process_noise = 0  # 真实因变量的noise,非常小
    measurement_noise = [11, 3, 4, 8, 10]  # observation的noise不同人有不同的值
    process_noise_cov = process_noise ** 2

    T = len(singal)
    z_list = singal + np.random.normal(0, process_noise, T)

    x = np.zeros((args.n_nodes, T))
    action_Complete = np.zeros((args.n_nodes, T))  # action
    action_Sparse = np.zeros((args.n_nodes, T))  # action
    action_Independent = np.zeros((args.n_nodes, T))  # action
    action_Identical = np.zeros((args.n_nodes, T))  # action

    action_ = np.zeros((args.n_nodes, T))  # action independent
    mean_ = np.zeros((args.n_nodes, T))  # estimate (posterior mean)
    cov_ = np.zeros((args.n_nodes, T))  # estimate (posterior cov)

    posteriors = [i for i in range(args.n_nodes)]

    Lambda_Complete = [[] for i in range(args.n_nodes)]
    Lambda_Sparse = [[] for i in range(args.n_nodes)]
    Lambda_Independent = [[] for i in range(args.n_nodes)]

    measurement_noise_cov = [[] for i in range(args.n_nodes)]
    w_Complete = [[] for i in range(args.n_nodes)]
    w_Sparse = [[] for i in range(args.n_nodes)]
    w_Independent = [[] for i in range(args.n_nodes)]
    w_Identical = [[] for i in range(args.n_nodes)]
    for id_time in range(T):

        for idx, sub_graph_res in enumerate(list_of_res):
            if id_time % args.iteration_num == 0:
                if id_time == 0:
                    w_Complete[idx].append(np.array([1] + [0] * (len(nodes_list[idx]) - 1)))
                    w_Sparse[idx].append(np.array([1] + [0] * (len(nodes_list[idx]) - 1)))
                    w_Independent[idx].append(np.array([1] + [0] * (len(nodes_list[idx]) - 1)))
                    w_Identical[idx].append(np.array([1] + [0] * (len(nodes_list[idx]) - 1)))
                    measurement_noise_cov[idx].append(2.0 ** 2)
                else:
                    pre = int((id_time / args.iteration_num - 1) * args.iteration_num)
                    temp_sigma = x[idx, pre:id_time]

                    L = copy.deepcopy(nodes_list[idx])
                    index = [nodes_list[idx].index(idx)]
                    for i in range(len(nodes_list[idx])):
                        if nodes_list[idx][i] != idx:
                            index.append(i)

                    temp_a_Compete = [action_[idx, pre + 1:id_time]]
                    temp_a_Sparse = [action_[idx, pre + 1:id_time]]
                    temp_a_Independent = [action_[idx, pre + 1:id_time]]
                    for i in range(len(nodes_list[idx])):
                        if nodes_list[idx][i] != idx:
                            temp_a_Compete.append(action_Complete[nodes_list[idx][i], pre:id_time - 1])
                            temp_a_Sparse.append(action_Sparse[nodes_list[idx][i], pre:id_time - 1])
                            temp_a_Independent.append(action_Independent[nodes_list[idx][i], pre:id_time - 1])

                    temp_a_Compete = np.array(temp_a_Compete)
                    temp_a_Sparse = np.array(temp_a_Sparse)
                    temp_a_Independent = np.array(temp_a_Independent)

                    S_Compete = np.cov(temp_a_Compete)
                    S_Sparse = np.cov(temp_a_Sparse)
                    S_Independent = np.cov(temp_a_Independent)

                    G_Complete = np.ones((len(nodes_list[idx]), len(nodes_list[idx])))
                    G_Sparse = np.ones((len(nodes_list[idx]), len(nodes_list[idx])))
                    G_Independent = np.zeros((len(nodes_list[idx]), len(nodes_list[idx])))
                    G_Sparse[graphs[idx].J == 0] = 0
                    G_Sparse = G_Sparse[index, :][:, index]

                    temp_Lambda_Complete, iter = ggmFitHtf(S_Compete, G_Complete, 30)
                    temp_Lambda_Sparse, iter = ggmFitHtf(S_Sparse, G_Sparse, 30)
                    temp_Lambda_Independent, iter = ggmFitHtf(S_Independent, G_Independent, 30)
                    if id_time / args.iteration_num < 2:
                        measurement_noise_cov[idx].append(np.var(temp_sigma))
                        Lambda_Complete[idx].append(temp_Lambda_Complete)
                        Lambda_Sparse[idx].append(temp_Lambda_Sparse)
                        Lambda_Independent[idx].append(temp_Lambda_Independent)
                    else:
                        measurement_noise_cov[idx].append((1 - alpha_sigma) * measurement_noise_cov[
                            idx][-1] + alpha_sigma * np.var(temp_sigma))

                        Lambda_Complete[idx].append(
                            (1 - alpha_Lambda) * Lambda_Complete[idx][-1] + alpha_Lambda * temp_Lambda_Complete)

                        Lambda_Sparse[idx].append(
                            (1 - alpha_Lambda) * Lambda_Sparse[idx][-1] + alpha_Lambda * temp_Lambda_Sparse)
                        Lambda_Independent[idx].append(
                            (1 - alpha_Lambda) * Lambda_Independent[idx][-1] + alpha_Lambda * temp_Lambda_Independent)

                    neighbor_num = len(nodes_list[idx])
                    temp_w_Complete = np.matmul(Lambda_Complete[idx][-1],
                                                np.ones(neighbor_num).reshape(-1, 1)) / np.matmul(
                        np.ones(neighbor_num).reshape(1, -1),
                        np.matmul(Lambda_Complete[idx][-1], np.ones(neighbor_num).reshape(-1, 1))).reshape(1, -1)
                    temp_w_Sparse = np.matmul(Lambda_Sparse[idx][-1],
                                              np.ones(neighbor_num).reshape(-1, 1)) / np.matmul(
                        np.ones(neighbor_num).reshape(1, -1),
                        np.matmul(Lambda_Sparse[idx][-1], np.ones(neighbor_num).reshape(-1, 1))).reshape(1, -1)
                    temp_w_Independent = np.matmul(Lambda_Independent[idx][-1],
                                                   np.ones(neighbor_num).reshape(-1, 1)) / np.matmul(
                        np.ones(neighbor_num).reshape(1, -1),
                        np.matmul(Lambda_Independent[idx][-1], np.ones(neighbor_num).reshape(-1, 1))).reshape(1, -1)

                    temp_w_Complete = np.squeeze(temp_w_Complete, axis=1)
                    temp_w_Sparse = np.squeeze(temp_w_Sparse, axis=1)
                    temp_w_Independent = np.squeeze(temp_w_Independent, axis=1)

                    w_Complete[idx].append(temp_w_Complete)
                    w_Sparse[idx].append(temp_w_Sparse)
                    w_Independent[idx].append(temp_w_Independent)
                    w_Identical[idx].append(np.array([1.0] * (len(nodes_list[idx]))) / (len(nodes_list[idx])))
                posteriors[idx] = copy.deepcopy(gaussian(initial_mu, measurement_noise_cov[idx][-1]))
                action_[idx, id_time] = action_Complete[idx, id_time] \
                    = action_Sparse[idx, id_time] = action_Independent[idx, id_time] \
                    = action_Identical[idx, id_time] = initial_action
                mean_[idx, id_time] = initial_mu
                cov_[idx, id_time] = measurement_noise_cov[idx][-1]
            x[idx, id_time] = z_list[id_time] + np.random.normal(0, measurement_noise[idx], 1)
            if id_time % args.iteration_num == 0:
                posteriors[idx] = gaussian(x[idx, id_time], measurement_noise_cov[idx][-1])
            # Kalman filter
            posterior = posteriors[idx]
            k = (posterior.cov + process_noise_cov) / (
                    posterior.cov + process_noise_cov + measurement_noise_cov[idx][-1])

            mean_[idx, id_time] = posterior.mean + k * (
                    x[idx, id_time] - posterior.mean)
            cov_[idx, id_time] = (1 - k) * posterior.cov
            action_[idx, id_time] = mean_[idx, id_time]
            posteriors[idx] = gaussian(mean_[idx, id_time], cov_[idx, id_time])

            # social
            # w = np.zeros(len(nodes_list[idx])) + 1 / (len(nodes_list[idx]))
            if id_time % args.iteration_num != 0:
                temp_a_Compete = [action_[idx, id_time]]
                temp_a_Sparse = [action_[idx, id_time]]
                temp_a_Independent = [action_[idx, id_time]]
                temp_a_Identical = [action_[idx, id_time]]
                for i in range(len(nodes_list[idx])):
                    if nodes_list[idx][i] != idx:
                        temp_a_Compete.append(action_Complete[nodes_list[idx][i], id_time - 1])
                        temp_a_Sparse.append(action_Sparse[nodes_list[idx][i], id_time - 1])
                        temp_a_Independent.append(action_Independent[nodes_list[idx][i], id_time - 1])
                        temp_a_Identical.append(action_Identical[nodes_list[idx][i], id_time - 1])

                action_Complete[idx, id_time] = np.dot(w_Complete[idx][-1], np.array(temp_a_Compete))
                action_Sparse[idx, id_time] = np.dot(w_Sparse[idx][-1], np.array(temp_a_Sparse))
                action_Independent[idx, id_time] = np.dot(w_Independent[idx][-1], np.array(temp_a_Independent))
                action_Identical[idx, id_time] = np.dot(w_Identical[idx][-1], np.array(temp_a_Identical))

            else:
                temp_a = [action_[idx, id_time]] + [0] * (len(nodes_list[idx]) - 1)
                action_Complete[idx, id_time] = np.dot(w_Complete[idx][-1], np.array(temp_a))
                action_Sparse[idx, id_time] = np.dot(w_Sparse[idx][-1], np.array(temp_a))
                action_Independent[idx, id_time] = np.dot(w_Independent[idx][-1], np.array(temp_a))
                action_Identical[idx, id_time] = np.dot(w_Identical[idx][-1], np.array(temp_a))

            # if idx == 0:
            #     print(temp_a, w[idx][-1])

            # if action[idx, id_time] > 10:
            #     print(np.array(temp_a).reshape(1, -1).squeeze())
    # print(measurement_noise_cov)
    result_dict = {"z": z_list, "x": x, "mean": mean_, "cov": cov_, "simga": measurement_noise_cov,

                   "Lambda_Complete": Lambda_Complete, "Lambda_Sparse": Lambda_Sparse,
                   "Lambda_Independent": Lambda_Independent,

                   "action_Complete": action_Complete, "action_Sparse": action_Sparse, "action_": action_,
                   "action_Independent": action_Independent, "action_Identical": action_Identical,

                   "w_Complete": w_Complete, "w_Sparse": w_Sparse, "w_Independent": w_Independent,
                   "w_Identical": w_Identical,

                   "iteration_num": args.iteration_num, "nodes_list": nodes_list, "true_sigma": measurement_noise}
    with open("data_constrain/neighbor_star2(" + str(args.iteration_num) + "_" + str(
            args.block_num) + ").pkl", "wb") as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # fig = plt.figure(figsize=(12, 15), dpi=300)
    # for idx, sub_graph_res in enumerate(list_of_res):
    #     plt.subplot(5, 1, idx + 1)
    #     plt.plot(z_list, "green")
    #     # plt.scatter(range(T), x[idx, :], color="red")
    #     plt.plot(mean_[idx, :], "black", linewidth=3)
    #     plt.plot(action[idx, :], "orange")
    #     plt.legend(["true z", r"mean of $p(z_t|x^{1:t})$", "action"])
    #     plt.title("agent " + str(idx + 1))
    #     # plt.savefig("imgs/agent"+str(idx+1)+".pdf")
    #     # plt.show()
    # plt.savefig("imgs/neighbor.pdf")
    # plt.show()
    # fig = plt.figure(figsize=(12, 15), dpi=300)
    # for idx, sub_graph_res in enumerate(list_of_res):
    #     plt.subplot(5, 1, idx + 1)
    #     plt.plot(z_list, "green")
    #     # plt.scatter(range(T), x[idx, :], color="red")
    #     plt.plot(mean_[idx, :], "black", linewidth=3)
    #     plt.plot(action2[idx, :], "orange")
    #     plt.legend(["true z", r"mean of $p(z_t|x^{1:t})$", "action"])
    #     plt.title("agent " + str(idx + 1))
    #     # plt.savefig("imgs/agent"+str(idx+1)+".pdf")
    #     # plt.show()
    # # plt.savefig("imgs/neighbor.pdf")
    # plt.show()
    # fig = plt.figure(figsize=(12, 15), dpi=300)
    # for idx, sub_graph_res in enumerate(list_of_res):
    #     plt.subplot(5, 1, idx + 1)
    #     plt.plot(z_list, "green")
    #     # plt.scatter(range(T), x[idx, :], color="red")
    #     plt.plot(mean_[idx, :], "black", linewidth=3)
    #     plt.plot(action3[idx, :], "orange")
    #     plt.legend(["true z", r"mean of $p(z_t|x^{1:t})$", "action"])
    #     plt.title("agent " + str(idx + 1))
    #     # plt.savefig("imgs/agent"+str(idx+1)+".pdf")
    #     # plt.show()
    # # plt.savefig("imgs/neighbor.pdf")
    # plt.show()
