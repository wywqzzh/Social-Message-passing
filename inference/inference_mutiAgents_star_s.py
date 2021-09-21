import copy
import torch
import random
import pickle
import pandas as pd
from function import *
from collections import namedtuple
import matplotlib.pyplot as plt

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
    initial_mu = initial_action = 0
    alpha_sigma = 0.1
    alpha_gamma = 0.1
    process_noise = 0  # 真实因变量的noise,非常小
    measurement_noise = [3, 5, 5, 6, 6, 1]  # observation的noise不同人有不同的值
    process_noise_cov = process_noise ** 2

    T = len(singal)
    z_list = singal + np.random.normal(0, process_noise, T)

    x = np.zeros((args.n_nodes, T))
    action = np.zeros((args.n_nodes, T))  # action
    action_ = np.zeros((args.n_nodes, T))  # action independent
    mean_ = np.zeros((args.n_nodes, T))  # estimate (posterior mean)
    cov_ = np.zeros((args.n_nodes, T))  # estimate (posterior cov)
    posteriors = [i for i in range(args.n_nodes)]
    Lambda = [[] for i in range(args.n_nodes)]
    Gamma = [[] for i in range(args.n_nodes)]
    measurement_noise_cov = [[] for i in range(args.n_nodes)]
    w = [[] for i in range(args.n_nodes)]

    for id_time in range(T):

        for idx, sub_graph_res in enumerate(list_of_res):
            if id_time % args.iteration_num == 0:
                if id_time == 0:
                    w[idx].append(np.array([1] + [0] * (len(nodes_list[idx]) - 1)))
                    measurement_noise_cov[idx].append(2.0 ** 2)
                else:
                    pre = int((id_time / args.iteration_num - 1) * args.iteration_num)
                    temp_sigma = x[idx, pre:id_time] - action[idx, pre:id_time]
                    if idx == 1:
                        print(np.sqrt(np.var(temp_sigma)))
                    temp_Gamma = [action_[idx, pre + 1:id_time]]
                    for i in range(len(nodes_list[idx])):
                        if nodes_list[idx][i] != idx:
                            temp_Gamma.append(action[nodes_list[idx][i], pre:id_time - 1])
                    temp_Gamma = np.array(temp_Gamma)
                    if id_time / args.iteration_num < 2:
                        measurement_noise_cov[idx].append(np.var(temp_sigma))
                        Gamma[idx] = np.cov(temp_Gamma)
                    else:
                        measurement_noise_cov[idx].append((1 - alpha_sigma) * measurement_noise_cov[
                            idx][-1] + alpha_sigma * np.var(temp_sigma))
                        Gamma[idx] = (1 - alpha_gamma) * Gamma[idx] + alpha_gamma * np.cov(temp_Gamma)
                    temp_lambda = np.linalg.inv(Gamma[idx])
                    Lambda[idx].append(np.linalg.inv(Gamma[idx]))
                    neighbor_num = len(nodes_list[idx])
                    temp_w = np.matmul(temp_lambda, np.ones(neighbor_num).reshape(-1, 1)) / np.matmul(
                        np.ones(neighbor_num).reshape(1, -1),
                        np.matmul(temp_lambda, np.ones(neighbor_num).reshape(-1, 1))).reshape(1, -1)
                    temp_w = np.squeeze(temp_w, axis=1)
                    w[idx].append(temp_w)
                posteriors[idx] = copy.deepcopy(gaussian(initial_mu, measurement_noise_cov[idx][-1]))
                action[idx, id_time] = action_[idx, id_time] = initial_action
                mean_[idx, id_time] = initial_mu
                x[idx, id_time] = action[idx, id_time] - z_list[id_time] + np.random.normal(0, measurement_noise[idx],
                                                                                            1)
                cov_[idx, id_time] = measurement_noise_cov[idx][-1]
                continue
            # Kalman filter
            posterior = posteriors[idx]
            k = -(posterior.cov + process_noise_cov) / (
                    posterior.cov + process_noise_cov + measurement_noise_cov[idx][-1])

            mean_[idx, id_time] = posterior.mean + k * (
                    x[idx, id_time - 1] + posterior.mean - action[idx, id_time - 1])
            cov_[idx, id_time] = (1 + k) * posterior.cov
            action_[idx, id_time] = mean_[idx, id_time]
            posteriors[idx] = gaussian(mean_[idx, id_time], cov_[idx, id_time])

            # social
            # w = np.zeros(len(nodes_list[idx])) + 1 / (len(nodes_list[idx]))
            temp_a = [action_[idx, id_time]]
            for i in range(len(nodes_list[idx])):
                if nodes_list[idx][i] != idx:
                    temp_a.append(action[nodes_list[idx][i], id_time - 1])

            action[idx, id_time] = np.dot(w[idx][-1], np.array(temp_a))
            # if idx == 0:
            #     print(temp_a, w[idx][-1])
            x[idx, id_time] = action[idx, id_time] - z_list[id_time] + np.random.normal(0, measurement_noise[idx], 1)
            # if action[idx, id_time] > 10:
            #     print(np.array(temp_a).reshape(1, -1).squeeze())
    # print(measurement_noise_cov)
    result_dict = {"z": z_list, "x": x, "mean": mean_, "cov": cov_, "simga": measurement_noise_cov,
                   "Lambda": Lambda, "action": action, "action_": action_, "w": w, "iteration_num": args.iteration_num,
                   "nodes_list": nodes_list, "true_sigma": measurement_noise}
    with open("data/neighbor_star_s_differentNoise_uniform3(" + str(args.iteration_num) + "_" + str(
            args.block_num) + ").pkl", "wb") as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    fig = plt.figure(figsize=(12, 15), dpi=300)
    for idx, sub_graph_res in enumerate(list_of_res):
        plt.subplot(6, 1, idx + 1)
        plt.plot(z_list, "green")
        # plt.scatter(range(T), x[idx, :], color="red")
        plt.plot(mean_[idx, :], "black", linewidth=3)
        plt.plot(action[idx, :], "orange")
        plt.legend(["true z", r"mean of $p(z_t|x^{1:t})$", "action"])
        plt.title("agent " + str(idx + 1))
        # plt.savefig("imgs/agent"+str(idx+1)+".pdf")
        # plt.show()
    plt.savefig("imgs/neighbor.pdf")
    plt.show()
    # belief1 = []
    # belief2 = []
    # for delta in deltas[idx]:
    #     belief1.append(delta[0][1])
    #     belief2.append(delta[0][2])
    # plt.scatter(range(1, len(belief1) + 1), belief1)
    # plt.scatter(range(1, len(belief2) + 1), belief2)
    # plt.legend(["$J_{12}$", "$J_{13}$"])
    # plt.show()
