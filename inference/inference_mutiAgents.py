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

    args = parse_dataset_args()
    singal = get_singal(args)  # 获取信号强度

    log_prob = 1 / 2 * np.log((1 - singal[0]) / singal[0])  # preset probability when blue ball is picked
    # 确定图参数
    w = initialize_w(log_prob, args, 1 - singal[0], np.random.choice([1, -1], p=[singal[0], 1 - singal[0]]))
    J = np.ones((args.n_nodes, args.n_nodes)) * args.J_times
    mask = get_graph_struct(args, log_prob)  # 获取图结构
    neighbors = dict(pd.DataFrame(np.where(mask)).T.groupby(0).apply(lambda x: x[1].values.tolist()))
    J *= mask
    # 获取子图信息
    list_of_res_history = []
    list_of_res, nodes_list, graphs, algo_obj = get_graphs(args, w, J, mask)

    list_of_res_history.append(list_of_res.copy())
    graphs_list = [copy.deepcopy(graphs)]



    # 参数初始化
    D = 1
    alpha_sigma = 0.1   #
    alpha_gamma = 0.1
    process_noise = 0.001  # 真实因变量的noise,非常小
    process_noise_cov = process_noise ** 2
    measurement_noise = [0.1, 0.2, 0.3, 0.4]  # observation的noise不同人有不同的值 增大

    # measurement_noise_cov = np.zeros(args.n_nodes)
    measurement_noise_cov = np.array(
        [1.0 ** 2 for i in range(len(measurement_noise))])  # 每个人初始对observation的noise，不等于真实noise

    T = len(singal)
    z_list = singal + np.random.normal(0, process_noise, T)

    x = np.zeros((args.n_nodes, T))
    action = np.zeros((args.n_nodes, T))  # action
    action_ = np.zeros((args.n_nodes, T))  # action independent
    action_[:, 0] = np.repeat(0.5, args.n_nodes)
    action[:, 0] = np.repeat(0.5, args.n_nodes)
    mean_ = np.zeros((args.n_nodes, T))  # estimate (posterior mean)
    mean_[:, 0] = np.repeat(0.5, args.n_nodes)
    cov_ = np.zeros((args.n_nodes, T))  # estimate (posterior cov)

    for i in range(args.n_nodes):
        x[i, 0] = action[i, 0] - z_list[0] + np.random.normal(0, measurement_noise[i], 1)

    posteriors = [copy.deepcopy(gaussian(0.5, measurement_noise_cov[i])) for i in range(args.n_nodes)]
    deltas = [[] for i in range(args.n_nodes)] #Lambda
    gama = 0 #Gamma
    w = [[] for i in range(args.n_nodes)]
    for id_time in range(1, T):  # determine how many loops are required

        for idx, sub_graph_res in enumerate(list_of_res):

            #sigma更新方式与Gamma一样
            if id_time % args.iteration_num == 0 and id_time != 0:
                pre = int((id_time / args.iteration_num - 1) * args.iteration_num)
                temp = x[idx, pre:id_time] - action[idx, pre:id_time]
                # print(temp)

                measurement_noise_cov[idx] = (1 - alpha_sigma) * measurement_noise_cov[idx] + alpha_sigma * np.var(temp)

                posteriors = [copy.deepcopy(gaussian(0.5, measurement_noise_cov[i])) for i in range(args.n_nodes)]

                action[idx, id_time] = action_[idx, id_time] = mean_[idx, id_time] = 0.5
                x[idx, id_time] = action_[idx, id_time] - z_list[id_time] + np.random.normal(0, measurement_noise[i], 1)

            if id_time / args.iteration_num < 1:
                w[idx] = np.array([1] + [0] * (len(nodes_list[idx]) - 1))
            elif id_time % args.iteration_num == 0:
                pre = int((id_time / args.iteration_num - 1) * args.iteration_num)
                temp = [action_[idx, pre + 1:id_time]]
                for i in range(len(nodes_list[idx])):
                    if nodes_list[idx][i] != idx:
                        temp.append(action[nodes_list[idx][i], pre:id_time - 1])
                temp = np.array(temp)
                if id_time / args.iteration_num < 2:
                    gama = np.cov(temp)
                else:

                    gama = (1 - alpha_gamma) * gama + alpha_gamma * np.cov(temp)
                delta = np.linalg.inv(gama)
                deltas[idx].append(delta)
                neighor_num = len(nodes_list[idx])#neighbor
                w[idx] = np.matmul(delta, np.ones(neighor_num).reshape(-1, 1)) / np.matmul(
                    np.ones(neighor_num).reshape(1, -1),
                    np.matmul(delta, np.ones(neighor_num).reshape(-1,
                                                                  1))).reshape(
                    1, -1)
                w[idx] = np.squeeze(w[idx], axis=1)
                continue
            # Kalman filter
            posterior = posteriors[idx]


            #process_noise_cov设置为0
            k = -(posterior.cov + process_noise_cov) / (posterior.cov + process_noise_cov + measurement_noise_cov[idx])
            # print(k)
            # k=-1/((posterior.cov**(-1)+measurement_noise_cov[idx])*measurement_noise_cov[idx])

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

            action[idx, id_time] = np.dot(w[idx], np.array(temp_a))
            x[idx, id_time] = action[idx, id_time] - z_list[id_time] + np.random.normal(0, measurement_noise[idx], 1)
            # if action[idx, id_time] > 10:
            #     print(np.array(temp_a).reshape(1, -1).squeeze())
    # print(measurement_noise_cov)
    fig = plt.figure(figsize=(12, 12), dpi=300)
    for idx, sub_graph_res in enumerate(list_of_res):
        plt.subplot(2, 2, idx + 1)
        plt.plot(z_list, "green")
        plt.scatter(range(T), x[idx, :], color="red")
        plt.plot(mean_[idx, :], "black")
        plt.plot(action[idx, :], "orange")
        plt.legend(["true z", r"mean of $p(z_t|x^{1:t})$", "action", "x"])
        plt.title("agent " + str(idx + 1))
        # plt.savefig("imgs/agent"+str(idx+1)+".pdf")
        # plt.show()
    # plt.savefig("imgs/neighbor.pdf")
    plt.show()
    belief1 = []
    belief2 = []
    for delta in deltas[idx]:
        belief1.append(delta[0][1])
        belief2.append(delta[0][2])
    plt.scatter(range(1, len(belief1) + 1), belief1)
    plt.scatter(range(1, len(belief2) + 1), belief2)
    plt.legend(["$J_{12}$", "$J_{13}$"])
    plt.show()
