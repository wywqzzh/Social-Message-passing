import copy
import torch
import random
import pickle
import pandas as pd
from function import *

if __name__ == "__main__":

    args = parse_dataset_args()
    last_v_w = 0.1
    last_v_J = 0.1
    ball_list = (list(np.repeat(-1, args.iteration_num)) + list(np.repeat(1, args.iteration_num))) * args.block_num

    singal = get_singal(args)  # 获取信号强度
    log_prob = 1 / 2 * np.log((1 - singal[0]) / singal[0])  # preset probability when blue ball is picked
    # 确定图参数
    w = initialize_w(log_prob, args, 1 - singal[0], ball_list[0])
    J = np.ones((args.n_nodes, args.n_nodes)) * args.J_times
    mask = get_graph_struct(args, log_prob)  # 获取图结构
    neighbors = dict(pd.DataFrame(np.where(mask)).T.groupby(0).apply(lambda x: x[1].values.tolist()))
    J *= mask

    # 获取子图信息
    list_of_res_history = []
    list_of_res, nodes_list, graphs, algo_obj = get_graphs(args, w, J, mask)
    # if needs to add initial guess into result
    if args.add_initial and args.get_subgraph in ['neighbor_node', 'interpolation_node']:
        list_of_res_history.append([{'belief': [np.vstack([np.exp(-w[i]), np.exp(w[i])]).T / np.vstack(
            [np.exp(-w[i]), np.exp(w[i])]).T.sum(1, keepdims=True)]} for i in nodes_list])

    list_of_res_history.append(list_of_res.copy())
    graphs_list = [copy.deepcopy(graphs)]

    # create list to store simulated data for each subgraph (initial data is w)
    simulated_history = []
    simulated_history.append([])
    for i in range(args.n_nodes):
        initial = ((w[nodes_list[i]] > 0).astype(int) - 0.5) * 2
        simulated_history[0].append(initial[nodes_list[i].index(i)])

    batch_history = [[] for i in range(args.n_nodes)]

    observation_history = [[] for i in graphs]
    gradient_front = dict(zip(range(args.n_nodes), [[]] * args.n_nodes))
    gradient = dict(zip(range(args.n_nodes), [[]] * args.n_nodes))

    for id_time, item in enumerate(ball_list):  # determine how many loops are required
        # Step 2: learning
        # 根据设定时长获取observation
        if id_time == 100:
            xxx = 0
        if id_time % args.skip_observe_t == 0:
            for idx in range(len(list_of_res)):
                observation_history[idx].append(
                    random.choices([item, -item], [1 - singal[id_time], singal[id_time]])[0])
        # 学习每个子图
        for idx, sub_graph_res in enumerate(list_of_res):
            # batch is prepared for gradient calculation
            batch = np.zeros(len(nodes_list[idx])).reshape(1, -1)
            for i in range(len(nodes_list[idx])):
                if nodes_list[idx][i] == idx and not args.self_batch_simulated:
                    batch[:, i] = observation_history[idx][-1]
                else:
                    batch[:, i] = simulated_history[id_time][nodes_list[idx][i]]
            batch_history[idx].append(batch)
            batch_T = batch_history[idx][max(len(batch_history[idx]) - args.J_T, 0):]

            mask = np.matrix((graphs[idx].adj != 0).astype(int))  # here mask is

            graphs[idx].w = assign_w(args, idx, nodes_list, graphs, observation_history, batch_history, log_prob)
            # fit J
            graphs[idx].J, gradient_front_temp, gradient_temp = fit_J(args, idx, graphs, batch_T, sub_graph_res, mask)
            gradient_front[idx] += gradient_front_temp
            gradient[idx] += gradient_temp

        # Step 3: bp using just fitted J and w
        if args.get_subgraph != 'single_node':
            list_of_res = algo_obj.run(graphs, verbose=args.verbose, use_binary=False)
            simulated_temp = []
            inverse_taus = np.repeat(args.inverse_tau, args.n_nodes)
            # print(inverse_taus)
            inverse_taus = [10, 10, 10, 10, 10]
            for idx, sub in enumerate(list_of_res):
                if args.policy == "deterministic":
                    result = ((np.argmax(sub['belief'][-1], 1) - 0.5) * 2).astype(int)
                else:
                    result = []
                    for i in range(len(sub['belief'][-1])):
                        b = torch.tensor(sub['belief'][-1][i] * inverse_taus[idx], dtype=torch.float)
                        p = torch.softmax(b, -1)
                        result.append(np.random.binomial(1, p[1], 1) * 2 - 1)
                    result = np.array(result)
                simulated_temp.append(float(result[nodes_list[idx].index(idx)]))
            simulated_history.append(simulated_temp)
            list_of_res_history.append(list_of_res.copy())  # store bp history
        if args.get_subgraph == 'single_node':
            list_of_res = {idx: ([np.exp(-g.w), np.exp(g.w)] / (np.exp(-g.w) + np.exp(g.w))).ravel() for idx, g in
                           enumerate(graphs)}
            for key, sub in list_of_res.items():
                if args.policy == "deterministic":
                    result = np.array(int((np.argmax(sub) - 0.5) * 2)).reshape(-1, 1).ravel()
                else:
                    p = np.array([np.exp(args.inverse_tau * sub[0]), np.exp(args.inverse_tau * sub[1])] / (
                            np.exp(args.inverse_tau * sub[0]) + np.exp(args.inverse_tau * sub[1]))).ravel()
                    result = (np.random.binomial(0, p[0], 1))
                    result = result * 2 - 1
                list_of_res_history.append(list_of_res.copy())
        graphs_list.append(copy.deepcopy(graphs))

    result_dict = get_finall_result(list_of_res_history, nodes_list, graphs_list, gradient_front, gradient, args,
                                    simulated_history)

    # save results
    if singal[0] == 2 / 10:
        base_path = "data/star/task_easy/"
    elif singal[0] == 3 / 10:
        base_path = "data/star/task_normal/"
    elif singal[0] == 4 / 10:
        base_path = "data/star/task_hard/"
    else:
        base_path = "data/star/task_continuous/"
    print(base_path)
    with open(base_path + 'simulated_sample_block_' + args.get_subgraph + '_' + args.output_format + '_lrw' + str(
            args.lr_w) + '_lrJ' + str(args.lr_J) + '_Jx' + str(
        args.J_times) + '_case' + str(args.case) + "_" + args.policy + '_star.pkl',
              'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
