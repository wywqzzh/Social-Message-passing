import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy

with open(
        "data/neighbor_square(5_100).pkl",
        "rb",
) as f:
    data = pickle.load(f)

z_list = data["z"]
x = data["x"]

mean = data["mean"]
cov = data["cov"]
simga = data["simga"]

Lambda_Complete = data["Lambda_Complete"]
# Lambda_Sparse = data["Lambda_Sparse"]
Lambda_Independent = data["Lambda_Independent"]

action_Complete = data["action_Complete"]
# action_Sparse = data["action_Sparse"]
action_Independent = data["action_Independent"]
action_Identical = data["action_Identical"]
action_ = data["action_"]

w_Complete = data["w_Complete"]
# w_Sparse = data["w_Sparse"]
w_Independent = data["w_Independent"]
w_Identical = data["w_Identical"]

iteration_num = data["iteration_num"]
nodes_list = data["nodes_list"]

point_color = {0: "#bf242a", 1: "#96541d", 2: "#e6b422", 3: "#006e54", 4: "#3869CB"}
edg_color = {"01": "#ab431d", "10": "#ab431d",
             "02": "#e39217", "20": "#e39217",
             "03": "#3b6d2b", "30": "#3b6d2b",
             "04": "#8851b6", "40": "#8851b6",
             "12": "#c28120", "21": "#c28120",
             "13": "#596920", "31": "#596920",
             "14": "#b33c74", "41": "#b33c74",
             "23": "#6a9a3c", "32": "#6a9a3c",
             "24": "#fa5889", "42": "#fa5889",
             "34": "#00779f", "43": "#00779f"}
T = len(z_list)
Xticks = []
true_sigma = data["true_sigma"]
temp = [str(i) for i in true_sigma]
basename = "different_noise(" + ','.join(temp) + ")_"
# basename=""
for i in range(0,int(T / iteration_num) + 1, 4):
    Xticks.append(i * iteration_num)

# grahp1 Kalman optimality
fig = plt.figure(figsize=(16, 3), dpi=300)
for idx in range(len(nodes_list)):
    plt.plot(cov[idx, :], color=point_color[idx])

plt.xlabel("trial")
plt.legend(["agent 1", "agent 2", "agent 3", "agent 4"])
plt.xlim(0, iteration_num * int(T / iteration_num))
plt.xticks(Xticks)
plt.tight_layout()
plt.savefig("imgs_constrain/" + basename + "Kalman Optimality.pdf")
plt.show()

# graph2 Kalman parameter learning:
fig = plt.figure(figsize=(10, 3), dpi=300)
for idx in range(len(nodes_list)):
    plt.plot(simga[idx][:], color=point_color[idx])
    plt.plot(np.repeat(true_sigma[idx] ** 2, len(simga[idx][:])), color=point_color[idx], linestyle="--")

plt.xticks(Xticks)
plt.xlim(0, int(T / iteration_num))
plt.xlabel("block num")
plt.tight_layout()
plt.savefig("imgs_constrain/" + basename + "Kalman Parameter Learning.pdf")
plt.show()

# graph3 Models' performance
fig = plt.figure(figsize=(16, 15), dpi=100)
for idx in range(len(nodes_list)):
    plt.subplot(4, 1, idx + 1)
    plt.plot(mean[idx, :] - z_list[:], "black", linewidth=3)
    plt.plot(action_Complete[idx, :] - z_list[:], "orange", linewidth=2.5)
    plt.plot(action_Independent[idx, :] - z_list[:], "#0000ff", linewidth=2)
    plt.plot(action_Identical[idx, :] - z_list[:], "#ff1493", linewidth=1.5)
    plt.plot(np.repeat(0, len(z_list)), "black")
    plt.legend(
        ["action of Isolated model", "action of Complete model",
         "action of  Independent model",
         "action of Identical model"
         ])
    plt.title("agent " + str(idx + 1))
    plt.xlim(0, iteration_num * int(T / iteration_num))
    plt.xticks(Xticks)
plt.tight_layout()
plt.xlabel("trial")
plt.savefig("imgs_constrain/" + basename + "Models' performance trial.pdf")
plt.show()

action_Isolated_block = [[] for i in range(len(nodes_list))]
action_Complete_block = [[] for i in range(len(nodes_list))]
# action_Sparse_block = [[] for i in range(len(nodes_list))]
action_Independent_block = [[] for i in range(len(nodes_list))]
action_Identical_block = [[] for i in range(len(nodes_list))]
z = [z_list[0]]
for idx in range(len(nodes_list)):
    temp_Isolated = 0
    temp_Complete = 0
    # temp_Sparse = 0
    temp_Independent = 0
    temp_Identical = 0

    for i in range(1, len(z_list)):
        if i % iteration_num == 0:
            action_Isolated_block[idx].append(temp_Isolated / (iteration_num - 1))
            action_Complete_block[idx].append(temp_Complete / (iteration_num - 1))
            # action_Sparse_block[idx].append(temp_Sparse / (iteration_num - 1))
            action_Independent_block[idx].append(temp_Independent / (iteration_num - 1))
            action_Identical_block[idx].append(temp_Identical / (iteration_num - 1))
            z.append(z_list[i])
            temp_Isolated = 0
            temp_Complete = 0
            # temp_Sparse = 0
            temp_Independent = 0
            temp_Identical = 0
            continue
        temp_Isolated += (action_[idx, i] - z_list[i]) ** 2
        temp_Complete += (action_Complete[idx, i] - z_list[i]) ** 2
        # temp_Sparse += (action_Sparse[idx, i] - z_list[i])**2
        temp_Independent += (action_Independent[idx, i] - z_list[i]) ** 2
        temp_Identical += (action_Identical[idx, i] - z_list[i]) ** 2

fig = plt.figure(figsize=(10, 15), dpi=100)
for idx in range(len(nodes_list)):
    plt.subplot(4, 1, idx + 1)
    plt.plot(action_Isolated_block[idx], "black", linewidth=3)
    plt.plot(action_Complete_block[idx], "orange", linewidth=2.5)
    # plt.plot(action_Sparse_block[idx], "#00ff00", linewidth=2)
    plt.plot(action_Independent_block[idx], "#0000ff", linewidth=2)
    plt.plot(action_Identical_block[idx], "#ff1493", linewidth=1.5)
    plt.legend(
        ["action of Isolated model", "action of Complete model",
         "action of  Independent model",
         "action of Identical model"
         ])
    plt.title("agent " + str(idx + 1))
    plt.xlim(0, int(T / iteration_num))
    plt.xticks([i for i in range(0,int(T / iteration_num), 10)])
plt.tight_layout()
plt.xlabel("bock num")
plt.savefig("imgs_constrain/" + basename + "Models' performance block.pdf")
plt.show()

# graph4 Graph parameters
Xticks = [i for i in range(0, int(T / iteration_num), 10)]

fig = plt.figure(figsize=(16, 12), dpi=100)
for idx in range(len(nodes_list)):
    # 对角线
    plt.subplot(4, 3, 3 * idx + 1)
    plt.title("agent " + str(idx + 1))
    temp = []
    for j in range(1, int(T / iteration_num)):
        temp.append(Lambda_Complete[idx][j - 1][0][0])
    plt.plot(temp, color=point_color[idx])
    legend = ["$J_{" + str(idx + 1) + str(idx + 1) + "}$"]
    L = copy.deepcopy(nodes_list[idx])
    L.remove(idx)
    for i in range(len(L)):
        temp = []
        for j in range(1, int(T / iteration_num)):
            temp.append(Lambda_Complete[idx][j - 1][i + 1][i + 1])
        plt.plot(temp, color=point_color[L[i]])
    for i in L:
        legend.append("$J_{" + str(i + 1) + str(i + 1) + "}$")
    plt.legend(legend)
    plt.xticks(Xticks)
    plt.xlabel("bock num")
    plt.xlim(0, int(T / iteration_num))

    # idx与其他节点的关系
    plt.subplot(4, 3, 3 * idx + 2)
    plt.title("agent " + str(idx + 1))
    index = 0
    L = copy.deepcopy(nodes_list[idx])
    L.remove(idx)
    for i in range(len(L)):
        temp = []
        for j in range(1, int(T / iteration_num)):
            temp.append(Lambda_Complete[idx][j - 1][0][i + 1])
        plt.plot( temp, color=point_color[L[i]])
    legend = []
    for i in L:
        legend.append("$J_{" + str(idx + 1) + str(i + 1) + "}$")
    plt.legend(legend)
    plt.xticks(Xticks)
    plt.xlabel("bock num")
    plt.xlim(0, int(T / iteration_num))

    # 其他节点直接的关系
    plt.subplot(4, 3, 3 * idx + 3)
    plt.title("agent " + str(idx + 1))
    L = copy.deepcopy(nodes_list[idx])
    L.remove(idx)
    legend = []
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            idJ = L[i]
            idK = L[j]
            temp = []
            for k in range(1, int(T / iteration_num)):
                temp.append(Lambda_Complete[idx][k - 1][i + 1][j + 1])
            legend.append("$J_{" + str(idJ + 1) + str(idK + 1) + "}$")
            plt.plot( temp, color=edg_color[str(idJ) + str(idK)])
    plt.legend(legend)
    plt.xticks(Xticks)
    plt.xlabel("bock num")
    plt.xlim(0, int(T / iteration_num))
plt.tight_layout()
plt.savefig("imgs_constrain/" + basename + "Graph Parameters Complete.pdf")
plt.show()

fig = plt.figure(figsize=(16, 12), dpi=100)
for idx in range(len(nodes_list)):
    # 对角线
    plt.subplot(4, 3, 3 * idx + 1)
    plt.title("agent " + str(idx + 1))
    temp = []
    for j in range(1, int(T / iteration_num)):
        temp.append(Lambda_Independent[idx][j - 1][0][0])
    plt.plot(temp, color=point_color[idx])
    legend = ["$J_{" + str(idx + 1) + str(idx + 1) + "}$"]
    L = copy.deepcopy(nodes_list[idx])
    L.remove(idx)
    for i in range(len(L)):
        temp = []
        for j in range(1, int(T / iteration_num)):
            temp.append(Lambda_Independent[idx][j - 1][i + 1][i + 1])
        plt.plot( temp, color=point_color[L[i]])
    for i in L:
        legend.append("$J_{" + str(i + 1) + str(i + 1) + "}$")
    plt.legend(legend)
    plt.xticks(Xticks)
    plt.xlabel("bock num")
    plt.xlim(0, int(T / iteration_num))

    # idx与其他节点的关系
    plt.subplot(4, 3, 3 * idx + 2)
    plt.title("agent " + str(idx + 1))
    index = 0
    L = copy.deepcopy(nodes_list[idx])
    L.remove(idx)
    for i in range(len(L)):
        temp = []
        for j in range(1, int(T / iteration_num)):
            temp.append(Lambda_Independent[idx][j - 1][0][i + 1])
        plt.plot( temp, color=point_color[L[i]])
    legend = []
    for i in L:
        legend.append("$J_{" + str(idx + 1) + str(i + 1) + "}$")
    plt.legend(legend)
    plt.xticks(Xticks)
    plt.xlabel("bock num")
    plt.xlim(0, int(T / iteration_num))

    # 其他节点直接的关系
    plt.subplot(4, 3, 3 * idx + 3)
    plt.title("agent " + str(idx + 1))
    L = copy.deepcopy(nodes_list[idx])
    L.remove(idx)
    legend = []
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            idJ = L[i]
            idK = L[j]
            temp = []
            for k in range(1, int(T / iteration_num)):
                temp.append(Lambda_Independent[idx][k - 1][i + 1][j + 1])
            legend.append("$J_{" + str(idJ + 1) + str(idK + 1) + "}$")
            plt.plot(temp, color=edg_color[str(idJ) + str(idK)])
    plt.legend(legend)
    plt.xticks(Xticks)
    plt.xlabel("bock num")
    plt.xlim(0, int(T / iteration_num))
plt.tight_layout()
plt.savefig("imgs_constrain/" + basename + "Graph Parameters Independent.pdf")
plt.show()

# graph 5 Cue combination Weights
fig = plt.figure(figsize=(8, 12), dpi=100)
for idx in range(len(nodes_list)):

    plt.subplot(len(nodes_list), 1, idx + 1)
    plt.title("agent " + str(idx + 1))
    L = copy.deepcopy(nodes_list[idx])
    L.remove(idx)
    legend = []
    temp = []
    for j in range(int(T / iteration_num)):
        temp.append(w_Complete[idx][j][0])
    legend.append("$w_{" + str(idx + 1) + str(idx + 1) + "}$")
    plt.plot(temp, color=point_color[idx])
    for i in L:
        legend.append("$w_{" + str(idx + 1) + str(i + 1) + "}$")
        temp = []
        for j in range(int(T / iteration_num)):
            temp.append(w_Complete[idx][j][L.index(i) + 1])

        plt.plot(temp, color=point_color[i])
    plt.legend(legend)
    plt.xticks(Xticks)
    plt.xlim(0, int(T / iteration_num))

# plt.legend(["agent 1","agent 2","agent 3","agent 4","agent 5",r"true $\sigma$"])
plt.xlabel("block num")
plt.tight_layout()
plt.savefig("imgs_constrain/" + basename + "Cue combination Weights Complete.pdf")
plt.show()



fig = plt.figure(figsize=(8, 12), dpi=100)
for idx in range(len(nodes_list)):

    plt.subplot(len(nodes_list), 1, idx + 1)
    plt.title("agent " + str(idx + 1))
    L = copy.deepcopy(nodes_list[idx])
    L.remove(idx)
    legend = []
    temp = []
    for j in range(int(T / iteration_num)):
        temp.append(w_Independent[idx][j][0])
    legend.append("$w_{" + str(idx + 1) + str(idx + 1) + "}$")
    plt.plot(temp, color=point_color[idx])
    for i in L:
        legend.append("$w_{" + str(idx + 1) + str(i + 1) + "}$")
        temp = []
        for j in range(int(T / iteration_num)):
            temp.append(w_Independent[idx][j][L.index(i) + 1])

        plt.plot(temp, color=point_color[i])
    plt.legend(legend)
    plt.xticks(Xticks)
    plt.xlim(0, int(T / iteration_num))


# plt.legend(["agent 1","agent 2","agent 3","agent 4","agent 5",r"true $\sigma$"])
plt.xlabel("block num")
plt.tight_layout()
plt.savefig("imgs_constrain/" + basename + "Cue combination Weights Independent.pdf")
plt.show()
