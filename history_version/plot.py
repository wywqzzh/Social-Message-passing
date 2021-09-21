import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy

with open(
        "data/neighbor_star_z_differentNoise_uniform2(40_30).pkl",
        "rb",
) as f:
    data = pickle.load(f)

z_list = data["z"]
x = data["x"]
mean = data["mean"]
cov = data["cov"]
simga = data["simga"]
Lambda = data["Lambda"]
action = data["action"]
action2 = data["action2"]
action3 = data["action3"]
action_ = data["action_"]
w = data["w"]
w2 = data["w2"]
w3 = data["w3"]
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
for i in range(int(T / iteration_num) + 1):
    Xticks.append(i * iteration_num)
# graph1
fig = plt.figure(figsize=(24, 30), dpi=100)
for idx in range(len(nodes_list)):
    plt.subplot(5, 1, idx + 1)
    plt.plot(z_list, "green")
    # plt.scatter(range(T), x[idx, :], color="red")
    plt.plot(mean[idx, :], "black", linewidth=2.5)
    plt.plot(action[idx, :], "orange", linewidth=2)
    plt.plot(action2[idx, :], "blue", linewidth=1.5)
    plt.plot(action3[idx, :], "#ff1493")
    plt.legend(["true z", r"mean of $p(z_t|x^{1:t})$", "action", "action2", "action3"])
    plt.title("agent " + str(idx + 1))
    plt.xlim(0, iteration_num * int(T / iteration_num))
    plt.xticks(Xticks)
    # plt.savefig("imgs/agent"+str(idx+1)+".pdf")
    # plt.show()
plt.tight_layout()
plt.xlabel("trial")
plt.savefig("imgs/" + basename + "KalmanFilterInference.pdf")
plt.show()


fig = plt.figure(figsize=(24, 30), dpi=100)
for idx in range(len(nodes_list)):
    plt.subplot(5, 1, idx + 1)
    plt.plot(action[idx, :]-z_list[:], "orange", linewidth=3)
    plt.plot(action2[idx, :]-z_list[:], "blue", linewidth=2)
    plt.plot(action3[idx, :]-z_list[:], "#ff1493", linewidth=1)
    plt.plot(np.repeat(0,len(z_list)),"black")
    plt.legend(["action", "action2", "action3"])
    plt.title("agent " + str(idx + 1))
    plt.xlim(0, iteration_num * int(T / iteration_num))
    plt.xticks(Xticks)
    # plt.savefig("imgs/agent"+str(idx+1)+".pdf")
    # plt.show()
plt.tight_layout()
plt.xlabel("trial")
plt.savefig("imgs/" + basename + "differenceOfActionAndZ.pdf")
plt.show()


# graph2
fig = plt.figure(figsize=(12, 6), dpi=200)
for idx in range(len(nodes_list)):
    plt.plot(cov[idx, :], color=point_color[idx])

plt.xlabel("trial")
plt.legend(["agent 1", "agent 2", "agent 3", "agent 4", "agent 5"])
# plt.savefig("imgs/agent"+str(idx+1)+".pdf")
# plt.show()
plt.xlim(0, iteration_num * int(T / iteration_num))
plt.xticks(Xticks)
plt.tight_layout()
plt.savefig("imgs/" + basename + "KalmanFilterVariance.pdf")
plt.show()

Xticks = [i for i in range(1, int(T / iteration_num))]

# graph3
fig = plt.figure(figsize=(16, 12), dpi=100)
for idx in range(len(nodes_list)):
    # 对角线
    plt.subplot(5, 3, 3 * idx + 1)
    plt.title("agent " + str(idx + 1))
    temp = []
    for j in range(1, int(T / iteration_num)):
        temp.append(Lambda[idx][j - 1][0][0])
    plt.plot(Xticks, temp, color=point_color[idx])
    legend = ["$J_{" + str(idx + 1) + str(idx + 1) + "}$"]
    L = copy.deepcopy(nodes_list[idx])
    L.remove(idx)
    for i in range(len(L)):
        temp = []
        for j in range(1, int(T / iteration_num)):
            temp.append(Lambda[idx][j - 1][i + 1][i + 1])
            if idx == 0:
                print(Lambda[idx][j - 1])
        plt.plot(Xticks, temp, color=point_color[L[i]])
    for i in L:
        legend.append("$J_{" + str(i + 1) + str(i + 1) + "}$")
    plt.legend(legend)
    plt.xticks(Xticks)
    plt.xlim(1, int(T / iteration_num))

    # idx与其他节点的关系
    plt.subplot(5, 3, 3 * idx + 2)
    plt.title("agent " + str(idx + 1))
    index = 0
    L = copy.deepcopy(nodes_list[idx])
    L.remove(idx)
    for i in range(len(L)):
        temp = []
        for j in range(1, int(T / iteration_num)):
            temp.append(Lambda[idx][j - 1][0][i + 1])
        plt.plot(Xticks, temp, color=point_color[L[i]])
    legend = []
    for i in L:
        legend.append("$J_{" + str(idx + 1) + str(i + 1) + "}$")
    plt.legend(legend)
    plt.xticks(Xticks)
    plt.xlim(1, int(T / iteration_num))

    # 其他节点直接的关系
    plt.subplot(5, 3, 3 * idx + 3)
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
                temp.append(Lambda[idx][j - 1][i + 1][j + 1])
            legend.append("$J_{" + str(idJ + 1) + str(idK + 1) + "}$")
            plt.plot(Xticks, temp, color=edg_color[str(idJ) + str(idK)])
    plt.legend(legend)
    plt.xticks(Xticks)
    plt.xlim(1, int(T / iteration_num))
plt.xlabel("block num")
plt.tight_layout()
plt.savefig("imgs/" + basename + "GraphLearningOptimality.pdf")
plt.show()

# graph4
fig = plt.figure(figsize=(8, 8), dpi=300)
for idx in range(len(nodes_list)):
    plt.plot(simga[idx][:], color=point_color[idx])
    plt.plot(np.repeat(true_sigma[idx] ** 2, len(simga[idx][:])), color=point_color[idx], linestyle="--")

plt.xticks(range(int(T / iteration_num)))
plt.xlim(0, int(T / iteration_num))

# plt.legend(["agent 1", "agent 2", "agent 3", "agent 4", "agent 5", r"true $\sigma^2$"])
plt.xlabel("block num")
plt.tight_layout()
plt.savefig("imgs/" + basename + "DetectionNoiseLearning.pdf")
plt.show()

# graph 5
fig = plt.figure(figsize=(24, 12), dpi=100)
for idx in range(len(nodes_list)):

    plt.subplot(len(nodes_list), 3, 3 * idx + 1)
    plt.title("agent " + str(idx + 1))
    L = copy.deepcopy(nodes_list[idx])
    L.remove(idx)
    legend = []
    temp = []
    for j in range(int(T / iteration_num)):
        temp.append(w[idx][j][0])
    legend.append("$w_{" + str(idx + 1) + str(idx + 1) + "}$")
    plt.plot(temp, color=point_color[idx])
    for i in L:
        legend.append("$w_{" + str(idx + 1) + str(i + 1) + "}$")
        temp = []
        for j in range(int(T / iteration_num)):
            temp.append(w[idx][j][L.index(i) + 1])

        plt.plot(temp, color=point_color[i])
    plt.legend(legend)
    plt.xticks(range(int(T / iteration_num)))
    plt.xlim(0, int(T / iteration_num))
    plt.xticks(range(int(T / iteration_num)))

    plt.subplot(len(nodes_list), 3, 3 * idx + 2)
    plt.title("agent " + str(idx + 1))
    L = copy.deepcopy(nodes_list[idx])
    L.remove(idx)
    legend = []
    temp = []
    for j in range(int(T / iteration_num)):
        temp.append(w2[idx][j][0])
    legend.append("$w_{" + str(idx + 1) + str(idx + 1) + "}$")
    plt.plot(temp, color=point_color[idx])
    for i in L:
        legend.append("$w_{" + str(idx + 1) + str(i + 1) + "}$")
        temp = []
        for j in range(int(T / iteration_num)):
            temp.append(w2[idx][j][L.index(i) + 1])

        plt.plot(temp, color=point_color[i])
    plt.legend(legend)
    plt.xticks(range(int(T / iteration_num)))
    plt.xlim(0, int(T / iteration_num))
    plt.xticks(range(int(T / iteration_num)))

    plt.subplot(len(nodes_list), 3, 3 * idx + 3)
    plt.title("agent " + str(idx + 1))
    L = copy.deepcopy(nodes_list[idx])
    L.remove(idx)
    legend = []
    temp = []
    for j in range(int(T / iteration_num)):
        temp.append(w3[idx][j][0])
    legend.append("$w_{" + str(idx + 1) + str(idx + 1) + "}$")
    plt.plot(temp, color=point_color[idx])
    for i in L:
        legend.append("$w_{" + str(idx + 1) + str(i + 1) + "}$")
        temp = []
        for j in range(int(T / iteration_num)):
            temp.append(w3[idx][j][L.index(i) + 1])

        plt.plot(temp, color=point_color[i])
    plt.legend(legend)
    plt.xticks(range(int(T / iteration_num)))
    plt.xlim(0, int(T / iteration_num))
    plt.xticks(range(int(T / iteration_num)))
# plt.legend(["agent 1","agent 2","agent 3","agent 4","agent 5",r"true $\sigma$"])
plt.xlabel("block num")
plt.tight_layout()
plt.savefig("imgs/" + basename + "weight.pdf")
plt.show()
