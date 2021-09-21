import copy
import torch
import random
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from collections import namedtuple
from scipy.stats import norm
from function import *
import matplotlib.pyplot as plt

gaussian = namedtuple('Gaussian', ['mean', 'cov'])


def dynamics(self, D: float, T: int, ini_state: float, noise_var: float):
    s = np.zeros(self.T)  # states initialization
    s[0] = self.ini_state
    noise = np.random.normal(0, self.noise_var, self.T)

    for t in range(self.T - 1):
        # calculate the state of t+1
        s[t + 1] = D * s[t] + noise[t]

    return s


if __name__ == "__main__":
    args = parse_dataset_args()

    process_noise = 0.01
    measurement_noise=1
    process_noise_cov = process_noise ** 2
    measurement_noise_cov = measurement_noise ** 2

    z_list = get_singal(args)
    T = len(z_list)
    z_list = z_list + np.random.normal(0, process_noise, T)

    D = 1
    x = np.zeros(T)

    mean_ = np.zeros(T)  # estimate (posterior mean)
    cov_ = np.zeros(T)
    choice = np.zeros(T + 1)

    # plt.plot(singal)
    # plt.scatter(range(len(singal)), m, marker='.', color='red', s=100)
    # plt.show()
    initial_guess = gaussian(0.5, measurement_noise_cov)
    posterior = initial_guess


    mean_[0] = 0.5
    x[0] = z_list[0] + np.random.normal(0, measurement_noise, 1)
    for id_time in range(T):
        x[id_time] = z_list[id_time] + np.random.normal(0, measurement_noise, 1)
        k = (posterior.cov + process_noise_cov) / (posterior.cov + process_noise_cov + measurement_noise_cov)
        # k=-1/((posterior.cov**(-1)+measurement_noise_cov)*measurement_noise_cov)
        mean_[id_time] = posterior.mean + k * (x[id_time] - posterior.mean)
        cov_[id_time] = (1 - k) * posterior.cov

        posterior = gaussian(mean_[id_time], cov_[id_time])


        # print(mean_[id_time]," ",z," ",mean_[id_time]-z)
plt.plot(z_list,"green")
plt.scatter(range(T),x,color="red")
plt.plot(mean_,"black")
plt.legend(["true z",r"mean of $p(z_t|x^{1:t})$","x"])
plt.show()
