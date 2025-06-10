import numpy as np
from numpy.random import dirichlet, multinomial
import matplotlib.pyplot as plt
import os


class HDPHMM:
    def __init__(self):
        """
        Generative model with fixed parameters to create a dataset
        """
        self.trends = [[0, 0, 0, 0.2], [0.3, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0.2, -0.1, 0], [0, 0.5, 0, 0]]
        self.intercept = [[8, 8, 8, 8], [-3, -3, -5, -5], [5, 0, 2, 0], [2, 0, 2, 0], [0, 2, -3, 0], [-4, -4, -4, -4]]

        self.n_features = 4
        self.beta = np.array([0.4, 0.2, 0.12, 0.14, 0.04, 0.1])
        self.pi = np.array([[0.8, 0.05, 0.03, 0.05, 0.01, 0.06],
                            [0.13, 0.77, 0.04, 0.04, 0.00, 0.03],
                            [0.10, 0.06, 0.76, 0.04, 0.02, 0.01],
                            [0.11, 0.07, 0.05, 0.7, 0.02, 0.05],
                            [0.09, 0.09, 0.05, 0.05, 0.7, 0.02],
                            [0.1, 0.05, 0.03, 0.05, 0.02, 0.75]])

    def get_x(self, state, d):
        x = d*np.array(self.trends[state]) + np.array(self.intercept[state])
        return x

    def generate(self, T):
        """
        Simulate a time series sample
        """
        sample_x = []
        sample_z = list(np.where(multinomial(1, dirichlet(self.beta)))[0])
        x_t = self.get_x(state=sample_z[0], d=1)
        sample_x.append(x_t)
        z_count = 1
        for _ in range(1, T):
            z_t = np.where(multinomial(1, self.pi[sample_z[-1]]))[0][0]
            if z_t == sample_z[-1]:
                z_count += 1
            else:
                z_count = 1
            x_t = self.get_x(state=sample_z[-1], d=z_count)
            sample_x.append(x_t)
            sample_z.append(z_t)
        return np.array(sample_z), np.stack(sample_x)+np.random.randn(T, self.n_features)*0.3

if __name__ == "__main__":
    np.random.seed(1234)
    N = 150
    n_features = 4
    T = 50
    k_max = 10
    model = HDPHMM()
    x_all, z_all = [], []
    for _ in range(N):
        z, x = model.generate(T)
        x_all.append(x)
        z_all.append(z)
    x_all = np.array(x_all)
    z_all = np.array(z_all)
    with open('./sim_semi_markov_x.npy', 'wb') as f:
        np.save(f, x_all)
    with open('./sim_semi_markov_z.npy', 'wb') as f:
        np.save(f, z_all)

    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    plt.figure(figsize=(10, 3))
    for i in range(x.shape[-1]):
        plt.plot(x[:, i], label="Feature %d" % i)

    color = iter(plt.cm.rainbow(np.linspace(0, 1, 10)))
    z_colors = {}
    for t in range(T - 1):
        if not z[t] in z_colors.keys():
            z_colors[z[t]] = next(color)
        plt.axvspan(t, t + 1, alpha=0.5, color=z_colors[z[t]])
    plt.title("Time series sample with underlying states")
    plt.savefig("./plots/semi_markov_sample.pdf")
