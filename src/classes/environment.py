import numpy as np
import matplotlib.pyplot as plt


class TradingEnvironment:
    @staticmethod
    def sample_Xi():
        mu, sigma = 0, 1
        return np.random.normal(mu, sigma, 1)

    def __init__(self, sigma=0.1, gamma=0.99, eta=0.6, epsilon=0.1, T=1000, N=1000, X=10000, P_init=500000):
        # hyper-parameters
        self.X = X
        self.sigma = sigma
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon
        self.T = T
        self.N = N
        self.tau = T / N

        # data
        self.step_array = np.arange(N)
        self.P = np.zeros(shape=(N,))
        self.P[0] = P_init
        self.x = np.zeros(shape=(N,))
        self.x[0] = X
        self.n = np.zeros(shape=(N,))

        # other variables
        self.k = 1

    def step(self, n=0):
        assert (self.k < self.N)
        self.n[self.k] = n
        self.step_inventory()
        self.step_price()
        self.k += 1

    def step_inventory(self):
        self.x[self.k] = self.x[self.k - 1] - self.n[self.k]

    def step_price(self):
        self.P[self.k] = self.P[self.k - 1] + self.sigma * np.sqrt(self.tau) * self.sample_Xi() - self.gamma * self.n[
            self.k]

    def plot_simulation(self):
        fig, axes = plt.subplots(3, figsize=(14, 10))
        fig.suptitle("Simulation Plot")

        thousand = 1000
        axes[0].plot(self.step_array * self.tau, self.P/thousand)
        axes[0].set(ylabel="Price ($k)")

        axes[1].plot(self.step_array * self.tau, self.x/thousand, color="orange")
        axes[1].set(ylabel="Inventory (k)")

        axes[2].plot(self.step_array * self.tau, self.n, color="m")
        axes[2].set(xlabel="Time", ylabel="Shares Sold/Iteration")

        for axis in axes.flat:
            axis.grid(True)

        plt.show()
