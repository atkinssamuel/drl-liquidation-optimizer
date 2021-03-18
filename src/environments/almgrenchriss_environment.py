import numpy as np
import matplotlib.pyplot as plt

from src.helpers import sample_Xi


class AlmgrenChrissEnvironment:
    """
    Initial Parameters:
    Real-World Example: Optimal Execution of Portfolio Transactions -  Robert Almgren and Neil Chriss
    - $50 current market price
    - 30% volatility
    - 10% expected annual rate of return
    - 1/8 bid-ask spread
    - 5 million share median daily trading volume
    - Trading year of 250 days

    The above settings yield the following sigma and alpha:
    - daily volatility of 0.3/sqrt(250) = 0.019
    - expected fractional return of 0.1/250 = 4 x 10^(-4)
    - to obtain parameters for sigma and alpha, we must scale by the price:
        - sigma = 0.019 * 50
        - alpha = 4 x 10^(-4) * 50 = 0.02

    - to choose epsilon, we divide the bid-ask spread by two: epsilon = 1/16
    - to determine eta, we assume that for each one percent of the daily volume we trade, we incur a price impact equal
    to the bid-ask spread
        - since we are trading at 5 million shares, we have (1/8) / (0.01 * 5 million) = 2.5 * 10^(-6)

    - price effects become "significant" when we sell 10% of the daily volume
        - we suppose that "significant" means that the price depression is one bid-ask spread and that the effect is
        linear for smaller and larger trading rates
        - gamma = (1/8) /(0.1 * 5 million) = 2.5 * 10^(-7)
    """

    def __init__(self):
        """
        Parameter and state initialization
        """
        self.initial_market_price = 50

        volatility = 0.3
        arr = 0.1
        bid_ask = 1 / 8
        daily_trading_volume = 5000000
        yearly_trading_days = 250
        daily_volatility = volatility / np.sqrt(yearly_trading_days)

        self.T = 60
        self.N = 60
        self.tau = self.T / self.N
        self.X = 10 ** 6

        self.sigma = daily_volatility * self.initial_market_price
        self.alpha = arr / yearly_trading_days * self.initial_market_price
        self.epsilon = bid_ask / 2
        self.eta = bid_ask / (0.01 * daily_trading_volume)
        self.gamma = bid_ask / (0.1 * daily_trading_volume)
        self.lam = 2 * 10 ** (-6)

        self.step_array = np.arange(self.N)
        self.P = np.zeros(shape=(self.N,))
        self.P[0] = self.initial_market_price
        self.x = np.zeros(shape=(self.N,))
        self.x[0] = self.X
        self.n = np.zeros(shape=(self.N,))
        self.c = np.zeros(shape=(self.N,))

        self.k = 1

    def step(self, n=0):
        """
        Sets the control at the previous time step to n and steps the state forward
        :param n: float [0, 1]
        :return: None
        """
        assert (self.k < self.N)
        self.n[self.k-1] = n
        self.step_inventory()
        self.step_price()
        self.step_cash()
        self.k += 1

    def step_inventory(self):
        """
        Steps the inventory forward:

        X_k = X_k-1 - n_k-1

        :return: None
        """
        self.x[self.k] = self.x[self.k - 1] - self.n[self.k-1]

    def step_price(self):
        """
        Steps the price forward:

        P_k = P_k-1 + sigma * sqrt(tau) * W - gamma * n_k-1

        next price = previous price + noise from random walk process - permanent price impact

        :return: None
        """
        self.P[self.k] = self.P[self.k - 1] + self.sigma * np.sqrt(self.tau) * sample_Xi() - self.gamma * self.n[
            self.k-1]

    def step_cash(self):
        """
        Steps the cash process forward:

        C_k = C_k-1 + P_k-1 * n_k

        :return: None
        """
        self.c[self.k] = self.c[self.k - 1] + self.P[self.k - 1] * self.n[self.k-1]

    def plot_simulation(self, save_path=None):
        """
        Plots the price dynamics, cash balance, control, and inventory
        :return: None
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Simulation Plot")

        a_thousand = 1000
        a_million = 1000000
        axes[0, 0].plot(self.step_array * self.tau, self.P)
        axes[0, 0].set(ylabel="Price")

        axes[0, 1].plot(self.step_array * self.tau, self.c / a_million, label="Cash Balance")
        axes[0, 1].plot(self.step_array * self.tau, np.ones(self.N, ) * self.initial_market_price * self.X / a_million,
                        label="Portfolio Value")
        axes[0, 1].legend()
        axes[0, 1].set(ylabel="Balance ($M)")

        # removed first value (may need to account for initial value)
        axes[1, 0].plot(self.step_array[1:] * self.tau, self.n[1:], color="m")
        axes[1, 0].set(xlabel="Time", ylabel="Shares Sold/Iteration")

        axes[1, 1].plot(self.step_array * self.tau, self.x / a_thousand, color="orange")
        axes[1, 1].set(ylabel="Inventory (k)")

        for axis in axes.flat:
            axis.grid(True)

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.clf()
