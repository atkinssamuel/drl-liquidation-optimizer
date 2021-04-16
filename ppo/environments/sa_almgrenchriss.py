import gym
import numpy as np
import matplotlib.pyplot as plt

from gym.spaces import Box
from shared.shared_utils import ind, sample_Xi


class SingleAgentAlmgrenChriss(gym.Env):
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


    Indexing reference:
    - Arrays start at 0 but the way in which states are referenced starts at 1
    - Therefore, k will range from 1 -> N and arrays will be indexed by calling the ind(k) callback
        - k = 1 corresponds to the 0-th element in an array
    - To go from k to k + 1, we liquidate n_k shares
        - n_k is the number of shares liquidated from k-1 to k
    """
    def __init__(self, D):
        # action space is continuous and 1-dimensional for a single agent
        self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # observation space for state-space formulation from MADRL paper is D + 3
        self.D = D
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.D+3,), dtype=np.float32)

        # initialize the state to [0, 0, ..., 0] (length D+1) + [1, 1]
        self.state = np.hstack((np.zeros(shape=(self.D+1)), [1, 1]))

        self.initial_market_price = 50
        volatility = 0.12
        arr = 0.1
        daily_trading_volume = 5000000
        yearly_trading_days = 250
        self.T = 60
        self.N = 60
        self.X = 10 ** 6
        self.lam = 1.2 * 10 ** (-6)

        bid_ask = 1 / 8
        daily_volatility = volatility / np.sqrt(yearly_trading_days)  # 0.007589
        self.tau = self.T / self.N  # 1.0
        self.sigma = daily_volatility * self.initial_market_price  # 0.3794
        self.alpha = arr / yearly_trading_days * self.initial_market_price  # 0.02
        self.epsilon = bid_ask / 2  # 0.0625
        self.eta = bid_ask / (0.01 * daily_trading_volume)  # 2.5e-06
        self.gamma = bid_ask / (0.1 * daily_trading_volume)  # 2.5e-07

        self.kappa = self.compute_kappa()  # 0.60626 for lam = 10**(-6)
        self.step_array = np.arange(self.N)

        self.k = 1

        # price
        self.S = np.zeros(shape=(self.N,))
        self.S[ind(self.k)] = self.initial_market_price
        self.S_tilde = np.zeros(shape=(self.N,))
        self.S_tilde[ind(self.k)] = self.initial_market_price

        # inventory
        self.x = np.zeros(shape=(self.N,))
        self.x[ind(self.k)] = self.X

        # n_k-1 is the number of shares sold at k-1
        self.n = np.zeros(shape=(self.N,))

        # revenue
        self.R = np.zeros(shape=(self.N,))

        # normalized number of remaining trades [0, 1]
        self.L = 1

    def step(self, action):
        self.n[ind(self.k)] = action * self.x[ind(self.k)]
        self.k += 1
        self.step_inventory()
        self.step_price()
        self.step_cash()
        self.step_trades()

        self.state[:self.D] = self.state[1:self.D + 1]
        self.state[self.D] = np.log(self.S_tilde[ind(self.k)]/self.S_tilde[ind(self.k)-1])
        self.state[-2] = (self.N - ind(self.k) * self.tau)/self.N
        self.state[-1] = self.x[ind(self.k)]/self.X

        reward = 0

        done = False
        if ind(self.k) == self.N-1 or self.x[ind(self.k)] < 1:
            reward = np.average(self.R) - self.gamma / 2 * np.var(self.R)
            done = True

        info = {}

        return self.state, reward, done, info

    def reset(self):
        """
        Resets the state according to the parameters set while instantiating the class
        :return: state
        """
        # initialize the state to [0, 0, ..., 0] (length D+1) + [1, 1]
        self.state = np.hstack((np.zeros(shape=(self.D+1)), [1, 1]))

        self.k = 1

        # price
        self.S = np.zeros(shape=(self.N,))
        self.S[ind(self.k)] = self.initial_market_price
        self.S_tilde = np.zeros(shape=(self.N,))
        self.S_tilde[ind(self.k)] = self.initial_market_price

        # inventory
        self.x = np.zeros(shape=(self.N,))
        self.x[ind(self.k)] = self.X

        # n_k-1 is the number of shares sold at k-1
        self.n = np.zeros(shape=(self.N,))

        # revenue
        self.R = np.zeros(shape=(self.N,))

        # normalized number of remaining trades [0, 1]
        self.L = 1

        return self.state

    def render(self):
        """
        Renders an episode and illustrates the price process, cash balance, shares sold/iteration, and inventory
        :return: None
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Simulation Plot")

        a_thousand = 1000
        a_million = 1000000
        axes[0, 0].plot(self.step_array * self.tau, self.S_tilde)
        axes[0, 0].set(ylabel="Price")

        axes[0, 1].plot(self.step_array * self.tau, self.R / a_million, label="Cash Balance")
        axes[0, 1].plot(self.step_array * self.tau, np.ones(self.N, ) * self.initial_market_price * self.X / a_million,
                        label="Portfolio Value")
        axes[0, 1].legend()
        axes[0, 1].set(ylabel="Balance ($M)")

        # removed final value (accounting for the fact that you can't sell any more shares at the final time step)
        axes[1, 0].plot(self.step_array[:-1] * self.tau, self.n[:-1], color="m")
        axes[1, 0].set(xlabel="Time", ylabel="Shares Sold/Iteration")

        axes[1, 1].plot(self.step_array * self.tau, self.x / a_thousand, color="orange")
        axes[1, 1].set(ylabel="Inventory (k)")

        for axis in axes.flat:
            axis.grid(True)

        plt.show()

    def step_trades(self):
        """
        Steps the normalized number of trades left forward

        L_k = L_k-1 - 1 / N

        :return: None
        """
        self.L -= 1 / self.N

    def step_inventory(self):
        """
        Steps the inventory forward:

        X_k = X_k-1 - n_k-1

        :return: None
        """
        self.x[ind(self.k)] = self.x[ind(self.k)-1] - self.n[ind(self.k)-1]

    def step_price(self):
        """
        Steps the price forward:

        S_k = S_k-1 + sigma * sqrt(tau) * W - tau * g(n_k/tau)
        S_k_tilde = S_k-1 - h(n_k/tau)

        next price = previous price + noise from random walk process - permanent price impact
        next price tilde = previous price - temporary price impact

        :return: None
        """
        self.S[ind(self.k)] = self.S[ind(self.k)-1] + self.sigma * np.sqrt(self.tau) * sample_Xi() - \
                              self.tau * self.compute_g(self.n[ind(self.k)-1])
        self.S_tilde[ind(self.k)] = self.S[ind(self.k)] - self.compute_h(self.n[ind(self.k)-1])

    def step_cash(self):
        """
        Steps the cash process forward:

        C_k = C_k-1 + P_tilde_k-1 * n_k-1

        :return: None
        """
        self.R[ind(self.k)] = self.R[ind(self.k) - 1] + self.S_tilde[ind(self.k) - 1] * self.n[ind(self.k) - 1]

    def compute_kappa(self):
        """
        Function that computes the unique positive solution for the optimal liquidation strategy

        kappa satisfies:
        2 / tau^2 * (cosh(kappa * tau) - 1) = kappa^2_tilde

        where:
        kappa^2_tilde = lambda * sigma ^2 / eta_tilde

        and:
        eta_tilde = (eta * (1 - gamma * tau / (2 * eta))

        therefore:
        kappa = arccosh(kappa^2_tilde * tau^2 / 2 + 1) / tau

        :return: float (kappa)
        """
        eta_tilde = self.eta * (1 - self.gamma * self.tau / (2 * self.eta))
        kappa_2_tilde = self.lam * self.sigma ** 2 / eta_tilde
        kappa = np.arccosh(kappa_2_tilde * self.tau ** 2 / 2 + 1) / self.tau
        return kappa

    def compute_h(self, n_k):
        """
        Computes and returns the h value for the specified input, n_k

        h(n_k/tau) = epsilon * sgn(n_k) + eta / tau * n_k

        :param: n_k: # of shares sold from k-1 to k
        :return: float
        """
        return self.epsilon * np.sign(n_k) + self.eta / self.tau * n_k

    def compute_g(self, n_k):
        """
        Computes and returns the g value (permanent price impact) for the specified input, n_k
        :param n_k: # of shares sold from k-1 to k
        :return: float
        """
        return self.gamma * n_k / self.tau
