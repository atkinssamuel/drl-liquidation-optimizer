import numpy as np
import matplotlib.pyplot as plt

from src.helpers import sample_Xi, ind
from src.constants import Algos


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


    Indexing reference:
    - Arrays start at 0 but the way in which states are referenced starts at 1
    - Therefore, k will range from 1 -> N and arrays will be indexed by calling the ind(k) callback
        - k = 1 corresponds to the 0-th element in an array
    - To go from k to k + 1, we liquidate n_k shares
        - n_k is the number of shares liquidated from k-1 to k
    """

    def __init__(self):
        """
        Parameter and state initialization
        """
        self.initial_market_price   = 50
        volatility                  = 0.12
        arr                         = 0.1
        daily_trading_volume        = 5000000
        yearly_trading_days         = 250
        self.T                      = 60
        self.N                      = 60
        self.X                      = 10 ** 6
        self.lam                    = 1.2 * 10 ** (-6)

        bid_ask = 1 / 8
        daily_volatility = volatility / np.sqrt(yearly_trading_days)            # 0.007589
        self.tau = self.T / self.N                                              # 1.0
        self.sigma = daily_volatility * self.initial_market_price               # 0.3794
        self.alpha = arr / yearly_trading_days * self.initial_market_price      # 0.02
        self.epsilon = bid_ask / 2                                              # 0.0625
        self.eta = bid_ask / (0.01 * daily_trading_volume)                      # 2.5e-06
        self.gamma = bid_ask / (0.1 * daily_trading_volume)                     # 2.5e-07

        self.kappa = self.compute_kappa()                                       # 0.60626 for lam = 10**(-6)

        self.k = 1

        self.step_array = np.arange(self.N)
        self.P = np.zeros(shape=(self.N,))
        self.P[ind(self.k)] = self.initial_market_price
        self.P_ = np.zeros(shape=(self.N,))
        self.P_[ind(self.k)] = self.initial_market_price
        self.x = np.zeros(shape=(self.N,))
        self.x[ind(self.k)] = self.X
        # n_k is the number of shares sold from n_k-1 to n_k
        self.n = np.zeros(shape=(self.N,))
        self.c = np.zeros(shape=(self.N,))
        self.L = np.zeros(shape=(self.N,))
        self.L[ind(self.k)] = 1
        self.Q = np.zeros(shape=(self.N,))
        self.compute_Q()
        self.U = np.zeros(shape=(self.N,))
        self.compute_U()

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
        eta_tilde = self.eta * (1 - self.gamma * self.tau/(2 * self.eta))
        kappa_2_tilde = self.lam * self.sigma**2 / eta_tilde
        kappa = np.arccosh(kappa_2_tilde * self.tau**2 / 2 + 1)/self.tau
        return kappa

    def step(self, n=0, reward_option="madrl"):
        """
        Sets the control at the previous time step to n and steps the state forward. Returns the reward specified
        by the reward_option parameter from k-1 to k
        :param n: float: number of shares sold from k-1 to k (n_k)
        :param reward_option: Algos.custom/Algos.madrl/etc.
        :return: reward: float
        """
        self.k += 1
        self.n[ind(self.k)] = n
        self.step_inventory()
        self.step_price()
        self.step_cash()
        self.step_trades()
        return self.get_reward(reward_option=reward_option)

    def step_trades(self):
        """
        Steps the normalized number of trades left forward

        L_k = L_k-1 - tau

        :return: None
        """
        self.L[ind(self.k)] = self.L[ind(self.k)-1] - 1/self.N

    def step_inventory(self):
        """
        Steps the inventory forward:

        X_k = X_k-1 - n_k * X_k-1

        :return: None
        """
        self.x[ind(self.k)] = self.x[ind(self.k)-1] - self.n[ind(self.k)]

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

    def compute_E(self):
        """
        Computes and returns the E value for the reward function from k-1 -> k:

        E_x = sum_{k=1}^N tau * x_k * g(n_k/tau) + sum_{k=1}^N n_k * h(n_k/tau)

        :return: float
        """
        E_x = 0
        for k_ in range(1, self.N+1):
            E_x += self.tau * self.x[ind(k_)] * self.compute_g(self.n[ind(k_)]) \
                   + self.n[ind(k_)] * self.compute_h(self.n[ind(k_)])
        return E_x

    def compute_V(self):
        """
        Computes and returns the V value for the reward function:

        V_x = sigma^2 * sum_{k=1}^N tau * x_k^2

        :return: float
        """
        V_x = 0
        for k_ in range(1, self.N+1):
            V_x += self.tau * self.x[ind(k_)]**2
        V_x = self.sigma**2 * V_x
        return V_x

    def compute_U(self):
        """
        Computes the U value and stores it in the U vector:

        U = E + lambda * V

        :return: None
        """
        self.U[ind(self.k)] = self.compute_E() + self.lam * self.compute_V()

    def compute_Q(self):
        eta_tilde = self.eta - self.gamma * self.tau / 2
        E_x_n = self.c[0] + self.x[0] * self.P_[0] - self.gamma / 2 * self.x[0] ** 2 - \
                eta_tilde * self.tau * np.sum(np.square(self.n))
        V_x_n = self.sigma ** 2 * self.tau * np.sum(np.square(self.x))
        self.Q[ind(self.k)] = E_x_n - self.gamma / 2 * V_x_n

    def get_reward(self, reward_option="custom"):
        """
        Returns the reward for the specified reward option for taking action k-1 and transitioning from state k-1 to
        state k

        "custom" reward:
        R = {0 if self.k != self.N
        E[x_n] - gamma / 2 * Var[x_n] otherwise}

        "madrl" reward:
        R = (U[k-1] - U[k])/U[k-1]

        :return: float
        """
        if reward_option == Algos.custom:
            self.compute_Q()
            # k ranges from 1 -> N
            if ind(self.k) == self.N-1:
                return self.Q[ind(self.k)]
            return 0
            # return (self.Q[ind(self.k)-1] - self.Q[ind(self.k)])/self.Q[ind(self.k)-1]
        elif reward_option == Algos.madrl:
            self.compute_U()
            return (self.U[ind(self.k)-1] - self.U[ind(self.k)]) \
                   / self.U[ind(self.k)-1]

    def step_price(self):
        """
        Steps the price forward:

        P_k = P_k-1 + sigma * sqrt(tau) * W - tau * g(n_k/tau)
        P_k_tilde = P_k-1 - h(n_k/tau)

        next price = previous price + noise from random walk process - permanent price impact
        next price tilde = previous price - temporary price impact

        :return: None
        """
        self.P[ind(self.k)] = self.P[ind(self.k)-1] + self.sigma * np.sqrt(self.tau) * sample_Xi() - \
                              self.tau * self.compute_g(self.n[ind(self.k)])
        self.P_[ind(self.k)] = self.P[ind(self.k)-1] - self.compute_h(self.n[ind(self.k)])

    def step_cash(self):
        """
        Steps the cash process forward:

        C_k = C_k-1 + P_tilde_k-1 * n_k

        :return: None
        """
        self.c[ind(self.k)] = self.c[ind(self.k)-1] + self.P_[ind(self.k)-1] * self.n[ind(self.k)]

    def plot_simulation(self, save_path=None):
        """
        Plots the price dynamics, cash balance, control, and inventory
        :return: None
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Simulation Plot")

        a_thousand = 1000
        a_million = 1000000
        axes[0, 0].plot(self.step_array * self.tau, self.P_)
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
