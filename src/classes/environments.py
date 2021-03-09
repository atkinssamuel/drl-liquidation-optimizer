import numpy as np
import matplotlib.pyplot as plt


def sample_Xi():
    mu, sigma = 0, 1
    return np.random.normal(mu, sigma, 1)

class AlmgrenChrissEnvironment:
    """
    Initial Parameters:
    Real-World Example: Optimal Execution of Portfolio Transactions -  Robert Almgreny and Neil Chriss
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
        self.X = 10**6

        self.sigma = daily_volatility * self.initial_market_price
        self.alpha = arr / yearly_trading_days * self.initial_market_price
        self.epsilon = bid_ask / 2
        self.eta = bid_ask / (0.01 * daily_trading_volume)
        self.gamma = bid_ask / (0.1 * daily_trading_volume)
        self.lam = 2*10**(-6)

        self.step_array = np.arange(self.N)
        self.P = np.zeros(shape=(self.N,))
        self.P[0] = self.initial_market_price
        self.x = np.zeros(shape=(self.N,))
        self.x[0] = self.X
        self.n = np.zeros(shape=(self.N,))
        self.c = np.zeros(shape=(self.N,))

        self.k = 1

    def step(self, n=0):
        assert (self.k < self.N)
        self.n[self.k] = n
        self.step_inventory()
        self.step_price()
        self.step_cash()
        self.k += 1

    def step_inventory(self):
        self.x[self.k] = self.x[self.k - 1] - self.n[self.k]

    def step_price(self):
        self.P[self.k] = self.P[self.k - 1] + self.sigma * np.sqrt(self.tau) * sample_Xi() - self.gamma * self.n[
            self.k]

    def step_cash(self):
        self.c[self.k] = self.c[self.k-1] + self.P[self.k-1] * self.n[self.k]

    def plot_simulation(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Simulation Plot")

        thousand = 1000
        axes[0, 0].plot(self.step_array * self.tau, self.P)
        axes[0, 0].set(ylabel="Price")

        axes[0, 1].plot(self.step_array * self.tau, self.c, label="Cash Balance")
        axes[0, 1].plot(self.step_array * self.tau, np.ones(self.N,) * self.initial_market_price * self.X,
                        label="Portfolio Value")
        axes[0, 1].legend()
        axes[0, 1].set(ylabel="Balance ($k)")

        # removed first value (may need to account for initial value)
        axes[1, 0].plot(self.step_array[1:] * self.tau, self.n[1:], color="m")
        axes[1, 0].set(xlabel="Time", ylabel="Shares Sold/Iteration")

        axes[1, 1].plot(self.step_array * self.tau, self.x/thousand, color="orange")
        axes[1, 1].set(ylabel="Inventory (k)")

        for axis in axes.flat:
            axis.grid(True)

        plt.show()


class JaimungalEnvironment:
    """
    Main environment variables:
    v_t - trading rate = the speed at which the agent is liquidating or acquiring shares (the control in the RL
    context)
    Q_t - agent's inventory
    S_t - the mid-price process
    S_hat_t - execution price process = the price at which the agent can sell or purchase the asset (execution
    price)
    X_t - agent's cash process

    Variable DEs:
    dQ_t = -v_t * dt, Q_0 = q
    dS_t = -g(v_t) * dt + sigma * dW_t, S_0 = S
    (W_t is standard Brownian motion, g is the permanent price impact function)
    S_hat_t = S_t - (1/2 * delta + f(v_t)), S_hat_0 = S_hat
    (delta is the bid-ask spread, f is the temporary price impact function)
    dX_t = S_hat_t * v_t * dt, X_0 = x

    Impact Functions:
    f(u) = ku
    g(u) = bu
    """
    def __init__(self):
        """
        initialization function
        """
        # time horizon
        self.T = 100
        # number of trades
        self.N = 100
        # increment
        self.tau = self.T/self.N

        # current time step
        self.t = 0

        # permanent and temporary impact parameters
        self.k = 0.001
        self.b = 0.001

        # penalty parameters
        # urgency
        self.phi = 0.0000001
        # remaining penalty
        self.alpha = 0.01

        # bid-ask spread
        self.delta = 1/8

        # initial values
        self.q = 10 ** 3
        self.S_init = 50
        self.X_init = 0

        # determining sigma
        volatility = 0.3
        yearly_trading_days = 250
        daily_volatility = volatility / np.sqrt(yearly_trading_days)
        self.sigma = daily_volatility * self.S_init

        # state variables
        self.Q = np.zeros(shape=(self.N,))
        self.Q[0] = self.q
        self.S = np.zeros(shape=(self.N,))
        self.S[0] = self.S_init
        self.S_hat = np.zeros(shape=(self.N,))
        self.X = np.zeros(shape=(self.N,))
        self.X[0] = self.X_init

        # control
        self.v = np.zeros(shape=(self.N,))
        self.v[0] = 0

        # initializes S_hat assuming 0 control
        self.update_S_hat()

    def f(self):
        """
        temporary price impact
        :return: float
        """
        return self.k * self.v[self.t]

    def g(self):
        """
        permanent price impact
        :return: float
        """
        return self.b * self.v[self.t]

    def update_Q(self):
        """
        updates Q at time T + 1
        :return: None
        """
        self.Q[self.t+1] = self.Q[self.t] - self.v[self.t] * self.tau

    def update_S(self):
        """
        updates S at time t + 1
        :return: None
        """
        self.S[self.t+1] = self.S[self.t] + (-self.g() + self.sigma * sample_Xi() * np.sqrt(self.tau)) * self.tau

    def update_S_hat(self):
        """
        updates S_hat at time t
        :return: None
        """
        self.S_hat[self.t] = self.S[self.t] - (1/2 * self.delta + self.f()) * self.tau

    def update_X(self):
        """
        updates X at time t + 1
        :return: None
        """
        self.X[self.t+1] = self.X[self.t] + self.S_hat[self.t] * self.v[self.t] * self.tau

    def step(self):
        """
        updates all state variables and increments the current time step
        :return: None
        """
        self.update_Q()
        self.update_S()
        self.update_X()
        self.t += 1
        self.update_S_hat()

    def get_constant(self):
        """
        calculates a constant MO trading strategy
        :return: float
        """
        return self.q/self.T

    def get_optimal(self):
        """
        calculates the optimal trading strategy
        :return: float
        """
        gamma = np.sqrt(self.phi / self.k)
        zeta = (self.alpha - 1/2 * self.b + np.sqrt(self.k * self.phi))/(self.alpha - 1/2 * self.b - np.sqrt(self.k *
                                                                                                       self.phi))
        return gamma * (zeta * np.exp(gamma * (self.T - self.t)) + np.exp(-gamma * (self.T - self.t)))/\
               (zeta * np.exp(gamma * self.T) - np.exp(-gamma * self.T)) * self.q

    def plot_simulation(self):
        """
        plots shares (Q), control (v), cash process (X), and execution price process (S_hat)
        :return: None
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Simulation Plot")

        interval = np.arange(len(self.Q))

        axes[0, 0].plot(interval, self.Q, label="Inventory (Q)", color="gold")
        axes[0, 0].set(xlabel="Trading Iteration")
        axes[0, 0].set(ylabel="Shares")
        axes[0, 0].legend()

        axes[0, 1].plot(interval[:-1], self.v[:-1], label="Control (v)", color="deepskyblue")
        axes[0, 1].set(xlabel="Trading Iteration")
        axes[0, 1].set(ylabel="Shares/Iteration")
        axes[0, 1].legend()

        axes[1, 0].plot(interval, self.X, label="Cash Process (X)", color="lime")
        axes[1, 0].set(xlabel="Trading Iteration")
        axes[1, 0].set(ylabel="CA$")
        axes[1, 0].legend()

        axes[1, 1].plot(interval, self.S_hat, label="Execution Price Process (S_hat)", color="magenta")
        axes[1, 1].set(xlabel="Trading Iteration")
        axes[1, 1].set(ylabel="CA$")
        axes[1, 1].legend()

        for axis in axes.flat:
            axis.grid(True)

        plt.show()
