import numpy as np
import matplotlib.pyplot as plt


class TradingEnvironment:

    @staticmethod
    def sample_Xi():
        mu, sigma = 0, 1
        return np.random.normal(mu, sigma, 1)

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
        initial_market_price = 50
        volatility = 0.3
        arr = 0.1
        bid_ask = 1 / 8
        daily_trading_volume = 5000000
        yearly_trading_days = 250
        daily_volatility = volatility / np.sqrt(volatility)

        self.T = 50
        self.N = 50
        self.tau = self.T / self.N
        self.X = 10**6

        self.sigma = daily_volatility * initial_market_price
        self.alpha = arr / yearly_trading_days * initial_market_price
        self.epsilon = bid_ask / 2
        self.eta = bid_ask / (0.01 * daily_trading_volume)
        self.gamma = bid_ask / (0.1 * daily_trading_volume)

        self.step_array = np.arange(self.N)
        self.P = np.zeros(shape=(self.N,))
        self.P[0] = initial_market_price
        self.x = np.zeros(shape=(self.N,))
        self.x[0] = self.X
        self.n = np.zeros(shape=(self.N,))

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

        # removed first value (may need to account for initial value)
        axes[2].plot(self.step_array[1:] * self.tau, self.n[1:], color="m")
        axes[2].set(xlabel="Time", ylabel="Shares Sold/Iteration")

        for axis in axes.flat:
            axis.grid(True)

        plt.show()
