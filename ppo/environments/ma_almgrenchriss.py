import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gym.spaces import Box

from shared.constants import PPODirectories
from shared.shared_utils import ind, sample_Xi


class MultiAgentAlmgrenChriss:
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
    def __init__(self, *D):
        self.num_agents = len(D)
        # action space is continuous and 1-dimensional for a single agent, # agents-dimensional for multi-agent
        self.action_space = Box(low=0, high=1, shape=(self.num_agents,), dtype=np.float32)

        # observation space for state-space formulation from MADRL paper is D + 3
        total_D = sum(D)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=((total_D+3*self.num_agents),), dtype=np.float32)

        # initialize the state to [0, 0, ..., 0] (length D+1) + [1, 1]
        for i in range(len(D)):
            D_state = np.hstack((np.zeros(shape=(D[i]+1)), [1, 1]))
            if i == 0:
                self.state = D_state
            else:
                self.state = np.hstack((self.state, D_state))

        self.initial_market_price = 50
        volatility = 0.12
        arr = 0.1
        daily_trading_volume = 5000000
        yearly_trading_days = 250
        self.T = 60
        self.N = 60
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

    def step(self, multi_agent_action_dict):
        """
        **kwargs must
        :param multi_agent_action_dict: dictionary containing {'actions': [action_1, action_2, ...],
                                                               'agents': [agent_1, agent_2, ...]}
        :return: multi_agent_step_dict: dictionary containing {'state': state,
                                                               'rewards': [reward_1, reward_2, ...],
                                                               'dones': [done_1, done_2, ...],
                                                               'infos': [info_1, info_2, ...]}
        """
        dones = multi_agent_action_dict['dones']
        # for each agent
        observation_index = 0
        total_permanent_price_impact = 0
        total_temporary_price_impact = 0
        for i in range(self.num_agents):
            agent = multi_agent_action_dict['agents'][i]

            if dones[i]:
                total_permanent_price_impact += -agent.X
                continue

            action = multi_agent_action_dict['actions'][i]

            if (ind(self.k) + 1) == (self.N - 1):
                action = 1

            num_shares = action * agent.x[ind(self.k)]
            new_revenue = self.S_tilde[ind(self.k)] * num_shares
            agent.n[ind(self.k)] = num_shares
            agent.step_inventory(self.k+1)
            agent.step_revenue(self.k+1, new_revenue)
            agent.step_trades(self.k+1)

            total_permanent_price_impact += agent.x[ind(self.k)+1] - agent.X
            total_temporary_price_impact -= num_shares

        self.k += 1
        self.step_price(total_permanent_price_impact=total_permanent_price_impact,
                        total_temporary_price_impact=total_temporary_price_impact)

        for i in range(self.num_agents):
            if dones[i]:
                continue
            agent = multi_agent_action_dict['agents'][i]
            D = agent.D
            next_observation_index = observation_index + D + 3
            self.state[observation_index:observation_index+D] = self.state[observation_index+1:observation_index+D+1]
            self.state[observation_index+D] = np.log(self.S_tilde[ind(self.k)]/self.S_tilde[ind(self.k)-1])
            self.state[observation_index+D+1] = (self.N - ind(self.k) * self.tau)/self.N
            self.state[observation_index+D+2] = agent.x[ind(self.k)]/agent.X
            observation_index = next_observation_index

        multi_agent_step_dict = {
            'state': self.state,
            'rewards': [0 for _ in range(self.num_agents)],
            'dones': [False for _ in range(self.num_agents)],
            'infos': [{} for _ in range(self.num_agents)]
        }

        for i in range(self.num_agents):
            if dones[i]:
                multi_agent_step_dict['dones'][i] = True
                continue
            agent = multi_agent_action_dict['agents'][i]
            if ind(self.k) == self.N-1 or agent.x[ind(self.k)] < 1:
                agent.R[ind(self.k):] = np.ones(len(agent.R) - ind(self.k)) * agent.R[ind(self.k)]
                multi_agent_step_dict['rewards'][i] = np.average(agent.R) - agent.risk_aversion / 2 * np.var(agent.R)
                multi_agent_step_dict['dones'][i] = True

        return multi_agent_step_dict

    def reset(self, *agents):
        """
        Resets the state according to the parameters set while instantiating the class
        :return: state
        """
        # initialize the state to [0, 0, ..., 0] (length D+1) + [1, 1]
        for i in range(len(agents)):
            D_state = np.hstack((np.zeros(shape=(agents[i].D + 1)), [1, 1]))
            if i == 0:
                self.state = D_state
            else:
                self.state = np.hstack((self.state, D_state))

        self.k = 1

        # price
        self.S = np.zeros(shape=(self.N,))
        self.S[ind(self.k)] = self.initial_market_price
        self.S_tilde = np.zeros(shape=(self.N,))
        self.S_tilde[ind(self.k)] = self.initial_market_price

        for agent in agents:
            agent.reset()

        return self.state

    def render(self, agents, episode):
        """
        Renders an episode and illustrates the price process as well as the inventory and revenue processes for all of
        the agents involved
        :return: None
        """
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle(f"Episode {episode}")
        gs = gridspec.GridSpec(2, 2)

        a_thousand = 1000
        a_million = 1000000

        # Price Process
        ax = plt.subplot(gs[0, 0])
        ax.set(title="Price Process")
        plt.plot(self.step_array * self.tau, self.S_tilde, label="Price Including Temporary Price Impact")
        plt.plot(self.step_array * self.tau, self.S, label="Price")
        plt.legend()
        ax.set(ylabel="Price")
        ax.grid(True)

        # Revenue Process
        ax = plt.subplot(gs[0, 1])
        ax.set(title="Revenue Process")
        for a in range(len(agents)):
            plt.plot(self.step_array * self.tau, agents[a].R / a_thousand, label=f"Agent {a+1}")
        plt.legend()
        ax.grid(True)
        ax.set(ylabel="Revenue ($k)")
        ax.set(xlabel="Time Step")

        # Inventory Process
        ax = plt.subplot(gs[1, :])
        ax.set(title="Inventory Process")
        for a in range(len(agents)):
            plt.plot(self.step_array * self.tau, agents[a].x / a_million, label=f"Agent {a+1}")
        plt.legend()
        ax.grid(True)
        ax.set(ylabel="Inventory (M)")
        ax.set(xlabel="Time Step")

        filename = PPODirectories.tmp + f"episode-{episode}-simulation.png"

        plt.savefig(filename)
        plt.close()

        return filename

    def step_price(self, total_permanent_price_impact, total_temporary_price_impact):
        """
        Steps the price forward:

        S_0(t) = S_0 + sigma * W(t) + drift
        S{X1, ..., XN}(t) = S_0(t) + gamma * total permanent price impact + lambda * total temporary price impact

        :return: None
        """
        # S_0(t) = S_0 + sigma * W(t) + drift
        self.S[ind(self.k)] = self.S[ind(self.k)-1] + self.sigma * np.sqrt(self.tau) * sample_Xi() + \
                              self.alpha * self.S[ind(self.k)-1]

        # S{X1, ..., XN}(t) = S_0(t) + gamma * total permanent price impact
        self.S[ind(self.k)] = self.S[ind(self.k)] + self.gamma * total_permanent_price_impact

        # S{X1, ..., XN}(t) = S{X1, ..., XN}(t)  + total temporary price impact
        self.S_tilde[ind(self.k)] = self.S[ind(self.k)] + self.lam * total_temporary_price_impact

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
