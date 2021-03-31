import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from src.constants import Directories
from src.environments.almgrenchriss_environment import AlmgrenChrissEnvironment
from datetime import datetime
from src.helpers import ind
from src.constants import Algos
from src.models.ddpg_models import DDPGCritic, DDPGActor


class DDPG:
    @staticmethod
    def layer_init_callback(layer):
        """
        Callback function for Xavier layer initialization
        :param layer: torch.nn.Linear
        :return: None
        """
        if type(layer) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    @staticmethod
    def init_madrl_obs(D):
        """
        :param D: # of previous instances to consider
        :return: (D+3,) observation array
        """
        return np.hstack((np.zeros(shape=(D+1)), [1, 1]))

    @staticmethod
    def init_custom_obs(environment):
        """
        Initializes the observatoin vector for the custom algo formulation
        :param environment: almgrenchriss_environment instance
        :return: (3,) environment array
        """
        return np.array([environment.P_[0], environment.x[0], environment.L[0]])

    def __init__(self, algo=Algos.madrl, D=5, lr=0.3, batch_size=1024, discount_factor=0.99, M=200, criticLR=0.001,
                 actorLR=0.001, checkpoint_frequency=20):
        """
        Initializes the parameters and the state, observation, and action vectors

        Indexing reference:
        - Arrays start at 0 but the way in which states are referenced starts at 1
        - Therefore, k will range from 1 -> N and arrays will be indexed by calling the ind(k) callback
            - k = 1 corresponds to the 0-th element in an array
        """
        # initializing a new environment
        self.environment = AlmgrenChrissEnvironment()

        # hyper-parameters
        self.D = D
        self.lr = lr
        self.batch_size = batch_size
        self.discount = discount_factor
        self.checkpoint_frequency = 20
        # number of episodes
        self.M = M

        # algo type
        self.algo = algo
        self.observation = None
        if self.algo == Algos.madrl:
            self.algo_results = Directories.madrl_results
            self.observation = self.init_madrl_obs(self.D)
        elif self.algo == Algos.custom:
            self.observation = self.init_custom_obs(self.environment)
            self.algo_results = Directories.custom_results

        # reward
        self.R = None
        # action
        self.a = None

        self.criticLR = criticLR
        self.actorLR = actorLR

        # plotting parameters
        # moving average length
        self.ma_length = 15
        self.inventory_sim_length = 300

        # replay buffer: obs_i + a + r + obs_i+1
        self.replay_buffer_size = self.observation.shape[0] + 1 + 1 + self.observation.shape[0]
        self.replay_buffer = None

        # critic network initialization
        self.critic = DDPGCritic(self).apply(self.layer_init_callback)
        self.critic_target = DDPGCritic(self).apply(self.layer_init_callback)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.criticLR)

        # actor network initialization
        self.actor = DDPGActor(self).apply(self.layer_init_callback)
        self.actor_target = DDPGActor(self).apply(self.layer_init_callback)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actorLR)

    def update_networks(self, current, target):
        """
        Updates the parameters in the target network using the parameters from the current network
        :param current: current network
        :param target: target network
        :return: None
        """
        for current_parameter, target_parameter in zip(current.parameters(), target.parameters()):
            target_parameter.data.copy_(self.lr * current_parameter.data * (1.0 - self.lr) * target_parameter.data)

    def get_r(self):
        """
        Slides the r vector left in the observation vector and sets the r_kth element in the observation vector to
        the log return at k-1
        :return: None
        """
        self.observation[:self.D] = self.observation[1:self.D + 1]
        self.observation[self.D] = np.log(self.environment.P_[ind(self.environment.k)] /
                                          self.environment.P_[ind(self.environment.k)-1])

    def get_m(self):
        """
        Sets the -2nd element in the observation vector to the m state value
        :return: None
        """
        self.observation[-2] = (self.environment.N - ind(self.environment.k) * self.environment.tau)/self.environment.N

    def get_l(self):
        """
        Sets the last element in the observation vector to the l state value:
        :return: None
        """
        self.observation[-1] = self.environment.x[ind(self.environment.k)] / self.environment.X

    def update_observation(self):
        """
        Updates the observation vector using the previously defined "get" equations and returns the updated observation
        :return: observation vector
        """
        if self.algo == Algos.custom:
            self.observation[0] = self.environment.P_[self.environment.k - 1]
            self.observation[1] = self.environment.x[self.environment.k - 1]
            self.observation[2] = self.environment.L[self.environment.k - 1]
        elif self.algo == Algos.madrl:
            if self.environment.k != 1:
                self.get_r()
            self.get_m()
            self.get_l()
        return self.observation

    def step(self, action):
        """
        Steps the simulation forward using the action, a, and returns obs, action, reward, next_obs
        :param action: float
        :return: obs, action, reward, next_obs
        """
        obs = np.array(self.update_observation())

        num_shares = action * self.environment.x[ind(self.environment.k)]
        self.R = self.environment.step(num_shares, reward_option=self.algo)
        reward = self.R

        next_obs = self.update_observation()

        return obs, action, reward, next_obs

    def add_transition(self, obs, action, reward, next_obs):
        """
        Saves an observation transition into the replay buffer
        :return: None
        """
        replay_buffer_entry = np.zeros(shape=(self.replay_buffer_size,))

        replay_buffer_entry[:obs.shape[0]] = obs
        replay_buffer_entry[obs.shape[0]] = action
        replay_buffer_entry[obs.shape[0]+1] = reward
        replay_buffer_entry[obs.shape[0]+2:] = next_obs

        if self.replay_buffer is None:
            self.replay_buffer = replay_buffer_entry
        else:
            self.replay_buffer = np.vstack((self.replay_buffer, replay_buffer_entry))
        return

    def sample_transitions(self, N):
        """
        Samples N transitions from the replay buffer and returns them as appropriately sized obs, action, reward,
        next_obs np.arrays
        :param N: number of transitions to sample
        :return: obs, action, reward, next_obs
        """
        transition_indices = np.random.randint(self.replay_buffer_size, size=N)
        sample = self.replay_buffer[transition_indices, :]

        obs = sample[:, :self.observation.shape[0]]
        action = sample[:, self.observation.shape[0]].reshape(-1, 1)
        reward = sample[:, self.observation.shape[0]+1].reshape(-1, 1)
        next_obs = sample[:, self.observation.shape[0]+2:]

        return obs, action, reward, next_obs

    def analyze_analytical(self, date_str):
        """
        Simulates the trained model and compares the model's inventory process with the inventory process defined
        by the analytical solution

        q*_n = q_0 sinh(alpha * (T-t))/sinh(alpha * T)

        :return: None
        """
        # noinspection PyUnresolvedReferences
        t = np.linspace(0, self.environment.T, self.environment.N)
        q_n = self.environment.x[0] * np.sinh(self.environment.kappa * (self.environment.T - t)) / \
              np.sinh(self.environment.kappa * self.environment.T)
        q_n_sim = None
        for i in range(self.inventory_sim_length):
            self.environment = AlmgrenChrissEnvironment()

            if self.algo == Algos.madrl:
                self.observation = self.init_madrl_obs(self.D)
            elif self.algo == Algos.custom:
                self.observation = self.init_custom_obs(self.environment)

            for k_ in range(self.environment.N-1):
                self.a = self.actor_target.forward(torch.FloatTensor(self.observation)).detach().numpy()
                self.step(self.a)

            if q_n_sim is None:
                q_n_sim = self.environment.x
                continue
            q_n_sim = np.vstack((q_n_sim, self.environment.x))
        q_n_sim = np.average(q_n_sim, axis=0)

        plt.plot(t, q_n, label="Analytical Solution")
        plt.plot(t, q_n_sim, label="DDPG Result")
        plt.title("Inventory Process")
        plt.xlabel("t")
        plt.ylabel("Inventory")
        plt.grid(True)
        plt.legend()
        plt.savefig(self.algo_results + Directories.model_inv + f"inventory-{date_str}.png")

    def run_ddpg(self):
        """
        Runs the DDPG algorithm for M episodes using the parameters defined above
        :return: None
        """
        critic_losses = []
        actor_losses = []
        is_list = []
        is_ma_list = []
        rewards_list = []

        for i in range(self.M):
            self.environment = AlmgrenChrissEnvironment()
            total_reward = 0
            for k_ in range(self.environment.N-1):
                noise = np.random.normal(0, 0.1, 1)
                self.a = self.actor_target.forward(torch.FloatTensor(self.observation)).detach().numpy() + noise

                obs, action, reward, next_obs = self.step(self.a)
                self.add_transition(obs, action, reward, next_obs)
                total_reward += reward

                if self.replay_buffer.shape[0] < self.batch_size:
                    continue

                obs, action, reward, next_obs = self.sample_transitions(self.batch_size)

                # critic updates
                best_action = self.actor_target.forward(torch.FloatTensor(obs))
                Q_next = self.critic_target.forward(torch.FloatTensor(next_obs), best_action)
                y = torch.FloatTensor(reward) + self.discount * Q_next
                critic_loss = F.mse_loss(self.critic.forward(torch.FloatTensor(obs), torch.FloatTensor(action)), y)
                critic_losses.append(critic_loss.detach().numpy())
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # actor updates
                actor_prediction = self.actor_target.forward(torch.FloatTensor(obs))
                # produces a Q-value that we wish to maximize
                actor_loss = -self.critic_target.forward(torch.FloatTensor(obs), actor_prediction).mean()
                actor_losses.append(actor_loss.detach().numpy())
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.update_networks(self.critic, self.critic_target)
                self.update_networks(self.actor, self.actor_target)

            implementation_shortfall = self.environment.initial_market_price * \
                                       self.environment.X - self.environment.c[-1]

            is_list.append(implementation_shortfall)

            ma_starting_index = max(0, i - self.ma_length)
            is_ma_list.append(sum(is_list[ma_starting_index:i + 1]) / (i - ma_starting_index + 1))
            rewards_list.append(total_reward)

            if i % self.checkpoint_frequency == 0:
                print(f"Episode {i} ({round(i / self.M * 100, 2)}%), Implementation Shortfall = "
                      f"{implementation_shortfall}")

        date_str = str(datetime.now())[2:10] + "_" + str(datetime.now())[11:13] + "-" + str(datetime.now())[14:16]

        fig, axes = plt.subplots(2, figsize=(14, 10))

        axes[0].plot(np.arange(len(critic_losses)), critic_losses, color="sienna")
        axes[0].set(title="Critic Loss")
        axes[0].set(ylabel="MSE Loss")

        axes[1].plot(np.arange(len(actor_losses)), actor_losses, color="firebrick")
        axes[1].set(title="Actor Loss")
        axes[1].set(ylabel="MSE Loss")
        axes[1].set(xlabel="Update Iteration")

        for axis in axes.flat:
            axis.grid(True)

        plt.savefig(self.algo_results + Directories.losses + f"losses-{date_str}.png")
        plt.clf()

        plt.plot(np.arange(len(is_list)), np.array(is_list),
                 label="Implementation Shortfall", color="cyan")
        plt.plot(np.arange(len(is_ma_list)), np.array(is_ma_list),
                 label=f"{self.ma_length} Day Moving Average Implementation Shortfall", color="magenta")
        plt.title("IS & Moving Average IS")
        plt.xlabel("Episode")
        plt.ylabel("Implementation Shortfall")
        plt.grid(True)
        plt.savefig(self.algo_results + Directories.is_ma + f"is-ma-{date_str}.png")
        plt.legend()
        plt.clf()

        plt.plot(np.arange(len(rewards_list)), np.array(rewards_list), color="lawngreen")
        plt.title("Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.savefig(self.algo_results + Directories.rewards + f"reward-{date_str}.png")
        plt.clf()

        self.analyze_analytical(date_str)
