import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from src.constants import Directories
from src.environments.almgrenchriss_environment import AlmgrenChrissEnvironment
from datetime import datetime

from src.models.ddpg_models import DDPGCritic, DDPGActor


class CustomDDPG:
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

    def __init__(self, lr=0.3, batch_size=1024, M=200, criticLR=0.001, actorLR=0.001):
        """
        Initializes the parameters and the state, observation, and action vectors

        State:
        [S_t, M_t, L] (price of stock, inventory, # of trades left)

        Reward:
        0 for all entries except for the last entry
        R_final = E[x_n] - gamma / 2 * Var[x_n] ~ U(x_N)

        Observation:
        [S_t, M_t, L] - size = 3
        """
        # initializing a new environment
        self.environment = AlmgrenChrissEnvironment()

        # hyper-parameters
        self.lr = lr
        self.batch_size = batch_size
        self.checkpoint_frequency = 20
        # number of episodes
        self.M = M

        self.criticLR = criticLR
        self.actorLR = actorLR

        # plotting parameters
        # moving average length
        self.ma_length = 100
        self.inventory_sim_length = 300

        # action, reward, observation, and replay buffer initialization
        # reward
        self.R = None
        # action
        self.a = None
        # observation vectors
        self.observation = np.zeros(shape=3)
        self.observation[0] = self.environment.P_[0]
        self.observation[1] = self.environment.x[0]
        self.observation[2] = self.environment.L[0]

        # observation buffer vectors
        self.B_prev_obs = None
        self.B_action = None
        self.B_R = None
        self.B_obs = None

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

    def update_observation(self):
        """
        Updates the observation vector
        :return: None
        """
        self.observation[0] = self.environment.P_[self.environment.k-1]
        self.observation[1] = self.environment.x[self.environment.k-1]
        self.observation[2] = self.environment.L[self.environment.k-1]

    def step(self, a):
        """
        Steps the simulation forward using the action, a
        :param a: action
        :return: None
        """
        num_shares = a * self.environment.x[self.environment.k-1]
        self.environment.step(num_shares)
        self.update_observation()

    def compute_E(self):
        """
        Computes and returns the E value for the reward function:

        E = sum{k=1->N}(tau * x_k * gamma * n_k / tau) + sum{k=1->N}(n_k * h(n_k/tau))
        E = gamma * sum{k=1->N}(x_k * n_k) + sum{k=1->N}(n_k * h(n_k/tau))

        :return: float
        """
        E_1 = self.environment.gamma * sum(np.multiply(self.environment.x, self.environment.n))
        E_2 = sum(np.multiply(self.environment.n, self.environment.compute_h()))
        return E_1 + E_2

    def compute_V(self):
        """
        Computes and returns the V value for the reward function:

        V = sigma^2 * sum{k=1->N}(tau * x_k^2)
        V = sigma^2 * tau * sum{k=1->N}(x_k^2)

        :return: float
        """
        return np.square(self.environment.sigma) * self.environment.tau * sum(np.square(self.environment.x))

    def get_reward(self):
        """
        Returns the reward

        R = {0 if self.k != self.N-1
        E[x_n] - gamma / 2 * Var[x_n] otherwise}

        :return:
        """
        if self.environment.k == self.environment.N-1:
            return self.compute_E() - self.environment.gamma / 2 * self.compute_V()
        return 0

    def add_transition(self, prev_obs):
        """
        Saves an observation transition as numpy arrays in the self.B_ vectors
        :param prev_obs: the previous observation
        :return: None
        """
        B_prev_obs = prev_obs.reshape((1, -1))
        B_action = self.a.reshape((1, -1))
        B_R = self.R.reshape((1, -1))
        B_obs = self.observation.reshape((1, -1))

        if self.B_obs is None:
            self.B_prev_obs = B_prev_obs
            self.B_action = B_action
            self.B_R = B_R
            self.B_obs = B_obs
            return

        self.B_prev_obs = np.vstack((self.B_prev_obs, B_prev_obs))
        self.B_action = np.vstack((self.B_action, B_action))
        self.B_R = np.vstack((self.B_R, B_R))
        self.B_obs = np.vstack((self.B_obs, B_obs))

    def sample_transitions(self, N):
        """
        Samples N transitions from the replay buffer vectors (B_) and returns them as a tuple
        :param N: number of transitions to sample
        :return: None
        """
        transition_indices = np.random.choice(self.B_obs.shape[0], N)
        return self.B_prev_obs[transition_indices], self.B_action[transition_indices], self.B_R[transition_indices], \
               self.B_obs[transition_indices]

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
            for k in range(self.environment.N - 1):
                self.a = self.actor_target(torch.FloatTensor(self.observation)).detach().numpy()
                self.R = self.get_reward()
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
        plt.savefig(Directories.custom_ddpg_model_inv_results + f"inventory-{date_str}.png")

    def run_ddpg(self):
        """
        Runs the DDPG algorithm for M episodes using the parameters defined above
        :return: None
        """
        critic_losses = []
        actor_losses = []
        is_list = []
        is_ma_list = []

        for i in range(self.M):
            self.environment = AlmgrenChrissEnvironment()
            _init = False
            for k in range(self.environment.N - 1):
                observation_tensor = self.observation
                noise = np.random.normal(0, 0.1, 1)
                self.a = self.actor_target(torch.FloatTensor(observation_tensor)).detach().numpy() + noise
                prev_obs = self.observation
                self.R = np.array(self.get_reward())
                self.step(self.a)
                self.add_transition(prev_obs)

                if self.B_obs.shape[0] < self.batch_size:
                    continue

                prev_observations, actions, rewards, observations = self.sample_transitions(self.batch_size)

                # critic updates
                best_actions = self.actor_target(torch.FloatTensor(observations))
                y = torch.FloatTensor(rewards) + self.environment.gamma * \
                    self.critic_target(torch.FloatTensor(observations), best_actions)
                critic_loss = F.mse_loss(self.critic(torch.FloatTensor(observations), torch.FloatTensor(actions)), y)
                critic_losses.append(critic_loss.detach().numpy())
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # actor updates
                actor_predictions = self.actor_target(torch.FloatTensor(observations))
                # produces a Q-value that we wish to maximize
                actor_loss = -self.critic_target(torch.FloatTensor(observations), actor_predictions).mean()
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

        plt.savefig(Directories.custom_ddpg_loss_results + f"losses-{date_str}.png")
        plt.clf()

        a_thousand = 1000
        plt.plot(np.arange(len(is_list)), np.array(is_list) / a_thousand,
                 label="Implementation Shortfall", color="cyan")
        plt.plot(np.arange(len(is_ma_list)), np.array(is_ma_list) / a_thousand,
                 label=f"{self.ma_length} Day Moving Average Implementation Shortfall", color="magenta")
        plt.title("IS & Moving Average IS")
        plt.xlabel("Episode")
        plt.ylabel("Implementation Shortfall ($k)")
        plt.grid(True)
        plt.savefig(Directories.custom_ddpg_is_ma_results + f"is-ma-{date_str}.png")
        plt.clf()

        self.analyze_analytical(date_str)
