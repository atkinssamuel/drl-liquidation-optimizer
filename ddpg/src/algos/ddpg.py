import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from shared.constants import DDPGDirectories
from ddpg.src.environments.almgrenchriss import AlmgrenChrissEnvironment
from datetime import datetime
from shared.helpers import ind, delete_files_in_folder
from shared.constants import DDPGAlgos
from ddpg.src.models.ddpg_models import DDPGCritic, DDPGActor


class DDPG:
    @staticmethod
    def clear_results(algo, clear=False):
        """
        Deletes the files in the directories in the results folders for the specified algorithm
        :param: algo: Algos.madrl/Algos.custom/etc.
        :param: clear: boolean
        :return: None
        """
        if algo == DDPGAlgos.madrl and clear:
            delete_files_in_folder(DDPGDirectories.madrl_results + DDPGDirectories.losses)
            delete_files_in_folder(DDPGDirectories.madrl_results + DDPGDirectories.is_ma)
            delete_files_in_folder(DDPGDirectories.madrl_results + DDPGDirectories.model_inv)
            delete_files_in_folder(DDPGDirectories.madrl_results + DDPGDirectories.rewards)

        if algo == DDPGAlgos.custom and clear:
            delete_files_in_folder(DDPGDirectories.custom_results + DDPGDirectories.losses)
            delete_files_in_folder(DDPGDirectories.custom_results + DDPGDirectories.is_ma)
            delete_files_in_folder(DDPGDirectories.custom_results + DDPGDirectories.model_inv)
            delete_files_in_folder(DDPGDirectories.custom_results + DDPGDirectories.rewards)

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
        Initializes the observation vector for the custom algo formulation
        :param environment: almgrenchriss environment instance
        :return: (3,) environment array
        """
        return np.array([environment.P_[0], environment.x[0], environment.L[0]])

    def __init__(self,
                 algo=DDPGAlgos.madrl,
                 D=5,
                 rho=0.99,
                 batch_size=1024,
                 discount_factor=0.99,
                 M=200,
                 critic_lr=0.001,
                 critic_weight_decay=0,
                 actor_lr=0.001,
                 replay_buffer_size=10000,
                 checkpoint_frequency=20,
                 inventory_sim_length=100,
                 pre_training_length=30,
                 post_training_length=30,
                 training_noise=0.1,
                 decay=True,
                 clear=False):
        """
        Initializes the parameters and the state, observation, and action vectors

        Indexing reference:
        - Arrays start at 0 but the way in which states are referenced starts at 1
        - Therefore, k will range from 1 -> N and arrays will be indexed by calling the ind(k) callback
            - k = 1 corresponds to the 0-th element in an array
        """
        # clearing results
        self.clear_results(algo=algo, clear=clear)

        # initializing a new environment
        self.environment = AlmgrenChrissEnvironment()

        # hyper-parameters
        self.D = D
        self.rho = rho
        self.batch_size = batch_size
        self.discount = discount_factor
        self.checkpoint_frequency = checkpoint_frequency
        # number of episodes
        self.M = M

        # algo type
        self.algo = algo
        self.observation = None
        if self.algo == DDPGAlgos.madrl:
            self.observation = self.init_madrl_obs(self.D)
            self.algo_results = DDPGDirectories.madrl_results
        elif self.algo == DDPGAlgos.custom:
            self.observation = self.init_custom_obs(self.environment)
            self.algo_results = DDPGDirectories.custom_results

        # reward
        self.R = None
        # action
        self.a = None

        self.critic_lr = critic_lr
        self.actor_lr = actor_lr

        # plotting parameters
        # moving average length
        self.ma_length              = 15
        self.inventory_sim_length   = inventory_sim_length
        self.pre_training_length    = pre_training_length
        self.post_training_length   = post_training_length
        self.training_noise         = training_noise
        if decay:
            self.training_noise_decay = self.training_noise/self.M
        else:
            self.training_noise_decay = decay

        # replay buffer: obs_i + a + r + obs_i+1
        self.replay_buffer_width = self.observation.shape[0] + 1 + 1 + self.observation.shape[0]
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = np.zeros(shape=(self.replay_buffer_size, self.replay_buffer_width))
        self.replay_buffer_index = 0
        self.replay_buffer_num_entries = 0

        # critic network initialization
        self.critic = DDPGCritic(self).apply(self.layer_init_callback)
        self.critic_target = DDPGCritic(self).apply(self.layer_init_callback)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, weight_decay=critic_weight_decay)

        # actor network initialization
        self.actor = DDPGActor(self).apply(self.layer_init_callback)
        self.actor_target = DDPGActor(self).apply(self.layer_init_callback)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

    def update_networks(self, current, target):
        """
        Updates the parameters in the target network using the parameters from the current network
        :param current: current network
        :param target: target network
        :return: None
        """
        for current_parameter, target_parameter in zip(current.parameters(), target.parameters()):
            target_parameter.data.copy_(self.rho * current_parameter.data * (1.0 - self.rho) * target_parameter.data)

    def get_r(self):
        """
        Slides the r vector left in the observation vector and sets the r_kth element in the observation vector to
        the log return at k-1
        :return: None
        """
        self.observation[:self.D] = self.observation[1:self.D + 1]
        self.observation[self.D] = np.log(self.environment.S_tilde[ind(self.environment.k)] /
                                          self.environment.S_tilde[ind(self.environment.k)-1])

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
        if self.algo == DDPGAlgos.custom:
            self.observation[0] = self.environment.P_[self.environment.k - 1]
            self.observation[1] = self.environment.x[self.environment.k - 1]
            self.observation[2] = self.environment.L[self.environment.k - 1]
        elif self.algo == DDPGAlgos.madrl:
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
        replay_buffer_entry = np.zeros(shape=(1, self.replay_buffer_width))

        replay_buffer_entry[0, :obs.shape[0]] = obs
        replay_buffer_entry[0, obs.shape[0]] = action
        replay_buffer_entry[0, obs.shape[0]+1] = reward
        replay_buffer_entry[0, obs.shape[0]+2:] = next_obs

        self.replay_buffer[self.replay_buffer_index, :] = replay_buffer_entry
        self.replay_buffer_index = (self.replay_buffer_index + 1) % self.replay_buffer_size
        self.replay_buffer_num_entries = max(self.replay_buffer_num_entries, self.replay_buffer_index)

        return

    def sample_transitions(self, N, L):
        """
        Samples N transitions from the first L replay buffer entries and returns them as appropriately sized obs,
        action, reward, next_obs torch tensors
        :param N: number of transitions to sample
        :param L: last non-zero entry in the replay buffer
        :return: obs, action, reward, next_obs
        """
        transition_indices = np.random.randint(L, size=N)
        sample = self.replay_buffer[transition_indices, :]

        obs = torch.FloatTensor(sample[:, :self.observation.shape[0]])
        action = torch.FloatTensor(sample[:, self.observation.shape[0]].reshape(-1, 1))
        reward = torch.FloatTensor(sample[:, self.observation.shape[0]+1].reshape(-1, 1))
        next_obs = torch.FloatTensor(sample[:, self.observation.shape[0]+2:])

        return obs, action, reward, next_obs

    def simulate_reward(self, length):
        """
        Simulates the total reward using the actor_target model
        :return: rewards np.array
        """
        rewards = []
        for i in range(length):
            self.environment = AlmgrenChrissEnvironment()
            total_reward = 0
            if self.algo == DDPGAlgos.madrl:
                self.observation = self.init_madrl_obs(self.D)
            elif self.algo == DDPGAlgos.custom:
                self.observation = self.init_custom_obs(self.environment)

            for k_ in range(self.environment.N-1):
                self.a = self.actor_target.forward(torch.FloatTensor(self.observation)).detach().numpy()
                _, _, reward, _ = self.step(self.a)
                total_reward += reward
            rewards.append(total_reward)
        return np.array(rewards)

    def analyze_analytical(self, date_str):
        """
        Simulates the trained model and compares the model's inventory process with the inventory process defined
        by the analytical solution. Also returns the average reward obtained by executing the optimal solution.

        q*_n = q_0 sinh(alpha * (T-t))/sinh(alpha * T)

        :return: float (average reward)
        """
        # noinspection PyUnresolvedReferences
        t = np.linspace(0, self.environment.T, self.environment.N)
        q_n = self.environment.x[0] * np.sinh(self.environment.kappa * (self.environment.T - t)) / \
              np.sinh(self.environment.kappa * self.environment.T)
        q_n_sim = None
        for i in range(self.inventory_sim_length):
            self.environment = AlmgrenChrissEnvironment()
            if self.algo == DDPGAlgos.madrl:
                self.observation = self.init_madrl_obs(self.D)
            elif self.algo == DDPGAlgos.custom:
                self.observation = self.init_custom_obs(self.environment)

            for k_ in range(self.environment.N-1):
                self.a = self.actor_target.forward(torch.FloatTensor(self.observation)).detach().numpy()
                _, _, reward, _ = self.step(self.a)
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
        plt.savefig(self.algo_results + DDPGDirectories.model_inv + f"inventory-{date_str}.png")
        plt.clf()

        rewards_list = []
        for i in range(self.inventory_sim_length):
            self.environment = AlmgrenChrissEnvironment()
            total_reward = 0
            for j in range(1, self.environment.N):
                t_j_ = (j - 1/2) * self.environment.tau
                n_j = 2 * np.sinh(1/2*self.environment.kappa * self.environment.tau)/\
                                np.sinh(self.environment.kappa*self.environment.T) *\
                                np.cosh(self.environment.kappa*(self.environment.T - t_j_)) * \
                                self.environment.x[0]
                total_reward += self.environment.step(n_j, reward_option=self.algo)
            rewards_list.append(total_reward)
        return np.average(np.array(rewards_list), axis=0)

    def run_ddpg(self):
        """
        Runs the DDPG algorithm for M episodes using the parameters defined above
        :return: None
        """
        critic_losses = []
        actor_losses = []
        is_list = []
        is_ma_list = []
        pre_training_rewards = self.simulate_reward(self.pre_training_length)
        training_rewards = []

        for i in range(self.M):
            self.environment = AlmgrenChrissEnvironment()
            total_reward = 0
            for k_ in range(self.environment.N-1):
                noise = np.random.normal(0, self.training_noise, 1)
                with torch.no_grad():
                    self.a = self.actor_target.forward(torch.FloatTensor(self.observation)).detach().numpy() + noise

                obs, action, reward, next_obs = self.step(self.a)
                self.add_transition(obs, action, reward, next_obs)
                total_reward += reward

                if self.replay_buffer_num_entries < self.batch_size:
                    continue

                obs, action, reward, next_obs = self.sample_transitions(self.batch_size, self.replay_buffer_num_entries)

                # critic updates
                best_action = self.actor_target.forward(obs)
                with torch.no_grad():
                    Q_next = self.critic_target.forward(next_obs, best_action)
                y = reward + self.discount * Q_next
                critic_loss = F.mse_loss(self.critic.forward(obs, action), y)
                critic_losses.append(critic_loss.detach().numpy())
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # actor updates
                actor_prediction = self.actor.forward(obs)
                # produces a Q-value that we wish to maximize
                Q_value = self.critic_target.forward(obs, actor_prediction)
                actor_loss = -Q_value.mean()
                actor_losses.append(actor_loss.detach().numpy())
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # update the target networks
                self.update_networks(self.critic, self.critic_target)
                self.update_networks(self.actor, self.actor_target)

            implementation_shortfall = self.environment.initial_market_price * \
                                       self.environment.X - self.environment.c[-1]

            is_list.append(implementation_shortfall)

            ma_starting_index = max(0, i - self.ma_length)
            is_ma_list.append(sum(is_list[ma_starting_index:i + 1]) / (i - ma_starting_index + 1))
            training_rewards.append(total_reward)

            if i % self.checkpoint_frequency == 0:
                print(f"Episode {i} ({round(i / self.M * 100, 2)}%), Implementation Shortfall = "
                      f"{implementation_shortfall}")

            self.training_noise = max(0, self.training_noise-self.training_noise_decay)

        training_rewards = np.array(training_rewards)

        # executing post-training simulation
        post_training_rewards = self.simulate_reward(self.post_training_length)

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

        plt.savefig(self.algo_results + DDPGDirectories.losses + f"losses-{date_str}.png")
        plt.clf()

        plt.plot(np.arange(len(is_list)), np.array(is_list),
                 label="Implementation Shortfall", color="cyan")
        plt.plot(np.arange(len(is_ma_list)), np.array(is_ma_list),
                 label=f"{self.ma_length} Day Moving Average Implementation Shortfall", color="magenta")
        plt.title("IS & Moving Average IS")
        plt.xlabel("Episode")
        plt.ylabel("Implementation Shortfall")
        plt.grid(True)
        plt.legend()
        plt.savefig(self.algo_results + DDPGDirectories.is_ma + f"is-ma-{date_str}.png")
        plt.clf()
        plt.close()

        analytical_average_reward = self.analyze_analytical(date_str)

        # reward plotting
        pre_train_ind = pre_training_rewards.shape[0]
        post_train_ind = pre_train_ind + training_rewards.shape[0]
        final_ind = post_train_ind + post_training_rewards.shape[0]
        # pre-training rewards
        plt.plot(np.arange(0, pre_train_ind), pre_training_rewards, label="Pre-Training Rewards", color="lawngreen")
        # training rewards
        plt.plot(np.arange(pre_train_ind-1, post_train_ind+1),
                 np.hstack((np.hstack((pre_training_rewards[-1], training_rewards)), post_training_rewards[0])),
                 label="Training Rewards", color="orange")
        # post-training rewards
        plt.plot(np.arange(post_train_ind, final_ind), post_training_rewards,
                 label="Post-Training Rewards", color="blue")
        # analytical reward average
        plt.plot(np.arange(final_ind), np.ones(shape=(final_ind,))*analytical_average_reward,
                 linestyle="dashed", color="brown",
                 label="Analytical Solution Average Reward")
        if self.algo == DDPGAlgos.madrl:
            plt.ylim(bottom=-7, top=-2)
        plt.title("Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.legend()
        plt.savefig(self.algo_results + DDPGDirectories.rewards + f"reward-{date_str}.png")
        plt.clf()
        plt.close()

