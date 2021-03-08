import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from src.classes.environments import AlmgrenChrissEnvironment
from datetime import datetime

class DDPGActor(torch.nn.Module):
    def __init__(self, DDPG):
        super(DDPGActor, self).__init__()
        # input = observation
        # observation = [rk - 2, rk - 1, rk, mk, lk] -> size(observation) = (D + 1) + 1 + 1 = D + 3
        input_size = DDPG.D + 3
        hidden_layer_size = 2 * input_size
        # output = action
        output_size = 1

        self.fc1 = torch.nn.Linear(input_size, hidden_layer_size)
        self.fc2 = torch.nn.Linear(hidden_layer_size, output_size)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class DDPGCritic(torch.nn.Module):
    def __init__(self, DDPG):
        super(DDPGCritic, self).__init__()
        # input = observation, action
        # observation = [rk-D, ..., rk, mk, lk]
        # observation example for D = 2: [rk - 2, rk - 1, rk, mk, lk] -> size(observation) = (D + 1) + 1 + 1 = D + 3
        # size(observation) = (DDPG.D + 3)
        input_size = (DDPG.D + 3)
        hidden_layer_size = 2 * input_size
        hidden_layer_2_size = (hidden_layer_size + 1) * 2
        # output = Q-value for observation-action pair
        output_size = 1

        self.fc1 = torch.nn.Linear(input_size, hidden_layer_size)
        self.fc2 = torch.nn.Linear(hidden_layer_size + 1, hidden_layer_2_size)
        self.fc3 = torch.nn.Linear(hidden_layer_2_size, output_size)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, state, action):
        x1 = self.relu(self.fc1(state))
        x = torch.cat((x1, action), dim=1)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


class DDPG:
    @staticmethod
    def layer_init_callback(layer):
        if type(layer) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def __init__(self):
        self.environment = AlmgrenChrissEnvironment()
        self.D = 5
        self.lr = 0.3
        self.a = None
        self.observation = np.zeros(shape=(self.D + 3))
        self.observation[self.D + 1:] = [1, 1]
        self.B_prev_obs = None
        self.B_action = None
        self.B_R = None
        self.B_obs = None
        self.batch_size = 256
        self.M = 10
        self.R = 0

        self.criticLR = 0.000001
        self.critic = DDPGCritic(self).apply(self.layer_init_callback)
        self.critic_target = DDPGCritic(self).apply(self.layer_init_callback)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.criticLR)

        self.actorLR = 0.000001
        self.actor = DDPGActor(self).apply(self.layer_init_callback)
        self.actor_target = DDPGActor(self).apply(self.layer_init_callback)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actorLR)

    def update_networks(self, current, target):
        for current_parameter, target_parameter in zip(current.parameters(), target.parameters()):
            target_parameter.data.copy_(self.lr * current_parameter.data * (1.0 - self.lr) * target_parameter.data)

    def get_r(self):
        self.observation[:self.D] = self.observation[1:self.D + 1]
        self.observation[self.D] = np.log(self.environment.P[self.environment.k - 1] /
                                          self.environment.P[self.environment.k - 2])

    def get_m(self):
        self.observation[-2] = (self.environment.N - (
                    self.environment.k - 1) * self.environment.tau) / self.environment.N

    def get_l(self):
        self.observation[-1] = self.environment.x[self.environment.k - 1] / self.environment.X

    def update_observation(self):
        self.get_r()
        self.get_m()
        self.get_l()

    def step(self, a):
        num_shares = a * self.environment.x[self.environment.k - 1]
        self.environment.step(num_shares)
        self.update_observation()

    def compute_h(self):
        return self.environment.epsilon * np.sign(self.environment.n) + self.environment.eta / self.environment.tau \
               * self.environment.n

    def compute_V(self):
        return np.square(self.environment.sigma) * \
               self.environment.tau * sum(np.square(self.environment.x))

    def compute_E(self):
        E_1 = self.environment.gamma * sum(np.multiply(self.environment.x, self.environment.n))
        h = self.compute_h()
        E_2 = sum(np.multiply(self.environment.n, h))
        return E_1 + E_2

    def compute_U(self):
        return self.compute_E() + self.environment.lam * self.compute_V()

    def get_reward(self):
        self.R = self.R - self.compute_U()

    def test_implementation(self):
        for k in range(self.environment.N - 1):
            self.step(0.5)
        self.environment.plot_simulation()

    def add_transition(self, prev_obs):
        B_prev_obs = torch.FloatTensor(prev_obs).reshape((1, -1))
        B_action = self.a.reshape((1, -1))
        B_R = torch.FloatTensor([self.R]).reshape((1, -1))
        B_obs = torch.FloatTensor(self.observation).reshape((1, -1))

        if self.B_obs is None:
            self.B_prev_obs = B_prev_obs
            self.B_action = B_action
            self.B_R = B_R
            self.B_obs = B_obs
            return

        self.B_prev_obs = torch.cat((self.B_prev_obs, B_prev_obs), 0)
        self.B_action = torch.cat((self.B_action, B_action), 0)
        self.B_R = torch.cat((self.B_R, B_R), 0)
        self.B_obs = torch.cat((self.B_obs, B_obs), 0)

    def sample_transitions(self, N):
        transition_indices = np.random.choice(self.B_obs.shape[0], N)
        return self.B_prev_obs[transition_indices], self.B_action[transition_indices], self.B_R[transition_indices], \
               self.B_obs[transition_indices]

    def run_ddpg(self):
        critic_losses = []
        actor_losses = []
        is_list = []
        for i in range(self.M):
            self.environment = AlmgrenChrissEnvironment()
            for k in range(self.environment.N - 1):
                observation_tensor = torch.FloatTensor(self.observation)
                noise = torch.FloatTensor(np.random.normal(0, 0.1, 1))
                self.a = self.actor_target(observation_tensor) + noise
                prev_obs = self.observation
                self.get_reward()
                self.step(self.a)
                self.add_transition(prev_obs)

                if self.B_obs.shape[0] < self.batch_size:
                    continue

                prev_observations, actions, rewards, observations = self.sample_transitions(self.batch_size)

                # critic updates
                best_actions = self.actor_target(observations)
                y = rewards + self.environment.gamma * self.critic_target(observations, best_actions)
                critic_loss = F.mse_loss(self.critic(observations, actions), y)
                critic_losses.append(critic_loss.detach().numpy())
                self.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critic_optimizer.step()

                # actor updates
                actor_predictions = self.actor_target(observations)
                actor_loss = -self.critic(observations, actor_predictions).mean()
                actor_losses.append(actor_loss.detach().numpy())
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.update_networks(self.critic, self.critic_target)
                self.update_networks(self.actor, self.actor_target)

            print(f"Episode {i} ({round(i/self.M*100, 2)}%)")

            implementation_shortfall = self.environment.c[-1] - self.environment.initial_market_price * \
                                       self.environment.X
            is_list.append(implementation_shortfall)
            print(f"Implementation Shortfall = {implementation_shortfall}\n")

        date_str = str(datetime.now())[2:10] + "_" + str(datetime.now())[11:13] + "-" + str(datetime.now())[14:16]

        fig, axes = plt.subplots(2, figsize=(14, 10))

        axes[0].plot(np.arange(len(critic_losses)), critic_losses)
        axes[0].set(title="Critic Loss")
        axes[0].set(ylabel="MSE Loss")

        axes[1].plot(np.arange(len(actor_losses)), actor_losses)
        axes[1].set(title="Actor Loss")
        axes[1].set(ylabel="MSE Loss")
        axes[1].set(xlabel="Update Iteration")

        for axis in axes.flat:
            axis.grid(True)
        plt.savefig(f"../results/losses/losses-{date_str}.png")
        plt.clf()

        a_million = 1000000
        plt.plot(np.arange(len(is_list)), np.array(is_list)/a_million)
        plt.title("Implementation Shortfall")
        plt.xlabel("Episode")
        plt.ylabel("Implementation Shortfall ($M)")
        plt.grid(True)
        plt.savefig(f"../results/implementation-shortfall/implementation_shortfall-{date_str}.png")
