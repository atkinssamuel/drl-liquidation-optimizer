import numpy as np
import torch
import math

from ppo.src.models import Actor, Critic
from ppo.src.memory import PPOMemory
from shared.shared_utils import copy_attributes, ind


class PPOAgent:
    def __init__(self, agent_args_object, env=None):

        copy_attributes(agent_args_object, self)

        self.N = env.N
        self.k = 1

        # inventory
        self.x = np.zeros(shape=(self.N,))
        self.x[0] = self.X

        # n_k-1 is the number of shares sold at k-1
        self.n = np.zeros(shape=(self.N,))

        # revenue
        self.R = np.zeros(shape=(self.N,))

        # normalized number of remaining trades [0, 1]
        self.L = 1

        self.input_dims = self.D + 3

        self.actor = Actor(self.input_dims, self.alpha)
        self.critic = Critic(self.input_dims, self.alpha)
        self.memory = PPOMemory(self.batch_size)

    def reset(self):
        # inventory
        self.x = np.zeros(shape=(self.N,))
        self.x[0] = self.X

        # n_k-1 is the number of shares sold at k-1
        self.n = np.zeros(shape=(self.N,))

        # revenue
        self.R = np.zeros(shape=(self.N,))

        # normalized number of remaining trades [0, 1]
        self.L = 1

    def step_inventory(self, k):
        """
        Steps the inventory forward:

        X_k = X_k-1 - n_k-1

        :param k: k + 1
        :return: None
        """
        self.x[ind(k)] = self.x[ind(k)-1] - self.n[ind(k)-1]

    def step_revenue(self, k, new_revenue):
        """
        Steps the revenue process forward:

        C_k = C_k-1 + new_revenue

        :param k: k + 1
        :return: None
        """
        self.R[ind(k)] = self.R[ind(k) - 1] + new_revenue

    def step_trades(self, k):
        """
        Steps the normalized number of trades left forward

        L_k = L_k-1 - 1 / N

        :param k: k + 1
        :return: None
        """
        self.L -= 1 / self.N

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)

        mu, var, _ = self.actor(state)

        sigma = torch.sqrt(var)
        action = torch.normal(mu, sigma)
        action = torch.clamp(action, 0, 1)

        p1 = -((mu - action) ** 2) / (2 * var.clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2 * math.pi * var))
        prob = torch.exp(p1 + p2)

        value = self.critic(state)
        value = torch.squeeze(value).item()

        return action, prob, value

    def learn(self):
        for _ in range(self.epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            values = torch.tensor(values).to(self.actor.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                action = torch.tensor(action_arr[batch]).to(self.actor.device)

                mu, var, _ = self.actor(states)

                p1 = -((mu - action) ** 2) / (2 * var.clamp(min=1e-3))
                p2 = - torch.log(torch.sqrt(2 * math.pi * var))
                new_probabilities = p1 + p2

                prob_ratio = new_probabilities.exp() / old_probs.exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                     1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_value = torch.squeeze(self.critic(states))
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
