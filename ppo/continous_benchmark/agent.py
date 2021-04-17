import numpy as np
import torch
import math

from ppo.src.models import Actor, Critic
from ppo.src.memory import PPOMemory


class PPOAgent:
    def __init__(self,
                 batch_size = 5,
                 alpha = 0.0003,
                 epochs = 4,
                 input_dims = 8,
                 gae_lambda = 0.95,
                 policy_clip = 0.2,
                 gamma = 0.99
                 ):

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.epochs = epochs
        self.gae_lambda = gae_lambda

        self.actor = Actor(input_dims, alpha)
        self.critic = Critic(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)

        mu, var, _ = self.actor(state)

        sigma = torch.sqrt(var)
        action = torch.normal(mu, sigma)
        action = torch.clamp(action, -1, 1)

        p1 = -((mu - action) ** 2)/(2 * var.clamp(min=1e-3))
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