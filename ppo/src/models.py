import torch
import os

from shared.constants import PPODirectories


class Actor(torch.nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, agent_number=1):
        super(Actor, self).__init__()

        self.base = torch.nn.Sequential(
            torch.nn.Linear(input_dims, fc1_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(fc1_dims, fc2_dims),
            torch.nn.ReLU()
        )
        self.mu = torch.nn.Sequential(
            torch.nn.Linear(fc2_dims, 1),
            torch.nn.Sigmoid()
        )
        self.var = torch.nn.Sequential(
            torch.nn.Linear(fc2_dims, 1),
            torch.nn.Softplus()
        )
        self.value = torch.nn.Linear(fc2_dims, 1)

        self.checkpoint_file = os.path.join(PPODirectories.models, f'agent_{agent_number}_actor')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        base = self.base(state)
        return self.mu(base), self.var(base), self.value(base)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Critic(torch.nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, agent_number=1):
        super(Critic, self).__init__()

        self.checkpoint_file = os.path.join(PPODirectories.models, f'agent_{agent_number}_critic')
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(input_dims, fc1_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(fc1_dims, fc2_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(fc2_dims, 1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

