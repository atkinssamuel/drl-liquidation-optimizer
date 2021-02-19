import numpy as np
import torch

from src.classes.environment import TradingEnvironment

class DDPGActor(torch.nn.Module):
    def __init__(self, environment):
        super(DDPGActor, self).__init__()

        # input = observation
        # observation = [r1, ..., rk, mk, lk] = k + 2 -> max(size(observation)) = N + 2
        input_size = environment.N + 2
        hidden_layer_size = 2 * input_size
        hidden_layer_size_2 = 2 * hidden_layer_size
        # output = action
        output_size = 1

        self.fc1 = torch.nn.Linear(input_size, hidden_layer_size)
        self.fc2 = torch.nn.Linear(hidden_layer_size, hidden_layer_size_2)
        self.fc3 = torch.nn.Linear(hidden_layer_size_2, output_size)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class DDPGCritic(torch.nn.Module):
    def __init__(self, environment):
        super(DDPGCritic, self).__init__()
        # input = observation, action
        # observation = [r1, ..., rk, mk, lk] = k + 2 -> max(size(observation)) = N + 2
        input_size = (environment.N + 2) + 1
        hidden_layer_size = 2 * input_size
        hidden_layer_size_2 = 2 * hidden_layer_size
        # output = Q-value for observation-action pair
        output_size = 1

        self.fc1 = torch.nn.Linear(input_size, hidden_layer_size)
        self.fc2 = torch.nn.Linear(hidden_layer_size, hidden_layer_size_2)
        self.fc3 = torch.nn.Linear(hidden_layer_size_2, output_size)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class DDPG:
    @staticmethod
    def layer_init_callback(layer):
        if type(layer) == torch.nn.Linear:
            torch.nn.init.xavier_uniform(layer.weight)
            layer.bias.data.fill_(0.01)

    def __init__(self):
        self.environment = TradingEnvironment()
        self.lr = 0.3
        self.r = []
        self.m = 1
        self.l = 1
        self.a = None
        self.observation = np.zeros(shape=(self.environment.N + 2,))
        self.B = []
        self.M = 100

        self.critic = DDPGCritic(self.environment).apply(self.layer_init_callback)
        self.target = DDPGCritic(self.environment).apply(self.layer_init_callback)

        self.actor = DDPGActor(self.environment).apply(self.layer_init_callback)

    def update_r(self):
        self.r.append(np.log(self.environment.P[self.environment.k] / self.environment.P[self.environment.k - 1]))

    def update_m(self):
        self.m = (self.environment.k * self.environment.tau) / self.environment.N

    def update_l(self):
        self.l = self.environment.x[self.environment.k] / self.environment.X

    def update_state(self):
        self.update_r()
        self.update_m()
        self.update_l()

    def get_observation(self):
        self.observation = [self.r, self.m, self.l]

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

    def compute_reward(self):
        return



    def run_ddpg(self):
        for i in range(self.M):
            self.environment = TradingEnvironment()
            for k in range(self.environment.N):
                self.get_observation()
                # action = self.actor()
        print("Hello World")
