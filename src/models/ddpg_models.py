import torch


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

