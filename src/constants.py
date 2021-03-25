class Directories:
    """
    Centralized reference for the names of the directories
    """
    # ddpg results
    ddpg_results = "ddpg-results/"
    ddpg_is_results = ddpg_results + "is/"
    ddpg_loss_results = ddpg_results + "losses/"
    ddpg_is_ma_results = ddpg_results + "is-ma/"
    ddpg_reward_results = ddpg_results + "rewards/"
    ddpg_sim = ddpg_results + "sim/"


class DDPGHyperparameters:
    D = 5
    lr = 0.3
    batch_size = 1024
    M = 200
    criticLR = 0.01
    actorLR = 0.01
