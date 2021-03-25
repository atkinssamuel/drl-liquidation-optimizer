class Directories:
    """
    Centralized reference for the names of the directories
    """
    results = "results/"

    # ddpg results
    ddpg_results = results + "ddpg/"
    ddpg_loss_results = ddpg_results + "losses/"
    ddpg_is_ma_results = ddpg_results + "is-ma/"
    ddpg_model_inv_results = ddpg_results + "model-inv/"

    # custom ddpg results
    custom_ddpg_results = results + "custom-ddpg/"
    custom_ddpg_loss_results = custom_ddpg_results + "losses/"
    custom_ddpg_is_ma_results = custom_ddpg_results + "is-ma/"
    custom_ddpg_model_inv_results = custom_ddpg_results + "model-inv/"


class DDPGHyperparameters:
    D = 5
    lr = 0.3
    batch_size = 1024
    M = 400
    criticLR = 0.01
    actorLR = 0.01


class CustomDDPGHyperparameters:
    lr = 0.3
    batch_size = 1024
    M = 200
    criticLR = 0.01
    actorLR = 0.01

