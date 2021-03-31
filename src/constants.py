class Directories:
    """
    Centralized reference for the names of the directories
    """
    results = "results/"

    # madrl results
    madrl_results = results + "madrl/"

    # custom ddpg results
    custom_results = results + "custom/"

    # results directories
    losses = "losses/"
    is_ma = "is-ma/"
    model_inv = "model-inv/"
    rewards = "rewards/"


class Algos:
    madrl = "madrl"
    custom = "custom"


class DDPGHyperparameters:
    algo                    = Algos.custom
    clear                   = True
    D                       = 5
    lr                      = 0.1
    discount                = 0.99
    batch_size              = 1024
    M                       = 60
    criticLR                = 0.01
    actorLR                 = 0.01
    checkpoint_frequency    = 20


