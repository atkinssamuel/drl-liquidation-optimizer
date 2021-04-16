class DDPGDirectories:
    """
    Centralized reference for the names of the directories
    """
    ddpg = "ddpg/"
    results = ddpg + "results/"

    # madrl results
    madrl_results = results + "madrl/"

    # custom ddpg results
    custom_results = results + "custom/"

    # results directories
    losses = "losses/"
    is_ma = "is-ma/"
    model_inv = "model-inv/"
    rewards = "rewards/"


class DDPGAlgos:
    """
    Algo types
    """
    madrl = "madrl"
    custom = "custom"


class DDPGParams:
    """
    Contains all of the hyper-parameters for the DDPG algorithm
    """
    algo                    = DDPGAlgos.madrl   # algorithm to execute (Algos.madrl, Algos.custom, etc.)
    clear                   = True              # boolean that determines if we clear the DDPG results folder
    D                       = 5                 # number of previous days in madrl formulation to keep track of
    rho                     = 0.03              # parameter soft-update factor
    discount                = 0.99              # discount factor in critic loss
    batch_size              = 512               # batch size
    M                       = 200               # number of episodes
    critic_lr               = 0.00001
    critic_weight_decay     = 0
    actor_lr                = 0.00001
    replay_buffer_size      = 10000
    checkpoint_frequency    = 20
    inventory_sim_length    = 100               # length of post-training inventory simulation
    pre_training_length     = 30                # length of pre-training reward simulation
    post_training_length    = 30                # length of post-training reward simulation
    training_noise          = 0.1               # initial training noise
    decay                   = True              # True: gradient decay, float: manually set


class PPODirectories:
    """
    Contains all of the directories for the PPO implementation
    """
    ppo = "ppo/"
    results = ppo + "results/"
    rewards = results + "rewards/"
    models = ppo + "models/"


class PPOEnvParams:
    """
    Contains all of the hyper-parameters for the environment used by the PPO algorithm
    """
    clear                   = True          # whether to clear the results in the algorithm's directory before running
    D                       = 5             # number of previous time instances to consider in the observation space


class PPOAgentParams:
    """
    Contains all of the hyper-parameters for the PPO agent(s)
    """
    alpha                   = 0.0003    #
    epochs                  = 4         #
    batch_size              = 5         #
    gae_lambda              = 0.95      #
    policy_clip             = 0.2       #
    gamma                   = 0.99      #
    episodes                = 500       #
    update_frequency        = 20        #
