from ppo.environments.sa_almgrenchriss import SingleAgentAlmgrenChriss
from ppo.continous_benchmark.agent import PPOAgent
from ppo.continous_benchmark.train import train_ppo

import gym



class PPOAgentParams:
    """
    Contains all of the hyper-parameters for the PPO agent(s)
    """
    alpha                   = 0.0003
    epochs                  = 10
    batch_size              = 20
    gae_lambda              = 0.95
    policy_clip             = 0.2
    gamma                   = 0.99
    episodes                = 50
    update_frequency        = 20


if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0")

    agent = PPOAgent(
        batch_size      = PPOAgentParams.batch_size,
        alpha           = PPOAgentParams.alpha,
        epochs          = PPOAgentParams.epochs,
        input_dims      = 2,
        gae_lambda      = PPOAgentParams.gae_lambda,
        policy_clip     = PPOAgentParams.policy_clip,
        gamma           = PPOAgentParams.gamma
    )

    train_ppo(agent, env,
              episodes          = PPOAgentParams.episodes,
              update_frequency  = PPOAgentParams.update_frequency
              )
