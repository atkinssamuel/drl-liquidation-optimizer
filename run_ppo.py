from ppo.environments.sa_almgrenchriss import SingleAgentAlmgrenChriss
from shared.constants import PPOEnvParams, PPOAgentParams

from ppo.src.agent import PPOAgent
from ppo.src.train import train_ppo

if __name__ == "__main__":
    env = SingleAgentAlmgrenChriss(
        D = PPOEnvParams.D
    )

    agent = PPOAgent(
        batch_size      = PPOAgentParams.batch_size,
        alpha           = PPOAgentParams.alpha,
        epochs          = PPOAgentParams.epochs,
        input_dims      = PPOEnvParams.D+3,
        gae_lambda      = PPOAgentParams.gae_lambda,
        policy_clip     = PPOAgentParams.policy_clip,
        gamma           = PPOAgentParams.gamma
    )

    train_ppo(agent, env,
              episodes          = PPOAgentParams.episodes,
              update_frequency  = PPOAgentParams.update_frequency
              )
