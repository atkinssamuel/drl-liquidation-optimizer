from ppo.environments.ma_almgrenchriss import MultiAgentAlmgrenChriss
from shared.constants import PPOAgent1Params, PPOAgent2Params, PPOTrainingParams, PPOAgent3Params

from ppo.src.agent import PPOAgent
from ppo.src.train import train_multi_agent_ppo


if __name__ == "__main__":
    agent_1_args = PPOAgent1Params()
    agent_2_args = PPOAgent2Params()
    agent_3_args = PPOAgent3Params()

    env = MultiAgentAlmgrenChriss(agent_1_args.D, agent_2_args.D, agent_3_args.D)

    agent_1 = PPOAgent(agent_1_args, env=env)
    agent_2 = PPOAgent(agent_2_args, env=env)
    agent_3 = PPOAgent(agent_3_args, env=env)

    ppo_training_params = PPOTrainingParams()
    train_multi_agent_ppo(agent_1, agent_2, agent_3, env=env, ppo_training_params=ppo_training_params)
