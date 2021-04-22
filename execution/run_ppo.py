from ppo.environments.ma_almgrenchriss import MultiAgentAlmgrenChriss
from ppo.src.evaluate import evaluate_agents
from shared.constants import PPOAgent1Params, PPOAgent2Params, PPOTrainingParams, PPOAgent3Params, PPOEnvParams

from ppo.src.agent import PPOAgent
from ppo.src.train import train_multi_agent_ppo


if __name__ == "__main__":
    agent_1_args = PPOAgent1Params()
    agent_2_args = PPOAgent2Params()
    agent_3_args = PPOAgent3Params()
    ppo_training_params = PPOTrainingParams()
    env_params = PPOEnvParams()

    env = MultiAgentAlmgrenChriss(agent_1_args, env_params=env_params)

    agent_1 = PPOAgent(agent_1_args, env=env)
    agent_2 = PPOAgent(agent_2_args, env=env)
    agent_3 = PPOAgent(agent_3_args, env=env)

    train_multi_agent_ppo(agent_1, env=env, ppo_training_params=ppo_training_params)

    evaluate_agents(agent_1, env=env)
