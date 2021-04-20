import numpy as np

from ppo.src.utils import build_simulation_gif, plot_multi_agent_rewards
from shared.constants import PPODirectories


def train_multi_agent_ppo(*agents, env=None, ppo_training_params=None):
    """
    Trains multiple PPO agents
    :param agents: multiple agent parameters
    :param env: environment object
    :param ppo_training_params: training parameter object
    :return: None
    """
    num_agents = len(agents)
    total_rewards = np.zeros(shape=(num_agents, ppo_training_params.episodes))

    learn_iterations = 0
    n_steps = 0

    filenames = []

    for i in range(ppo_training_params.episodes):
        observation = env.reset()
        dones = [False for _ in range(num_agents)]
        while sum(dones) < len(dones):
            multi_agent_action_dict = {
                'actions': [None]*num_agents,
                'dones': dones,
                'probabilities': [None]*num_agents,
                'values': [None]*num_agents,
                'agents': agents
            }
            observation_index = 0
            for a in range(num_agents):
                if dones[a]:
                    continue
                agent = agents[a]
                next_observation_index = observation_index + agent.D + 3
                agent_observation = observation[observation_index:next_observation_index]
                action, probability, value = agent.choose_action(agent_observation)

                # recording the action, probability, value, and current observation for each agent
                multi_agent_action_dict['actions'][a] = action
                multi_agent_action_dict['probabilities'][a] = probability
                multi_agent_action_dict['values'][a] = value

                observation_index = next_observation_index

            multi_agent_step_dict = env.step(multi_agent_action_dict)
            dones = multi_agent_step_dict['dones']
            observation_ = multi_agent_step_dict['state']

            n_steps += 1

            observation_index = 0
            for j in range(num_agents):
                agent = agents[j]
                next_observation_index = observation_index + agent.D + 3

                # extracting observation from state vector
                agent_observation = observation[observation_index:next_observation_index]

                action = multi_agent_action_dict['actions'][j]
                if action is None:
                    continue
                probability = multi_agent_action_dict['probabilities'][j]
                value = multi_agent_action_dict['values'][j]
                reward = multi_agent_step_dict['rewards'][j]
                done = multi_agent_step_dict['dones'][j]

                agent.remember(agent_observation, action, probability, value, reward, done)

                total_rewards[j][i] += reward
                observation_index = next_observation_index

            observation = observation_

            if n_steps % ppo_training_params.update_frequency == 0:
                for a in range(num_agents):
                    agents[a].learn()
                    learn_iterations += 1

        if i % ppo_training_params.checkpoint_frequency == 0:
            print(f"Episode {i}/{ppo_training_params.episodes}:" + " " * (6-len(str(i))), end="")
            for w in range(num_agents):
                agents[w].save_models()
                print('Reward for Agent {0} = {1:15,.2f}    '.format(w+1, round(total_rewards[w][i], 2)), end= "")
            print("")
            filenames.append(env.render(agents, i))

    plot_multi_agent_rewards(total_rewards, PPODirectories.results + ppo_training_params.reward_file_name)
    build_simulation_gif(filenames)

    return
