import numpy as np

from shared.constants import PPODirectories
from shared.shared_utils import plot_learning_curve


def train_ppo(agent, env, episodes=500, update_frequency=20, moving_average_length=50, checkpoint_frequency=20,
              reward_file_name="rewards.png", discrete=False):

    best_reward_average = env.reward_range[0]
    total_rewards = []

    learn_iterations = 0
    n_steps = 0

    for i in range(episodes):
        observation = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            total_reward += reward
            agent.remember(observation, action, prob, val, reward, done)

            if n_steps % update_frequency == 0:
                agent.learn()
                learn_iterations += 1

            observation = observation_
        total_rewards.append(total_reward)
        average_reward = np.mean(total_rewards[-moving_average_length:])

        checkpoint_message = f"Episode {i}, Reward = {total_reward}, " \
                             f"{moving_average_length}-Episode Average Reward = {average_reward}, " \
                             f"Time Steps = {n_steps}, " \
                             f"Learning Steps = {learn_iterations}"

        if average_reward > best_reward_average:
            print(f"{checkpoint_message} (saving new optimal model)")
            best_reward_average = average_reward
            env.render()
            agent.save_models()

        if i % checkpoint_frequency == 0:
            print(checkpoint_message)

    if discrete:
        reward_file = PPODirectories.discrete_rewards + reward_file_name
    else:
        reward_file = PPODirectories.rewards + reward_file_name

    plot_learning_curve(total_rewards, moving_average_length, reward_file)

    return best_reward_average


def train_multi_agent_ppo(*agents, env=None, ppo_training_params=None):
    num_agents = len(agents)
    rewards = np.zeros(shape=(num_agents, ppo_training_params.episodes))
    total_rewards = np.zeros(shape=num_agents)

    learn_iterations = 0
    n_steps = 0

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

                rewards[j][i] = reward
                total_rewards[j] += reward
                observation_index = next_observation_index

            observation = observation_

            if n_steps % ppo_training_params.update_frequency == 0:
                for a in range(num_agents):
                    agents[a].learn()
                    learn_iterations += 1

        if i % ppo_training_params.checkpoint_frequency == 0:
            spacing_index = 0
            spacing_increment = 40
            spacing_index += spacing_increment
            print(f"Episode {i}/{ppo_training_params.episodes}:" + " " * (6-len(str(i))), end="")
            for w in range(num_agents):
                print('Reward for Agent {0} = {1:14,.2f}    '.format(w+1, round(rewards[w][i], 2)), end= "")
            print("")
    return
