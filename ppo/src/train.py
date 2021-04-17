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
            agent.save_models()

        if i % checkpoint_frequency == 0:
            print(checkpoint_message)

    if discrete:
        reward_file = PPODirectories.discrete_rewards + reward_file_name
    else:
        reward_file = PPODirectories.rewards + reward_file_name

    plot_learning_curve(total_rewards, moving_average_length, reward_file)

    return best_reward_average
