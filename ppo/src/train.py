import numpy as np

from shared.constants import PPODirectories
from shared.shared_utils import plot_learning_curve


def train_ppo(agent, env, episodes=500, update_frequency=20):

    best_score = env.reward_range[0]
    score_history = []

    learn_iterations = 0
    n_steps = 0

    for i in range(episodes):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % update_frequency == 0:
                agent.learn()
                learn_iterations += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iterations)
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, PPODirectories.rewards + "rewards.png")