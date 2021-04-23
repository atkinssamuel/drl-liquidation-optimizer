import numpy as np
from matplotlib import pyplot as plt

from shared.constants import PPODirectories


def get_sa_analytical_solution(agent, env=None):
    """
    Extracts the analytical solution for the single agent scenario

    sigma(t) = (x_1(0) + x_2(0)) exp(-gamma * t / (6 * lam)) * s1(t) / s2(t)
    s1(t) = sinh((T - t) sqrt(gamma^2 + 12 * alpha * lam * sigma^2) / (6 * lam))
    s2(t) = sinh((T * sqrt(gamma^2 + 12 * alpha * lam * sigma^2) / (6 * lam))

    delta(t) = (x_1(0) - x_2(0)) exp (gamma * t / (2 * lam)) * s3 / s4
    s3(t) = sinh((T - t) * sqrt(gamma^2 + 4 * alpha * lam * sigma^2) / (2 * lam))
    s4(t) = sinh(T * sqrt(gamma^2 + 4 * alpha * lam * sigma^2) / (2 * lam))


    X_1^*(t) = 1 / 2 * (sigma(t) + delta(t))
    X_2^*(t) = 1 / 2 * (sigma(t) - delta(t))

    :return: np.array inventory process
    """
    t = np.linspace(0, env.T, env.N)

    eta_tilde = env.eta * (1 - env.gamma * env.tau / (2 * env.eta))
    kappa_2_tilde = agent.risk_aversion * env.sigma ** 2 / eta_tilde
    kappa = np.arccosh(kappa_2_tilde * env.tau ** 2 / 2 + 1) / env.tau

    x_analytical = agent.x[0] * np.sinh(kappa * (env.T - t)) / np.sinh(kappa * env.T)

    return x_analytical


def evaluate_agents(*agents, env=None):
    """
    Evaluates the provided agents by comparing their solutions with the true analytical solution(s)
    :param agents: agent_1, agent_2, ...
    :param env: environment object
    :return: None
    """
    num_agents = len(agents)
    num_simulations = 200
    inventory_averages = []
    for agent in agents:
        inventory = []
        agent.load_models()
        next_observation_index = agent.D + 3
        for i in range(num_simulations):
            observation = env.reset()
            done = False
            while not done:
                action_dict = {
                    'actions': [None],
                    'dones': [done],
                    'probabilities': [None],
                    'values': [None],
                    'agents': [agent]
                }

                agent_observation = observation[:next_observation_index]
                action, probability, value = agent.choose_action(agent_observation, noise=False)

                # recording the action, probability, value, and current observation for each agent
                action_dict['actions'][0] = action
                action_dict['probabilities'][0] = probability
                action_dict['values'][0] = value

                step_dict = env.step(action_dict)
                done = step_dict['dones'][0]
                observation_ = step_dict['state']

                observation = observation_
            inventory.append(agent.x)
        inventory_average = np.average(np.array(inventory), axis=0)
        inventory_averages.append(inventory_average)

    if num_agents == 1:
        analytical_solution = get_sa_analytical_solution(agents[0], env=env)

        plt.figure(figsize=(12, 12))
        plt.title(f"Analytical Comparison for Single Agent")
        plt.plot(np.arange(len(inventory_averages[0])), inventory_averages[0], label="Simulated Inventory Process")
        plt.plot(np.arange(len(analytical_solution)), analytical_solution, label="Analytical Solution")

        plt.grid()
        plt.legend()
        plt.savefig(PPODirectories.results + f"single_agent_analytical_comparison.png")
        plt.close()
