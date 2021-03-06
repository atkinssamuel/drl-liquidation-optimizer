import inspect

import numpy as np
import os
import shutil
import matplotlib.pyplot as plt


def sample_Xi(mu=0, sigma=1):
    """
    Samples the normal distribution
    :return: float [0, 1]
    """
    return np.random.normal(mu, sigma, 1)


def ind(k):
    """
    Returns k - 1 (to make array indexing more clear)
    :param k: integer
    :return: integer
    """
    return k-1


def delete_files_in_folder(folder):
    """
    Deletes all the files in a specified folder
    :param folder: string
    :return: None
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def plot_rewards(rewards, moving_average_length, figure_file):
    """
    Plots the reward as a function of the episode
    :param rewards: np.array
    :param moving_average_length: integer
    :param figure_file: string
    :return: None
    """
    rewards_ma = np.zeros(len(rewards))
    for i in range(len(rewards_ma)):
        rewards_ma[i] = np.mean(rewards[max(0, i-moving_average_length):(i+1)])

    plt.plot(np.arange(rewards_ma.shape[0]), rewards_ma, label=f"{moving_average_length} Day Moving Average", color="darkgoldenrod")
    plt.grid()
    plt.legend()
    plt.title('Moving Average Rewards Plot vs. Episodes')
    plt.savefig(figure_file)
    plt.close()


def copy_attributes(copy_from, copy_to):
    """
    Copies all of the attributes from the "copy_from" object to the "copy_to" object
    :param copy_from: object
    :param copy_to: object
    :return: None
    """
    for n, v in inspect.getmembers(copy_from):
        if "__" in n:
            continue
        setattr(copy_to, n, v)

