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


def plot_learning_curve(x, scores, figure_file):
    ma_length = 100
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-ma_length):(i+1)])

    plt.plot(x, running_avg, label=f"{ma_length} Day Moving Average", color="darkgoldenrod")
    plt.grid()
    plt.legend()
    plt.title('Reward Plot vs. Episode')
    plt.savefig(figure_file)
