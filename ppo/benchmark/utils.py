import numpy as np
import matplotlib.pyplot as plt


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
