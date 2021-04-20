import imageio
import os
import matplotlib.pyplot as plt
import numpy as np

from shared.constants import PPODirectories


def build_simulation_gif(filenames):
    """
    Builds a gif from the simulation data in the plotting folder and then deletes the simulation data
    :return: None
    """
    starting_frames = 5
    ending_frames = 10
    frames_per_image = 2

    def add_frames(writer_arg, image_arg, frames):
        for _ in range(frames):
            writer_arg.append_data(image_arg)

    with imageio.get_writer(PPODirectories.results + "training_simulations.gif", mode="I") as writer:
        for i in range(len(filenames)):
            image = imageio.imread(filenames[i])
            if i == 0:
                add_frames(writer, image, starting_frames)
            elif i == len(filenames)-1:
                add_frames(writer, image, ending_frames)
            add_frames(writer, image, frames_per_image)

    for filename in set(filenames):
        os.remove(filename)


def plot_multi_agent_rewards(rewards_matrix, figure_file):
    """
    Plots the rewards for all of the agents
    :param rewards_matrix: np.array (number of agents, number of episodes)
    :param figure_file: string
    :return: None
    """
    plt.figure(figsize=(12, 12))
    plt.title("Total Rewards vs. Episodes")
    for i in range(rewards_matrix.shape[0]):
        plt.plot(np.arange(len(rewards_matrix[i, :])), rewards_matrix[i, :], label=f"Agent {i+1}")
    plt.grid(True)
    plt.legend()
    plt.savefig(figure_file)
    plt.close()
