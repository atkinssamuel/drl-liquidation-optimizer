import numpy as np
import os
import shutil

from src.constants import Directories, Algos

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


def clear_results(algo, clear=False):
    """
    Deletes the files in the directories in the results folders for the specified algorithm
    :param: algo: Algos.madrl/Algos.custom/etc.
    :param: clear: boolean
    :return: None
    """
    if algo == Algos.madrl and clear:
        delete_files_in_folder(Directories.madrl_results + Directories.losses)
        delete_files_in_folder(Directories.madrl_results + Directories.is_ma)
        delete_files_in_folder(Directories.madrl_results + Directories.model_inv)
        delete_files_in_folder(Directories.madrl_results + Directories.rewards)

    if algo == Algos.custom and clear:
        delete_files_in_folder(Directories.custom_results + Directories.losses)
        delete_files_in_folder(Directories.custom_results + Directories.is_ma)
        delete_files_in_folder(Directories.custom_results + Directories.model_inv)
        delete_files_in_folder(Directories.custom_results + Directories.rewards)



