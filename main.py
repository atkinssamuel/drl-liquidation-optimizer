from src.algos.ddpg import DDPG
from src.constants import DDPGHyperparameters
from src.helpers import clear_results

if __name__ == "__main__":
    clear_results(algo=DDPGHyperparameters.algo, clear=DDPGHyperparameters.clear)

    ddpg = DDPG(algo=DDPGHyperparameters.algo, D=DDPGHyperparameters.D, lr=DDPGHyperparameters.lr,
                batch_size=DDPGHyperparameters.batch_size, discount_factor=DDPGHyperparameters.discount,
                M=DDPGHyperparameters.M, criticLR=DDPGHyperparameters.criticLR, actorLR=DDPGHyperparameters.actorLR,
                checkpoint_frequency=DDPGHyperparameters.checkpoint_frequency)
    ddpg.run_ddpg()

