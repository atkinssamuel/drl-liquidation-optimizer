from src.ddpg import DDPG
from src.constants import DDPGHyperparameters

if __name__ == "__main__":
    ddpg = DDPG(D=DDPGHyperparameters.D, lr=DDPGHyperparameters.lr, batch_size=DDPGHyperparameters.batch_size,
                M=DDPGHyperparameters.M, criticLR=DDPGHyperparameters.criticLR, actorLR=DDPGHyperparameters.actorLR)
    ddpg.run_ddpg()
