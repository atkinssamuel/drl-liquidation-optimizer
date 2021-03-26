from src.algos.ddpg import DDPG
from src.constants import DDPGHyperparameters
from src.algos.custom_ddpg import CustomDDPG
from src.constants import CustomDDPGHyperparameters

if __name__ == "__main__":
    # ddpg = DDPG(D=DDPGHyperparameters.D, lr=DDPGHyperparameters.lr, batch_size=DDPGHyperparameters.batch_size,
    #             M=DDPGHyperparameters.M, criticLR=DDPGHyperparameters.criticLR, actorLR=DDPGHyperparameters.actorLR)
    # ddpg.run_ddpg()

    custom_ddpg = CustomDDPG(lr=CustomDDPGHyperparameters.lr, batch_size=CustomDDPGHyperparameters.batch_size,
                             M=CustomDDPGHyperparameters.M, criticLR=CustomDDPGHyperparameters.criticLR,
                             actorLR=CustomDDPGHyperparameters.actorLR)
    custom_ddpg.run_ddpg()
