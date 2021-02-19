from src.classes.constant import ConstantStrategy
from src.classes.ddpg import DDPG

if __name__ == "__main__":
    ddpg = DDPG()
    ddpg.run_ddpg()

    print("Hello World!")
