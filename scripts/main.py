from src.classes.ddpg import DDPG
from datetime import datetime

if __name__ == "__main__":
    ddpg = DDPG()
    ddpg.run_ddpg()

    print("Hello World!")
