from src.ddpg import DDPG
from src.environments.jaimungal_environment import JaimungalEnvironment

if __name__ == "__main__":
    jenv = JaimungalEnvironment()

    while jenv.t < jenv.N - 1:
        jenv.v[jenv.t] = jenv.get_optimal()
        jenv.step()
    jenv.plot_simulation()

    # ddpg = DDPG()
    # ddpg.run_ddpg()

    print("Hello World!")
