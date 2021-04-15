from ppo.environments.sa_almgrenchriss_discrete import SingleAgentAlmgrenChrissDiscrete
from shared.constants import PPOParams

import matplotlib.pyplot as plt


if __name__ == "__main__":
    env = SingleAgentAlmgrenChrissDiscrete(
        D   = PPOParams.D
    )

