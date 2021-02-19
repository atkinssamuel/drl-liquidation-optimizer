import numpy as np

class DDPGCritic():


class DDPG:
    @staticmethod
    def compute_h(environment):
        return environment.epsilon * np.sign(environment.n) + environment.eta / environment.tau * environment.n

    @staticmethod
    def compute_V(environment):
        return np.square(environment.sigma) * \
               environment.tau * np.sum(np.square(environment.x))

    @staticmethod
    def compute_E(environment):
        E_1 = environment.gamma * np.sum(np.multiply(environment.x, environment.n))
        h = DDPG.compute_h(environment)
        E_2 = np.sum(np.multiply(environment.n, h))
        return E_1 + E_2

    @staticmethod
    def compute_U(environment):
        return DDPG.compute_E(environment) + environment.lam * DDPG.compute_V(environment)

    @staticmethod
    def compute_reward(environment):
        return

    def __init__(self):
        self.lr = 0.3
        self.r = []
        self.m = 1
        self.l = 1
        self.a = None

    def update_r(self, environment):
        self.r.append(np.log(environment.P[environment.k]/environment.P[environment.k-1]))

    def update_m(self, environment):
        self.m = (environment.k * environment.tau)/environment.N

    def update_l(self, environment):
        self.l = environment.x[environment.k]/environment.X

    def update_state(self, environment):
        self.update_r(environment)
        self.update_m(environment)
        self.update_l(environment)

    def run_ddpg(self):

