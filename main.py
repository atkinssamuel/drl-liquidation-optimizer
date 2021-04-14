from ddpg.src.algos.ddpg import DDPG
from ddpg.src.constants import DDPGParams
from ddpg.src.helpers import clear_results

if __name__ == "__main__":
    clear_results(algo=DDPGParams.algo, clear=DDPGParams.clear)

    ddpg = DDPG(algo                    = DDPGParams.algo,
                D                       = DDPGParams.D,
                rho                     = DDPGParams.rho,
                batch_size              = DDPGParams.batch_size,
                discount_factor         = DDPGParams.discount,
                M                       = DDPGParams.M,
                critic_lr               = DDPGParams.critic_lr,
                critic_weight_decay     = DDPGParams.critic_weight_decay,
                actor_lr                = DDPGParams.actor_lr,
                replay_buffer_size      = DDPGParams.replay_buffer_size,
                checkpoint_frequency    = DDPGParams.checkpoint_frequency,
                inventory_sim_length    = DDPGParams.inventory_sim_length,
                pre_training_length     = DDPGParams.pre_training_length,
                post_training_length    = DDPGParams.post_training_length,
                training_noise          = DDPGParams.training_noise,
                decay                   = DDPGParams.decay)

    ddpg.run_ddpg()

