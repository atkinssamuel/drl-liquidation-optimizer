from ppo.environments.ma_almgrenchriss import MultiAgentAlmgrenChriss
from ppo.environments.sa_almgrenchriss import SingleAgentAlmgrenChriss
from ppo.src.discrete.ppo_torch import DiscreteAgent
from shared.constants import PPOAgent1Params, PPODirectories, PPOAgent2Params, PPOTrainingParams

from ppo.src.agent import PPOAgent
from ppo.src.train import train_ppo, train_multi_agent_ppo

import json

if __name__ == "__main__":
    hyper_parameter_tuning = False
    normal_run = True

    # if hyper_parameter_tuning:
    #     batch_sizes = [5, 10, 20, 40, 100]
    #     epochs = [4, 8, 16, 32]
    #     gae_lambdas = [0.9, 0.95, 1]
    #     policy_clips = [0.1, 0.2, 0.3]
    #     gammas = [0.99]
    #
    #     hyper_parameter_dict = {}
    #     hyper_parameter_index = 0
    #
    #     total_hyper_parameter_settings = len(batch_sizes) * len(epochs) * len(gae_lambdas) * len(policy_clips) \
    #                                      * len(gammas)
    #
    #     for batch_size in batch_sizes:
    #         for epoch in epochs:
    #             for gae_lambda in gae_lambdas:
    #                 for policy_clip in policy_clips:
    #                     for gamma in gammas:
    #                         hyper_parameter_dict[hyper_parameter_index] = {}
    #                         hyper_parameter_dict[hyper_parameter_index]["batch_size"] = batch_size
    #                         hyper_parameter_dict[hyper_parameter_index]["epoch"] = epoch
    #                         hyper_parameter_dict[hyper_parameter_index]["gae_lambda"] = gae_lambda
    #                         hyper_parameter_dict[hyper_parameter_index]["policy_clip"] = policy_clip
    #                         hyper_parameter_dict[hyper_parameter_index]["gamma"] = gamma
    #
    #                         print(f"\n\nTesting hyper-parameter set: {hyper_parameter_dict[hyper_parameter_index]}\n"
    #                               f"Hyper-parameter setting {hyper_parameter_index}/{total_hyper_parameter_settings}\n")
    #
    #                         env = SingleAgentAlmgrenChriss(
    #                             D=PPOEnvParams.D,
    #                             risk_aversion=PPOEnvParams.risk_aversion
    #                         )
    #
    #                         agent = PPOAgent(
    #                             batch_size=batch_size,
    #                             alpha=PPOAgentParams.alpha,
    #                             epochs=epoch,
    #                             input_dims=PPOEnvParams.D + 3,
    #                             gae_lambda=gae_lambda,
    #                             policy_clip=policy_clip,
    #                             gamma=gamma
    #                         )
    #
    #                         max_reward = train_ppo(agent, env,
    #                                                episodes=PPOAgentParams.episodes,
    #                                                update_frequency=PPOAgentParams.update_frequency,
    #                                                moving_average_length=PPOAgentParams.moving_average_length,
    #                                                checkpoint_frequency=PPOAgentParams.checkpoint_frequency,
    #                                                reward_file_name=f"rewards_{hyper_parameter_index}.png"
    #                                                )
    #
    #                         hyper_parameter_index += 1
    #
    #                         hyper_parameter_dict_file = open(PPODirectories.tuning + "hyper_parameter_dict.json", "w")
    #                         json.dump(hyper_parameter_dict, hyper_parameter_dict_file)


    if normal_run:

        agent_1_args = PPOAgent1Params()
        agent_2_args = PPOAgent2Params()

        env = MultiAgentAlmgrenChriss(agent_1_args.D, agent_2_args.D, agent_1_args.D)

        agent_1 = PPOAgent(agent_1_args, env=env)

        agent_2 = PPOAgent(agent_2_args, env=env)

        agent_3 = PPOAgent(agent_1_args, env=env)

        ppo_training_params = PPOTrainingParams()
        train_multi_agent_ppo(agent_1, agent_2, agent_3, env=env, ppo_training_params=ppo_training_params)
