import numpy as np
import torch
import pandas as pd
import os

from ped_sim.envs.utils.info import *


def evaluate(actor_critic, eval_envs, num_processes, device, 
             logging, config, args, model_num, eval_dir, visualize=False):
    """ function to run all evaluation episodes and log the testing metrics """

    # initializations
    eval_episode_rewards = []
    eval_recurrent_hidden_states = {}

    node_num = 1
    edge_num = actor_critic.base.human_num + 1
    eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num, 
                                                                 actor_critic.base.human_node_rnn_size,
                                                                 device=device)

    eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, edge_num,
                                                                 actor_critic.base.human_human_edge_rnn_size,
                                                                 device=device)

    eval_masks = torch.zeros(num_processes, 1, device=device)

    success_times = []
    collision_frames = []
    collision_veh_frames = []
    timeout_times = []

    success = 0
    collision = 0
    collision_veh = 0
    timeout = 0
    too_close_ratios = []
    min_dist = []

    collision_cases = []
    min_intrusion_speed = []
    collision_veh_cases = []
    timeout_cases = []

    all_path_len = []


    if hasattr(eval_envs.venv, 'envs'):
        baseEnv = eval_envs.venv.envs[0].env
    else:
        baseEnv = eval_envs.venv.unwrapped.envs[0].env

    test_size = baseEnv.dataloader.num_scenarios 
    print('============== test size: ', test_size, ' =============  ')

    # start the testing episodes
    for k in range(test_size):
        baseEnv.episode_k = k
        done = False
        rewards = []
        stepCounter = 0
        episode_rew = 0
        obs = eval_envs.reset()
        global_frame = 0.0
        path_len = 0.
        too_close = 0.
        last_pos = obs['robot_node'][0, 0, :2].cpu().numpy()
        frame_limit = baseEnv.time_out_f


        while not done:
            stepCounter = stepCounter + 1
            # run inference on the NN policy
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=True)
            if not done:
                global_frame = baseEnv.frame

            # if the vec_pretext_normalize.py wrapper is used, send the predicted traj to env
            if args.env_name == 'PedSimPred-v0' and config.env.use_wrapper:
                if config.sim.uncertainty_aware:
                    out_pred = obs['pred_pos'][:, :, 6:].to('cpu')
                else:
                    # [nenv, max_human_num, 2*pred_steps]
                    out_pred = obs['spatial_edges'][:, :, 2:].to('cpu')
                next_obs = obs['ped_pos'] # [nenv, obs_len, max_human_num, 5]
                next_mask = obs['ped_mask'] # [nenv, obs_len, max_human_num]

                if config.sim.uncertainty_aware:
                    data = torch.zeros(1, config.sim.pred_len, config.sim.max_human_num, 12)
                    data[:,:,:,:6] = out_pred.reshape(1, config.sim.max_human_num, 
                                                      config.sim.pred_len, 6).permute(0, 2, 1, 3)
                    data[:,:,:,6:11] = next_obs
                    data[:,:,:,11] = next_mask
                else:
                    data = torch.zeros(1, config.sim.pred_len, config.sim.max_human_num, 8)
                    data[:,:,:,:2] = out_pred.reshape(1, config.sim.max_human_num, 
                                                      config.sim.pred_len, 2).permute(0, 2, 1, 3)
                    data[:,:,:,2:7] = next_obs
                    data[:,:,:,7] = next_mask
                # send manager action to all processes
                ack = eval_envs.talk2Env(data)
                assert all(ack)
            # render
            if visualize:
                eval_envs.render()

            # Obser reward and next obs
            obs, rew, done, infos = eval_envs.step(action)

            # record the info for calculating testing metrics
            rewards.append(rew)

            path_len = path_len + np.linalg.norm(
                                            obs['robot_node'][0, 0, :2].cpu().numpy() - last_pos)
            last_pos = obs['robot_node'][0, 0, :2].cpu().numpy()

            if isinstance(infos[0]['info'], Danger):
                too_close = too_close + 1
                min_dist.append(infos[0]['info'].min_dist)
                min_intrusion_speed.append(infos[0]['info'].speedAtdmin)

            episode_rew += rew[0]

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])

        # an episode ends!
        print('')
        print('Reward={}'.format(episode_rew))
        print('Episode', k, 'ends in', stepCounter)
        all_path_len.append(path_len)
        too_close_ratios.append(too_close/stepCounter*100)


        if isinstance(infos[0]['info'], ReachGoal):
            success += 1
            nav_tim = (global_frame-baseEnv.obs_length+1) * (baseEnv.time_step)
            success_times.append(nav_tim)
            print('Success')
        elif isinstance(infos[0]['info'], Collision):
            collision += 1
            collision_cases.append(baseEnv.scenario_num)
            collision_frames.append(global_frame)
            print('Collision')
        elif isinstance(infos[0]['info'], Timeout):
            timeout += 1
            timeout_cases.append(baseEnv.scenario_num)
            timeout_time = (frame_limit-baseEnv.obs_length+1) * (baseEnv.time_step)
            timeout_times.append(timeout_time)
            print('Time out')
        elif isinstance(infos[0]['info'], Collision_Vehicle):
            collision_veh += 1
            collision_veh_cases.append(baseEnv.scenario_num)
            collision_veh_frames.append(global_frame)
            print('Vehicle Collision')
        elif isinstance(infos[0]['info'] is None):
            pass
        else:
            raise ValueError('Invalid end signal from environment')

    # all episodes end
    success_rate = success / test_size
    collision_rate = collision / test_size
    collision_veh_rate = collision_veh / test_size
    timeout_rate = timeout / test_size
    assert success + collision + collision_veh + timeout == test_size
    avg_nav_time = sum(success_times) / len(
        success_times) if success_times else (frame_limit-baseEnv.obs_length+1)*(baseEnv.time_step)


    # saving the informatino of all episodes on the eval dataset in a csv file
    df = pd.DataFrame({'model_num': [model_num], 'success_rate': [success_rate],
                               'collision_rate': [collision_rate], 
                               ' vehicle collision rate': [collision_veh_rate],
                               'timeout_rate': [timeout_rate], 'avg_nav_time': [avg_nav_time],
                               'all_path_len': [np.mean(all_path_len)], 
                               'too_close_ratios': [np.mean(too_close_ratios)],
                               'min_dist': [np.mean(min_dist)], 
                               'intrusion_speed': [np.mean(min_intrusion_speed)]})
    if os.path.exists(os.path.join(eval_dir, 'progress.csv')):
        df.to_csv(os.path.join(eval_dir, 'progress.csv'), mode='a', header=False, index=False)
    else:
        df.to_csv(os.path.join(eval_dir, 'progress.csv'), mode='w', header=True, index=False)

    out = (success_rate, collision_rate, collision_veh_rate,
            timeout_rate, avg_nav_time, np.mean(all_path_len),
            np.mean(too_close_ratios), np.mean(min_dist))

    return out
