import numpy as np
import torch
import os
import imageio
import time
from ped_sim.envs.utils.info import *



def evaluate(actor_critic, eval_envs, num_processes, device, logging,
              config, args, visualize=False):
    """ function to run all testing episodes and log the testing metrics """

    # initializations
    eval_episode_rewards = []
    eval_recurrent_hidden_states = {}

    if not config.robot.human_driver:
        node_num = 1
        edge_num = actor_critic.base.human_num + 1
        eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num, 
                                                                     actor_critic.base.human_node_rnn_size,
                                                                     device=device)

        eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, 
                                                                     edge_num,
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

    computation_time = []

    if hasattr(eval_envs.venv, 'envs'):
        baseEnv = eval_envs.venv.envs[0].env
    else:
        baseEnv = eval_envs.venv.unwrapped.envs[0].env

    test_size = baseEnv.dataloader.num_scenarios 

    print('test_case: ', baseEnv.test_case)
    if baseEnv.test_case is not None:
        test_size = 1 # we want to test a specifc scenario

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
        scenario_num = baseEnv.scenario_num


        while not done:
            stepCounter = stepCounter + 1
            time_start = time.time()
            if not config.robot.human_driver:
                # run inference on the NN policy
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=True)
            else:
                # for now. In the human_driver case the actual action will be calculated
                #  within the step function of the environment
                action = torch.zeros(1,2) 

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

            time_end = time.time()
            computation_time.append(time_end - time_start)

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

        # update the last frame's prediction before rendering
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
        
        if visualize:
            eval_envs.render()

        # an episode ends!
        print('')
        print('Reward={:.2f}'.format(episode_rew.item()))
        print('Episode', k, 'scenario #', scenario_num, 'ends in', stepCounter)
        too_close_ratios.append(too_close/stepCounter*100)


        if isinstance(infos[0]['info'], ReachGoal):
            all_path_len.append(path_len)
            success += 1
            nav_tim = (global_frame-baseEnv.obs_length+1) * (baseEnv.time_step)
            success_times.append(nav_tim)
            print('Success')
        elif isinstance(infos[0]['info'], Collision):
            collision += 1
            collision_cases.append(scenario_num)
            collision_frames.append(global_frame)
            # collision_speed.append(infos[0]['info'].col_speed)
            print('Collision')
        elif isinstance(infos[0]['info'], Timeout):
            timeout += 1
            timeout_cases.append(scenario_num)
            timeout_time = (frame_limit-baseEnv.obs_length+1) * (baseEnv.time_step)
            timeout_times.append(timeout_time)
            print('Time out')
        elif isinstance(infos[0]['info'], Collision_Vehicle):
            collision_veh += 1
            collision_veh_cases.append(scenario_num)
            collision_veh_frames.append(global_frame)
            print('Vehicle Collision')
        elif isinstance(infos[0]['info'] is None):
            pass
        else:
            raise ValueError('Invalid end signal from environment')
        
        if visualize:
            # save the sceario as a gif
            early_stop_frame = baseEnv.frame
            gif_folder = os.path.join(config.data.visual_save_path, 'gifs')
            plot_folder = os.path.join(config.data.visual_save_path, 'plots')
            Create_animation(plot_folder, gif_folder, baseEnv.simulation_lenght,
                        baseEnv.obs_length, baseEnv.scenario_num,
                        early_stop_frame, fps=3)
            # Create_video(plot_folder, gif_folder, baseEnv.simulation_lenght,
            #             baseEnv.obs_length-1, baseEnv.scenario_num,
            #             early_stop_frame, fps=3)
        
        print('__________________')

    # all episodes end
    success_rate = success / test_size
    collision_rate = collision / test_size
    collision_veh_rate = collision_veh / test_size
    timeout_rate = timeout / test_size
    assert success + collision + collision_veh + timeout == test_size
    avg_nav_time = sum(success_times) / len(
        success_times) if success_times else (frame_limit-baseEnv.obs_length+1)*(baseEnv.time_step)
    nav_time_mean, nav_time_std = calculate_mean_and_std(success_times)
    path_len_mean, path_len_std = calculate_mean_and_std(all_path_len)
    too_close_ratio_mean, too_close_ratio_std = calculate_mean_and_std(too_close_ratios)
    min_dist_mean, min_dist_std = calculate_mean_and_std(min_dist)
    min_intrusion_speed_mean, min_intrusion_speed_std = calculate_mean_and_std(min_intrusion_speed)
    computatioal_time_mean, computational_time_std = calculate_mean_and_std(computation_time)
    # logging
    logging.info(
        'Testing success rate: {:.2f}, collision rate: {:.2f}, '
        'vehicle collision rate: {:.2f}, timeout rate: {:.2f}, '
        'nav time: {:.2f}±{:.2f}, path length: {:.2f}±{:.2f}, '
        'average intrusion ratio: {:.2f}±{:.2f}%, '
        'average minimal distance during intrusions: {:.2f}±{:.2f}, ' 
        'speed at minimal distance: {:.2f}±{:.2f},'
        'one_step_compute_time: {:.2f}±{:.2f}'.
        format(success_rate, collision_rate, collision_veh_rate, timeout_rate,
                nav_time_mean, nav_time_std, path_len_mean, path_len_std, 
                too_close_ratio_mean, too_close_ratio_std,
                min_dist_mean, min_dist_std, min_intrusion_speed_mean, min_intrusion_speed_std,
                computatioal_time_mean, computational_time_std))

    logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
    logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
    # logging.info('Vehicle Collision cases: ' + ' '.join([str(x) for x in collision_veh_cases]))
    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    eval_envs.close()

   
def Create_animation(dir, gif_dir, num_frames, start_frame, scenario_num,
                     early_stop_frame, fps=5):
     
    if not os.path.exists(gif_dir):
        os.mkdir(gif_dir)
    frames = []
    if early_stop_frame is not None:
        num_frames = early_stop_frame
    for t in range(start_frame, num_frames+1):
        image_dir = os.path.join(dir, f'img_{t}.png')
        image = imageio.imread(image_dir)
        frames.append(image)
    frames.append(frames[-1])
    save_dir = os.path.join(gif_dir, f'scenario{scenario_num}.gif')
    duration = 1000 * 1/fps
    imageio.mimsave(save_dir, # output gif
                frames,          # array of input frames
                duration=duration,
                loop = True)
    

def Create_video(dir, video_dir, num_frames, start_frame, scenario_num,
                 early_stop_frame, fps=25, codec='libx264'):
     
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    frames = []
    if early_stop_frame is not None:
        num_frames = early_stop_frame
    for t in range(start_frame, num_frames + 1):
        image_dir = os.path.join(dir, f'img_{t}.png')
        image = imageio.imread(image_dir)
        frames.append(image)
    
    # Duplicate the last frame for smoother video looping
    frames.append(frames[-1])

    save_dir = os.path.join(video_dir, f'scenario{scenario_num}.mp4')
    duration = 1 / fps

    # Write frames to video file
    writer = imageio.get_writer(save_dir, fps=fps, codec=codec)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def calculate_mean_and_std(data):
    """
    Calculate the mean and standard deviation of a list of numbers.

    Parameters:
    - data: List of numbers

    Returns:
    - mean: Mean of the data
    - std_dev: Standard deviation of the data
    """

    data = np.asarray(data)
    mean = np.mean(data)
    std_dev = np.std(data)
    return mean, std_dev