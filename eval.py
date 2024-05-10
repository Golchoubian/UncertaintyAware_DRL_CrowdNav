import logging
import argparse
import os
import sys
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from rl.networks.envs import make_vec_envs
from rl.evaluation_eval import evaluate
from rl.networks.model import Policy
from ped_sim import *


def main():
    """
    The main function for evaluating the saved models during training
    Based on the evalation dataset, here we find the best model
    """
    # the following parameters will be determined for each test run
    parser = argparse.ArgumentParser('Parse configuration file')
    # the model directory that we are testing
    parser.add_argument('--model_dir', type=str, default='trained_models/my_model')
    # the model directory that we store the eval data in
    parser.add_argument('--log_dir', type=str, default='trained_models/ColliGrid_predictor')
    # render the environment or not
    parser.add_argument('--visualize', default=False, action='store_true')
    # if -1, it will run all scenarios in the test set; 
    # if >=0, it will run the specified test case
    parser.add_argument('--test_case', type=int, default=-1)
    # the epoch number of the lasr saved model from training that we want to evaluate here
    parser.add_argument('--last_test_model', type=int, default=83332)
    # whether to save trajectories of episodes
    parser.add_argument('--render_traj', default=False, action='store_true')
    # whether to save slide show of episodes
    parser.add_argument('--save_slides', default=False, action='store_true')
    test_args = parser.parse_args()
    if test_args.save_slides:
        test_args.visualize = True

    from importlib import import_module
    model_dir_temp = test_args.model_dir
    if model_dir_temp.endswith('/'):
        model_dir_temp = model_dir_temp[:-1]
        # import arguments.py from saved directory
        # if not found, import from the default directory
    try:
        model_dir_temp = model_dir_temp.replace('/', '.') + '.arguments'
        model_arguments = import_module(model_dir_temp)
        get_args = getattr(model_arguments, 'get_args')
    except:
        print('Failed to get get_args function from ', test_args.model_dir, '/arguments.py')
        from arguments import get_args

    algo_args = get_args()

    # import config class from saved directory
    # if not found, import from the default directory
    try:
        model_dir_string = model_dir_temp.replace('/', '.') + '.configs.config'
        model_arguments = import_module(model_dir_string)
        Config = getattr(model_arguments, 'Config')

    except:
        print('Failed to get config function form ', test_args.model_dir)
        from move_plan.configs.config import Config
    env_config = config = Config()

    # configure logging and device
    # print eval result in log file
    
    eval_dir = os.path.join(test_args.log_dir, 'eval')
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

    log_file = os.path.join(test_args.log_dir, 'eval', 'eval.log')

    file_handler = logging.FileHandler(log_file, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', 
                        datefmt="%Y-%m-%d %H:%M:%S")

    logging.info('robot FOV %f', config.robot.FOV)

    torch.manual_seed(algo_args.seed)
    torch.cuda.manual_seed_all(algo_args.seed)
    if algo_args.cuda:
        if algo_args.cuda_deterministic:
            # reproducible but slower
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            # not reproducible but faster
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    torch.set_num_threads(1)
    device = torch.device("cuda" if algo_args.cuda else "cpu")

    logging.info('Create other envs with new settings')

    # set up visualization
    if test_args.visualize:
        fig, ax = plt.subplots()
        plt.ion()
        plt.show()
    else:
        fig = None
        ax = None

    # create an environment
    env_name = algo_args.env_name

    envs = make_vec_envs(env_name, algo_args.seed, 1,
                         algo_args.gamma, eval_dir, device, allow_early_resets=True,
                         config=env_config, ax=ax, fig=fig, test_case=test_args.test_case,
                         pretext_wrapper=config.env.use_wrapper)

    # load the policy weights
    actor_critic = Policy(
        envs.observation_space.spaces,
        envs.action_space,
        base_kwargs=algo_args,
        base=config.robot.policy)
    
    success_rate = []
    collision_rate = []
    collision_veh_rate = []
    timeout_rate = []
    avg_nav_time = []
    path_len= []
    too_close_ratios = []
    min_dist = []
    model_id = []


    for model_num in range(10000, test_args.last_test_model, 200):
        load_path = os.path.join(test_args.model_dir, 'checkpoints', str(model_num)+'.pt')
        print(load_path)
        model_id.append(model_num)

        actor_critic.load_state_dict(torch.load(load_path, map_location=device))
        actor_critic.base.nenv = 1

        # allow the usage of multiple GPUs to increase the 
        # number of examples processed simultaneously
        nn.DataParallel(actor_critic).to(device)

        # call the evaluation function
        out = evaluate(actor_critic, envs, 1, device, logging, config, algo_args,
                        model_num, eval_dir, test_args.visualize)
        (success_rate_model, collision_rate_model, collision_veh_rate_model,
            timeout_rate_model, avg_nav_time_model, path_len_model,
            too_close_ratios_model, min_dist_model) = out
        
        success_rate.append(success_rate_model)
        collision_rate.append(collision_rate_model)
        collision_veh_rate.append(collision_veh_rate_model)
        timeout_rate.append(timeout_rate_model)
        avg_nav_time.append(avg_nav_time_model)
        path_len.append(path_len_model)
        too_close_ratios.append(too_close_ratios_model)
        min_dist.append(min_dist_model)
    
    idx = success_rate.index(max(success_rate))
    best_model_id = model_id[idx]
    print('best model id: ', best_model_id)
    
    # logging
    logging.info(
        'Best model id: {:d}, '
        'Evaluting success rate: {:.2f}, collision rate: {:.2f}, '
        'vehicle collision rate: {:.2f}, timeout rate: {:.2f}, '
        'nav time: {:.2f}, path length: {:.2f}, average intrusion ratio: {:.2f}%, '
        'average minimal distance during intrusions: {:.2f}'.
        format(best_model_id, success_rate[idx], collision_rate[idx],
                collision_veh_rate[idx], timeout_rate[idx], 
                avg_nav_time[idx], path_len[idx], too_close_ratios[idx],
                min_dist[idx]))


if __name__ == '__main__':
    main()
