import logging
import argparse
import os
import sys
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from rl.networks.envs import make_vec_envs
from rl.evaluation import evaluate
from rl.networks.model import Policy

from ped_sim import *


def main():
    """
    The main function for testing a trained model
    """
    # the following parameters will be determined for each test run
    parser = argparse.ArgumentParser('Parse configuration file')
    # the model directory that we are testing
    parser.add_argument('--model_dir', type=str, default='trained_models/ColliGrid_predictor')
    # render the environment or not. 
    # Set this to True to visualized and save the gif of the senarios
    parser.add_argument('--visualize', default=True, action='store_true')
    # if -1, it will run all scenarios in the test set; 
    # if >=0, it will run the specified test case (scenario #)
    # valid scenario#: 248-310
    parser.add_argument('--test_case', type=int, default=-1) # >=0 values works when visualize is set to True
    # Trained model weight file you want to test
    # UAW_PCG_pred, PCG_pred, GT_pred, No_pred, No_pred_SD (SD:speed depend. danger penalty), Human_Driver
    parser.add_argument('--test_model', type=str, default='UAW_PCG_pred')
    # whether to save trajectories of episodes
    parser.add_argument('--render_traj', default=False, action='store_true')
    # whether to save slide show of episodes
    parser.add_argument('--save_slides', default=False, action='store_true')
    test_args = parser.parse_args()
    if test_args.save_slides:
        test_args.visualize = True

    from importlib import import_module
    model_dir_temp = test_args.model_dir
    model_dir_temp = os.path.join(model_dir_temp, 'checkpoints', test_args.test_model)
    if model_dir_temp.endswith('/'):
        model_dir_temp = model_dir_temp[:-1]
        # import arguments.py from saved directory
        # if not found, import from the default directory
    try:
        model_dir_temp_arg = model_dir_temp.replace('/', '.') + '.arguments'
        model_arguments = import_module(model_dir_temp_arg)
        get_args = getattr(model_arguments, 'get_args')
        print(model_dir_temp_arg, ',argumnet')
    except:
        print('Failed to get get_args function from ', model_dir_temp,'/arguments.py')
        from arguments import get_args

    algo_args = get_args()

    # import config class from saved directory
    # if not found, import from the default directory
    try:
        model_dir_string = model_dir_temp.replace('/', '.') + '.configs.config'
        model_arguments = import_module(model_dir_string)
        Config = getattr(model_arguments, 'Config')
        print(model_dir_string, ',config')
    except:
        print('Failed to get config function from ', model_dir_temp,'/configs/config.py')
        from move_plan.configs.config import Config
    env_config = config = Config()

    if env_config.sim.render == True:
        raise ValueError("During test time the config.sim.render should be 'False'\
                          to esure skipping automatic reset in dummy_vec_env")

    # configure logging and device
    # print test result in log file
    log_file = os.path.join(test_args.model_dir, 'test')
    if not os.path.exists(log_file):
        os.mkdir(log_file)
    if test_args.visualize:
        log_file = os.path.join(test_args.model_dir, 'test', 'test_visual.log')
    else:
        log_file = os.path.join(test_args.model_dir, 'test', 'test_visual.log')

    file_handler = logging.FileHandler(log_file, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

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
        ax = None
        fig = NotImplementedError

    test_model = test_args.test_model + '.pt'
    load_path = os.path.join(test_args.model_dir, 'checkpoints', test_args.test_model, test_model)
    print(load_path)

    # create an environment
    env_name = algo_args.env_name

    eval_dir = os.path.join(test_args.model_dir, 'eval')
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

    env_config.render_traj = test_args.render_traj
    env_config.save_slides = test_args.save_slides
    env_config.save_path = os.path.join(test_args.model_dir, 'social_eval', 
                                        test_args.test_model[:-3])
    envs = make_vec_envs(env_name, algo_args.seed, 1,
                         algo_args.gamma, log_file, device, allow_early_resets=True,
                         config=env_config, ax=ax, fig=fig, test_case=test_args.test_case,
                         pretext_wrapper=config.env.use_wrapper)

    # load the policy weights
    if not config.robot.human_driver:
        actor_critic = Policy(
            envs.observation_space.spaces,
            envs.action_space,
            base_kwargs=algo_args,
            base=config.robot.policy)
        actor_critic.load_state_dict(torch.load(load_path, map_location=device))
        actor_critic.base.nenv = 1

        # allow the usage of multiple GPUs to increase the 
        # number of examples processed simultaneously
        nn.DataParallel(actor_critic).to(device)
    else:
        actor_critic = None

    # call the evaluation function
    evaluate(actor_critic, envs, 1, device, logging, config, algo_args, test_args.visualize)


if __name__ == '__main__':
    main()
