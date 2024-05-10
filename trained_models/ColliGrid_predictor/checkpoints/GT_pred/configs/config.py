from arguments import get_args
import numpy as np


class BaseConfig(object):
    def __init__(self):
        pass


class Config(object):

    args = get_args()
    training = BaseConfig()
    training.device = "cuda:0" if args.cuda else "cpu"
    training.load_path = args.load_path

    # general configs for OpenAI gym env
    env = BaseConfig()

    # The following time_step shouldn't be changed
    # It is based on HBS dataset's fps
    env.time_step = 0.5 # 2 fps

    env.num_processes = args.num_processes

    # config for reward function
    reward = BaseConfig()
    reward.success_reward = 10
    reward.collision_penalty = -20
    reward.veh_collision_penalty = -20
    # discomfort distance
    reward.discomfort_dist = 1 # meter
    reward.discomfort_penalty_factor = 40 # smooth continuous transition to collision penality of -20 considering that the discomfort distance is 1.
    reward.gamma = 0.99

    # config for simulation
    sim = BaseConfig()
    sim.max_human_num = 60  # the maximum number of pedestrians in the HBS scenarios is 60
    sim.max_vehicle_num = 15  # the maximum number of vehicles in the HBS scenarios is 15
    # the number of frames beyond the exsting frames in ground truth data for continuing the simulation
    sim.extended_time_out = 30

    # 'truth': ground truth future traj
    # 'inferred': inferred future traj from UAW-PCG prediction model
    # 'none': no prediction
    sim.predict_method = 'truth' # Also adjust the uncertainty-aware args in the argument.py file!
    sim.predict_network = 'CollisionGrid' # VanillaLSTM or CollisionGrid
    sim.uncertainty_aware = args.uncertainty_aware
    # render the simulation during training or not
    # during test time make sure to keep this False (for skipping automatic reset in dummy_vec_env during test time)
    sim.render = False 

    sim.obs_len = args.obs_length
    sim.pred_len = args.pred_length

    # for test part
    render_traj = False
    save_slides = False
    save_path = None

    # whether wrap the vec env with VecPretextNormalize class
    # = True only if we are using a network for human trajectory prediction (sim.predict_method = 'inferred')
    if sim.predict_method in {'inferred', 'truth'}:
        env.use_wrapper = True
    else:
        env.use_wrapper = False

    # action space of the robot
    action_space = BaseConfig()
    # holonomic or unicycle
    action_space.kinematics = 'unicycle'  # 'unicycle', 'holonomic', 'double_integrator_unicycle', 'bicycle'

    # robot config
    robot = BaseConfig()
    # whether to use the human driver actual trajectory in the dataset as an upper bound to navigtion algorithm performance
    robot.human_driver = False # only works for no prediction case ('PedSim-v0' in argument and 'none' in config) in test.py file
    robot.policy = 'social_move_plan'
    robot.radius = 1 # meter
    robot.v_pref = 15/3.6  # 20 km/h is the maximun speed allowed in the shared space of the HBS dataset
    robot.max_steering_angle = 25 * np.pi/180 # degrees
    robot.L = 1.75 # polaris gem e2 wheelbase from documents
    robot.FOV = 2
    robot.sensor_range = 15 # radius of perception range

    # ped config
    peds = BaseConfig()
    peds.policy = 'none'
    peds.radius = 0.3
    peds.v_pref = 1

    # other veh config
    vehs = BaseConfig()
    vehs.policy = 'none'
    vehs.radius = 0.7
    vehs.v_pref = 20/3.6  
    # 20 km/h is the maximun speed allowed in the shared space of the HBS dataset

    # config for data collection
    data = BaseConfig()
    data.visual_save_path = 'trained_models/ColliGrid_predictor/visual'

    if sim.predict_method in {'inferred', 'truth'} and env.use_wrapper == False:
        raise ValueError("If using inferred prediction, you must wrap the envs!")
    if sim.predict_method not in {'inferred', 'truth'} and env.use_wrapper:
        raise ValueError("If not using inferred prediction, you must NOT wrap the envs!")
    
    if sim.predict_method == 'truth' and sim.uncertainty_aware:
        raise ValueError("If using truth prediction, you must set the uncertainty_aware in args to False!")
