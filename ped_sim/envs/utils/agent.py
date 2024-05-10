from move_plan.policy.policy_factory import policy_factory
import torch

class Agent(object):
    def __init__(self, config, section):
        '''
        Base class for robot (AV) and human.
        '''

        if section == 'robot':
            subconfig = config.robot
        elif section == 'ped':
            subconfig = config.peds
        elif section == 'veh':
            subconfig = config.vehs

        self.obs_length = config.sim.obs_len
        self.pred_length = config.sim.pred_len


        self.policy = policy_factory[subconfig.policy](config)
        self.v_pref = subconfig.v_pref
        self.radius = subconfig.radius

        
        self.px = None
        self.py = None
        self.vx = None
        self.vy = None
        self.gx = None
        self.gy = None
        self.theta = None
        self.time_step = config.env.time_step

        self.true_pose = None
        self.lookup = None
        self.agent_type = section

        self.curr_obs = None
        self.curr_mask = None   

    def set_scen_data(self, pos_scen, mask_scen):
        
        self.true_pose = pos_scen
        self.mask = mask_scen
    
        self.px = self.true_pose[self.obs_length-1,:,0] # for all pedestrians and all vehicles
        self.py = self.true_pose[self.obs_length-1,:,1] 

        self.vx = self.true_pose[self.obs_length-1,:,2] 
        self.vy = self.true_pose[self.obs_length-1,:,3] 


    def get_traj_history(self):
        return self.curr_obs, self.curr_mask
    
    def get_position(self):
        return [self.px, self.py]
    
    def get_velocity(self):
        return [self.vx, self.vy]
    
    def calculate_theta(self, vx, vy):
        '''
        calculatig the heading angle in radian
        given the velocity components
        '''
        theta = torch.atan2(vy, vx)
        return theta
    
