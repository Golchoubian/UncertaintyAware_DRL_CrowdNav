from ped_sim.envs.utils.agent import Agent
from ped_sim.envs.utils.action import ActionXY, ActionRot, ActionAcc, ActionBicycle
import numpy as np
import torch

class Robot(Agent):
    # see Agent class in agent.py for details!!!
    def __init__(self, config, section):
        super().__init__(config, section)

        self.generated_traj_UptoNow = None
        self.kinematics = config.action_space.kinematics
        self.time_step = config.env.time_step
        self.sensor_range = config.robot.sensor_range
        self.max_steering_angle = config.robot.max_steering_angle
        self.L = config.robot.L # robotic vehicle's wheelbase

    def set_scen_data(self, pos_scen, mask_scen):
        
        self.true_pose = pos_scen
        self.mask = mask_scen
   
        self.px = self.true_pose[self.obs_length-1,0,0] 
        self.py = self.true_pose[self.obs_length-1,0,1] 

        # self.gx = self.true_pose[-1,0,0]
        # self.gy = self.true_pose[-1,0,1]

        self.vx = self.true_pose[self.obs_length-1,0,2]
        self.vy = self.true_pose[self.obs_length-1,0,3]
        self.v = np.sqrt(self.vx**2 + self.vy**2)
        self.w = 0
        self.phi = 0
        self.theta = self.calculate_theta(self.vx, self.vy)

        self.generated_traj_UptoNow = self.true_pose[:self.obs_length,:,:]

    def get_curr_ob(self, f):
        '''
        returing a history of the robots trajectory data
        for the last obs_length period given the current frame number
        [f-obs_length:f]
        this history will be built using the planned trajecotry during the simulation
        '''
        
        self.dataset_curr_timestep = self.generated_traj_UptoNow[f,0,4] 
        self.curr_obs = self.generated_traj_UptoNow[f-self.obs_length+1:f+1,:,:]
        self.curr_mask = self.mask[f-self.obs_length+1:f+1,:]

        curr_ob = {'pos': self.curr_obs,
                    'mask': self.curr_mask}
              
        return curr_ob


    def planned_traj(self, reset=False):

        '''
        This function derives the planned trajecotry of the robot for the next 
        pred_length assuming constant velocity.
        during reset this is an assumed plan. But during the step, this is the
        plan that the pedestrian will assume the robot will take according to its
        previous action and assuming the vehicle will keep a constnat velocity model
        This planned traj will be used for both prediction and plotting
        In step this function is called after executing the action and updating the states

        Output:
        [[planned_length, 1, 5]]

        '''

        planned_length = self.pred_length
        planned_traj = torch.zeros(planned_length,1, 5)
        
    
        [px, py] = self.get_position()
        [vx, vy] = self.get_velocity()
        start_timestep = self.get_curr_dataset_timestep()

        vel = torch.FloatTensor([vx, vy])
        pos = torch.FloatTensor([px, py])
        vel_rp = vel.unsqueeze(dim=0).repeat(planned_length,1)
        pos_rp = pos.unsqueeze(dim=0).repeat(planned_length,1)
        timestep = torch.FloatTensor([*range(1, planned_length+1)])
        timestep_rp = self.time_step * timestep.unsqueeze(dim=1).repeat(1, 2)
        planned_traj[:,0,0:2] = torch.mul(timestep_rp,vel_rp) + pos_rp
        planned_traj[:,0,2:4] = vel_rp
        planned_traj[:,0,4] =  start_timestep + (self.time_step * timestep) 

        return planned_traj

    
    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)
        elif self.kinematics == 'unicycle':
            assert isinstance(action, ActionRot)
        elif self.kinematics == 'double_integrator_unicycle':
            assert isinstance(action, ActionAcc)
        elif self.kinematics == 'bicycle':
            assert isinstance(action, ActionBicycle)
        else:
            raise Exception('Please specify the kinematics of the robot in the config file')

    def compute_position(self, action):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            px = self.px + action.vx * self.time_step
            py = self.py + action.vy * self.time_step
        
        elif self.kinematics == 'unicycle':
            # # naive dynamics
            # theta = self.theta + action.r * delta_t # if action.r is w
            # # theta = self.theta + action.r # if action.r is delta theta
            # px = self.px + np.cos(theta) * action.v * delta_t
            # py = self.py + np.sin(theta) * action.v * delta_t

            # differential drive
            epsilon = 0.0001
            if abs(action.r) < epsilon:
                px = self.px + action.v * np.cos(self.theta) * self.time_step
                py = self.py + action.v * np.sin(self.theta) * self.time_step
            else:
                w = action.r/self.time_step # action.r is delta theta
                R = action.v/w
                px = self.px - R * np.sin(self.theta) + R * np.sin(self.theta + action.r) 
                py = self.py + R * np.cos(self.theta) - R * np.cos(self.theta + action.r)


        elif self.kinematics == 'double_integrator_unicycle':
            v = np.clip(self.v + action.u_a * self.time_step, 0, self.v_pref)
            w = np.clip(self.w + action.u_alpha * self.time_step, -0.1, 0.1) # adjust these bounds !!!!
            theta = self.theta + w * self.time_step
            px = self.px + v * np.cos(theta) * self.time_step
            py = self.py + v * np.sin(theta) * self.time_step

        elif self.kinematics == 'bicycle':
            v = np.clip(self.v + action.a * self.time_step, 0, self.v_pref)
            phi = np.clip(self.phi + action.steering_change_rate * self.time_step,
                           -self.max_steering_angle, self.max_steering_angle) # steering angle
            px = self.px + v * np.cos(self.theta) * self.time_step
            py = self.py + v * np.sin(self.theta) * self.time_step
            theta = self.theta + v/self.L * np.tan(phi) * self.time_step

        else: 
            raise Exception('Please specify the kinematics of the robot in the config file')

        return px, py
    
    
    def compute_velocity(self, action):

        self.check_validity(action)
        if self.kinematics == 'holonomic':
            next_vx = action.vx
            next_vy = action.vy
        elif self.kinematics == 'unicycle':
            next_theta = (self.theta + action.r) % (2 * np.pi) # action.r is delta theta
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
        elif self.kinematics == 'double_integrator_unicycle':
            next_v = self.v + action.u_a * self.time_step
            next_w = self.w + action.u_alpha * self.time_step
            next_theta = self.theta + next_w * self.time_step
            next_vx = next_v * np.cos(next_theta)
            next_vy = next_v * np.sin(next_theta)
            return next_vx, next_vy, next_theta, next_v, next_w
        elif self.kinematics == 'bicycle':
            next_v = np.clip(self.v + action.a * self.time_step, 0, self.v_pref)
            next_phi = np.clip(self.phi + action.steering_change_rate * self.time_step, 
                               -self.max_steering_angle, self.max_steering_angle) # steering angle
            next_theta = self.theta + next_v/self.L * np.tan(next_phi) * self.time_step
            next_vx = next_v * np.cos(next_theta)
            next_vy = next_v * np.sin(next_theta)
            return next_vx, next_vy, next_theta, next_v, next_phi
        else: 
            raise Exception('Please specify the kinematics of the robot in the config file')
        
        return next_vx, next_vy, next_theta
    
    
    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        pos = self.compute_position(action)
        self.px, self.py = pos
        vel = self.compute_velocity(action)

        prev_vel = self.v

        if self.kinematics == 'double_integrator_unicycle':
            self.vx, self.vy, self.theta, self.v, self.w = vel
        elif self.kinematics == 'bicycle':
            self.vx, self.vy, self.theta, self.v, self.phi = vel
            # print('phi: ', self.phi)
        else:
            self.vx, self.vy, self.theta = vel
            self.v = np.sqrt(self.vx**2 + self.vy**2)
        
        new_traj_data = torch.zeros(1,1,5)
        new_traj_data[0,0,:] = torch.FloatTensor([self.px, self.py, self.vx, self.vy, self.time_step])
        self.generated_traj_UptoNow = torch.cat((self.generated_traj_UptoNow, new_traj_data),0)


    def get_goal_position(self):
        return [self.gx, self.gy]
    
    def get_curr_dataset_timestep(self):
        return self.dataset_curr_timestep
    
    def get_full_state_list_noV(self):
        return [self.px, self.py, self.radius, self.gx, self.gy, self.v_pref, self.theta]
           




