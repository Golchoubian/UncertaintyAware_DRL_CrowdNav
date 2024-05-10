import numpy as np
import torch
import gym
import os
from numpy.linalg import norm

from ped_pred.DataLoader import DataLoader
from ped_sim.envs.utils.robot import Robot
from ped_sim.envs.utils.peds import Peds
from ped_sim.envs.utils.vehs import Vehs

from ped_sim.envs.utils.action import ActionRot, ActionXY
from ped_sim.envs.utils.info import *


class PedSim(gym.Env):

    def __init__(self):

        self.frame = None

        self.robot = None
        self.peds = None
        self.vehs = None
        self.start_pos = None
        self.goal_pos = None
        self.scen_max_human_num = None

        self.config = None
        self.dataloader = None
        self.ped_traj_pred = None

        self.scenario_ind = None
        self.scenario_num = None
        self.scenario_length = None
        self.time_out_f = None
        self.extended_time_out = None

        self.observation_space = None
        self.action_space = None

        self.thisSeed = None  # the seed will be set when the env is created
        self.nenv = None  # the number of env will be set when the env is created.

        self.phase = None # This is set in envs (when env_num>1 is train)
        self.test_case = None  # the test scenario number to be run.

        # for render
        self.render_axis = None
        self.render_figure = None

        self.pred_method = None
        self.ped_prediction = None
        self.robot_planned = None

        self.desiredVelocity = [0.0, 0.0]

        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None

        self.epo_info = None

        self.case_counter = None

        self.robot_fov = None # limit FOV


    def step(self, action):
        """
        step function
        Execute robto's actions,
        detect collision, update environment and return (ob, reward, done, info)
        """

        if not self.config.robot.human_driver:

            # clip the action to obey robot's constraint
            action = self.robot.policy.clip_action(action, self.robot.v_pref)

        self.frame += 1

        if not self.config.robot.human_driver:
            # apply action and update the robot's state
            self.robot.step(action)
            ob = {}
            ob['robot'] = self.robot.get_curr_ob(self.frame)

        else:
            # the data of the actual human driven trajectory in the dataset
            human_driver_next_pose = self.robot.true_pose[self.frame,:,:]
            previous_theta = self.robot.theta

            # directly update the new pose of the robot here instead of calling robot.step
            self.robot.px = human_driver_next_pose[0,0]
            self.robot.py = human_driver_next_pose[0,1]
            self.robot.vx = human_driver_next_pose[0,2]
            self.robot.vy = human_driver_next_pose[0,3]
            self.robot.v = np.sqrt(self.robot.vx**2 + self.robot.vy**2)
            self.robot.theta = self.robot.calculate_theta(self.robot.vx, self.robot.vy)

            # derive the action having the next state
            heading_change = self.robot.theta - previous_theta
            action = ActionRot(self.robot.v.item(), heading_change.item())

            ob = {}
            curr_obs = self.robot.true_pose[self.frame-self.obs_length+1:self.frame+1,:,:]
            self.robot.curr_obs = curr_obs
            curr_mask = self.robot.mask[self.frame-self.obs_length+1:self.frame+1,:]
            curr_ob = {'pos': curr_obs,
                    'mask': curr_mask}
            ob['robot'] = curr_ob


        ob['peds'] = self.peds.get_curr_ob(self.frame)
        ob['vehs'] = self.vehs.get_curr_ob(self.frame)

        reward, done, episode_info = self.calc_reward(action)
        info = {'info': episode_info}

        _ob = self.generate_ob(ob, sort=True)

        return _ob, reward, done, info 


    def get_human_availability(self, ob, veh=False):
        '''
        returns a list of True/False values in the index position of pos tensor,
        True where the pedestrian is present at the current step.
        will be used as a mask later on to extract the valid prediction data
        '''

        mask = ob['mask']

        if veh:
            last_scen_mask = mask[self.obs_length-1, :]  # just looking at the current step, 
            # for vehicle the mask is of size (max_vehicle_num, obs_len+pred_len) 
            complete_mask = torch.zeros((1, self.config.sim.max_vehicle_num), dtype=torch.int8)
            complete_mask[0, :self.scen_max_vehicle_num] = last_scen_mask
        else:
            last_scen_mask = mask[-1, :]  # just looking at the current step
            # for pedestrian the mask is of size (max_ped_num, obs_len)
            complete_mask = torch.zeros((1, self.config.sim.max_human_num), dtype=torch.int8)
            complete_mask[0, :self.scen_max_human_num] = last_scen_mask
    
        human_ind = complete_mask.bool().squeeze().tolist()
        num_available_human = torch.sum(last_scen_mask)
       

        return human_ind, num_available_human
    
    
    def get_human_visibility(self, ob, veh=False):
        '''
        returns a list of True/False values in the index position of pos tensor,
        True where the pedestrian is present in the robot's FOV (visible to the robot).
        '''

        mask = ob['mask']

        if veh:
            pres_time_mask = mask[self.obs_length-1, :]  # just looking at the current step, 
            # for vehicle the mask is of size (max_vehicle_num, obs_len+pred_len) 
            complete_visib_mask = torch.zeros((1, self.config.sim.max_vehicle_num), 
                                              dtype=torch.int8)
            neighbor_agent = ob['pos'][self.obs_length-1,:,:]
            neigbor_radius = self.vehs.radius
            agent_num = self.scen_max_vehicle_num
        else:
            pres_time_mask = mask[-1, :]  # just looking at the current step
            # for pedestrian the mask is of size (max_ped_num, obs_len)
            complete_visib_mask = torch.zeros((1, self.config.sim.max_human_num), 
                                              dtype=torch.int8)
            neighbor_agent = ob['pos'][-1,:,:]
            neigbor_radius = self.peds.radius
            agent_num = self.scen_max_human_num
        
        for i in range(agent_num):
            if pres_time_mask[i]:
                complete_visib_mask[0, i] = int(self.detect_visible(self.robot, 
                                                                    neighbor_agent[i,:], 
                                                                    neigbor_radius))

        neighbor_visib_ind = complete_visib_mask.bool().squeeze().tolist()
        num_visib_neighbor = torch.sum(complete_visib_mask)
       

        return neighbor_visib_ind, num_visib_neighbor
    
  
    def detect_visible(self, state1, state2, neigbor_radius, custom_fov=None, 
                       custom_sensor_range=None):
        '''
        # Caculate whether agent2 is in agent1's FOV
        # Not the same as whether agent1 is in agent2's FOV!!!!
        # arguments:
        # state1; robots state (class of robto)
        # state2; other agents state (in form of pos tensor)
        # return value:
        # return True if state2 is visible to state1, else return False
        '''
      
        real_theta = state1.theta

        # angle of center line of FOV of agent1
        v_fov = [np.cos(real_theta), np.sin(real_theta)]

        # angle between agent1 and agent2
        v_12 = [state2[0] - state1.px, state2[1] - state1.py]
        
        # angle between center of FOV and agent 2
        v_fov = v_fov / np.linalg.norm(v_fov)
        v_12 = v_12 / np.linalg.norm(v_12)
        offset = np.arccos(np.clip(np.dot(v_fov, v_12), a_min=-1, a_max=1))

        if custom_fov:
            fov = custom_fov
        else:
            fov = self.robot_fov
          

        if np.abs(offset) <= fov / 2:
            inFov = True
        else:
            inFov = False

        # detect whether state2 is in state1's sensor_range
        dist = np.linalg.norm(
                [state1.px - state2[0], state1.py - state2[1]]) - neigbor_radius - state1.radius
        if custom_sensor_range:
            inSensorRange = dist <= custom_sensor_range
        else:
            inSensorRange = dist <= self.robot.sensor_range
           
        return (inFov and inSensorRange)
    

    

    def reset(self, phase='train', test_case=None):
        """
        Reset the environment
        """

        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case = self.test_case

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            # test case is passed in to calculate specific seed to generate case
            self.case_counter[phase] = test_case

        np.random.seed(self.case_counter[phase] + self.thisSeed)

        self.scenario_ind += 1

        if self.scenario_ind == 0:
            # for the first step of training
            scen_data = self.dataloader.get_scenario(self.scenario_ind, first_time=True, 
                                                     test_case = test_case)

        elif self.scenario_ind >= self.dataloader.num_scenarios:
            '''
            When one round of training on all scenarios is done
            reset the cycle of scenario extraction with shuffle set to True
            we shuffle the scenarios only when starting the scenarios all over again
            '''
            self.scenario_ind = 0
            scen_data = self.dataloader.get_scenario(self.scenario_ind, shuffle=True)
        else:
            scen_data = self.dataloader.get_scenario(self.scenario_ind)

        (ego_data, EgoList, ped_data, PedsList, veh_data, VehsList, scenario_num) = scen_data

        self.scenario_num = scenario_num
        self.scenario_length = len(ego_data)

        pos_ego_scen, mask_ego_scen = self.dataloader.convert_proper_array(ego_data, EgoList, 
                                                                           self.scenario_length)
        pos_ped_scen, mask_ped_scen = self.dataloader.convert_proper_array(ped_data, PedsList, 
                                                                           self.scenario_length)
        pos_veh_scen, mask_veh_scen = self.dataloader.convert_proper_array(veh_data, VehsList, 
                                                                           self.scenario_length)

        self.scen_max_human_num = pos_ped_scen.shape[1]
        self.scen_max_vehicle_num = pos_veh_scen.shape[1]

        # setting the goal position of the robot before extending the trajectories
        self.robot.gx = pos_ego_scen[-1, 0, 0]
        self.robot.gy = pos_ego_scen[-1, 0, 1]

        self.simulation_lenght = self.scenario_length - self.obs_length

        self.time_out_f = self.scenario_length + self.extended_time_out

        ext_len = self.extended_time_out + self.pred_length + 1 
        # +1 for geberate_ob to work in step function at the last extended scenario
        pos_veh_scen_ext, mask_veh_ext = self.dataloader.extend_traj(pos_veh_scen, 
                                                                     mask_veh_scen, ext_len)
        # The extension for the ego will be used as a ground truth
        pos_ego_scen_ext, mask_ego_ext = self.dataloader.extend_traj(pos_ego_scen, 
                                                                     mask_ego_scen, ext_len)
        # In an envrionment where we do no have predictions, I also need to extended
        # the ped trajectories beyond the real data scenario length assuming constnat
        # velcoity for them
        pos_ped_scen_ext, mask_ped_ext = self.dataloader.extend_traj(pos_ped_scen, 
                                                                     mask_ped_scen, ext_len)

        self.robot.set_scen_data(pos_ego_scen_ext, mask_ego_ext)
        self.peds.set_scen_data(pos_ped_scen_ext, mask_ped_ext)
        self.vehs.set_scen_data(pos_veh_scen_ext, mask_veh_ext)

        self.start_pos = self.robot.get_position()
        self.goal_pos = self.robot.get_goal_position()

        # the simulation starts at frame = obs_lenght to have enoguht obervation at the start
        self.frame = self.obs_length - 1

        ob = {}
        ob['robot'] = self.robot.get_curr_ob(self.frame)
        ob['peds'] = self.peds.get_curr_ob(self.frame, reset=True)
        ob['vehs'] = self.vehs.get_curr_ob(self.frame)

        self.desiredVelocity[0] = (self.robot.vx.item()**2 + self.robot.vy.item()**2)**0.5

        # initialize potential and angular potential
        rob_goal_vec = np.array([self.robot.gx, self.robot.gy]) - \
                       np.array([self.robot.px, self.robot.py])
        self.potential = -abs(np.linalg.norm(rob_goal_vec))

        _ob = self.generate_ob(ob, sort=True, reset=True)

        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + \
                                     int(1*self.nenv)) % self.case_size[phase]

        return _ob 
    


    def configure(self, config):

        self.config = config

        self.predict_steps = config.sim.pred_len

        self.obs_length = config.sim.obs_len
        self.pred_length = config.sim.pred_len
        self.seq_length = self.obs_length + self.pred_length

        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty
        self.discomfort_dist = config.reward.discomfort_dist
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor
        self.veh_collision_penalty = config.reward.veh_collision_penalty
        

        self.robot_fov = np.pi * config.robot.FOV

        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        self.dataloader = DataLoader(phase=self.phase)
        self.scenario_ind = -1

        self.case_size = {self.phase: self.dataloader.num_scenarios}

        rob_RL = Robot(config, 'robot')
        self.set_robot(rob_RL)

        self.peds = Peds(config, 'ped')
        self.vehs = Vehs(config, 'veh')

        # we extend the scenarios from the datset to give the robot more time to reach the goal
        self.extended_time_out = self.config.sim.extended_time_out
        self.time_step = config.env.time_step

        self.pred_method = config.sim.predict_method


    def set_robot(self, robot):
        self.robot = robot

        d = {}
        # robot node: px, py, r, gx, gy, v_pref, theta
        d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7,), 
                                         dtype=np.float32)
        # temporal edges for the robot from time t-1 to t
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), 
                                             dtype=np.float32)
        '''
        format of spatial_edges: [max_human_num, [state_t, state_(t+1), ..., state(t+self.pred_steps)]]
        '''

        d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.config.sim.max_human_num, 2),
                                            dtype=np.float32)

        # whether each human is visible to robot (ordered by human ID, should not be sorted)
        d['visible_masks'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.config.sim.max_human_num,),
                                            dtype=np.bool)

        # number of humans detected at each timestep
        d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), 
                                                 dtype=np.float32)

        if self.config.args.consider_veh:
            d['spatial_edges_veh']  = gym.spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.config.sim.max_vehicle_num, 2),
                                            dtype=np.float32)
            # number of vehicles detected at each timestep
            d['detected_vehicle_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, 
                                                       shape=(1,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(d)

        high = np.inf * np.ones([2, ])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

 
    def generate_ob(self, ob, sort=False, reset=False):
        """Generate observation for reset and step functions"""
        # since prediction is not used spatial edges should be sort here

        _ob = {}

        _ob['robot_node'] = self.robot.get_full_state_list_noV()

        # edges
        _ob['temporal_edges'] = np.array([self.robot.vx, self.robot.vy])

        all_spatial_edges = np.ones((self.config.sim.max_human_num, 2)) * np.inf
        human_visibility, num_available_human = self.get_human_visibility(ob['peds'])

        for i in range(self.scen_max_human_num):
            if human_visibility[i]:
                relative_pos = np.array([ob['peds']['pos'][-1, i, 0] - self.robot.px, 
                                         ob['peds']['pos'][-1, i, 1] - self.robot.py])
                all_spatial_edges[i, :2] = relative_pos

        _ob['visible_masks'] = np.zeros(self.config.sim.max_human_num, dtype=np.bool)

        # sort all humans by distance (invisible humans will be in the end automatically)
        if sort:
            _ob['spatial_edges'] = np.array(sorted(all_spatial_edges, 
                                                   key=lambda x: np.linalg.norm(x)))
            # after sorting, the visible humans must be in the front
            if num_available_human > 0:
                _ob['visible_masks'][:num_available_human] = True
        else:
            _ob['spatial_edges'] = all_spatial_edges
            _ob['visible_masks'][:self.scen_max_human_num] = human_visibility[:self.scen_max_human_num]
    
        _ob['spatial_edges'][np.isinf(_ob['spatial_edges'])] = 2 * self.robot.sensor_range

        _ob['detected_human_num'] = num_available_human
        # if no human is detected, assume there is one dummy human to make the pack_padded_sequence work
        if _ob['detected_human_num'] == 0:
            _ob['detected_human_num'] = 1

        # vehicle spatial edges
        if self.config.args.consider_veh:
            all_spatial_edges_veh = np.ones((self.config.sim.max_vehicle_num, 2)) * np.inf
            vehicle_availability, num_available_vehicle = self.get_human_visibility(ob['vehs'], 
                                                                                    veh=True)

            for i in range(self.scen_max_vehicle_num):
                if vehicle_availability[i]:
                    relative_pos_veh = np.array(
                        [ob['vehs']['pos'][self.obs_length-1, i, 0] - \
                        self.robot.px, ob['vehs']['pos'][self.obs_length-1, i, 1] - self.robot.py])
                    all_spatial_edges_veh[i, :2] = relative_pos_veh
            # I will alway sort vehicles spatial edges since for vehciles we never do prediction 
            _ob['spatial_edges_veh'] = np.array(sorted(all_spatial_edges_veh, 
                                                       key=lambda x: np.linalg.norm(x)))
            _ob['spatial_edges_veh'][np.isinf(_ob['spatial_edges_veh'])] = 2 * self.robot.sensor_range

            _ob['detected_vehicle_num'] = num_available_vehicle
            # if no vehicle is detected, assume there is one dummy vehicle to make the pack_padded_sequence work
            if _ob['detected_vehicle_num'] == 0:
                _ob['detected_vehicle_num'] = 1


        # A robot plan is generated for rendering.
        # We use a constant velocity for this initial plan
        robot_planned_traj = self.robot.planned_traj(reset=reset)
        self.robot_planned = robot_planned_traj

        self.ob = _ob
        self.ped_ob = ob['peds']['pos'][-1,:,:]
        self.ped_visibility = human_visibility[:]

        return _ob

  
    def calc_reward(self, action):
        # collision detection
        dmin = float('inf')
        speedAtdmin = float('inf')

        danger_dists = []
        collision = False

        # collision check with humans
        for i in range(self.scen_max_human_num):
            if self.peds.curr_mask[-1, i] != 0:  # The pedestrian is present at this time step
                dx = self.peds.px[i] - self.robot.px
                dy = self.peds.py[i] - self.robot.py
                closest_dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.peds.radius - self.robot.radius
                closest_dist = closest_dist.numpy()


                if closest_dist < self.discomfort_dist:
                    danger_dists.append(closest_dist)

                if self.config.robot.human_driver:
                    # Human driver in the dataset of course had no collision,
                    # so we count closest_dist<0 also as an intrusion rather than collision
                    # closest_dist<0 in the human_driven trajecotry is mainly due to the 
                    # simplified assumption of the robot shape as a circle in our model
                    if closest_dist < dmin:
                        dmin = closest_dist
                        speedAtdmin = self.robot.v
                else:
                    if closest_dist < 0:
                        collision = True
                        break
                    elif closest_dist < dmin:
                        dmin = closest_dist
                        speedAtdmin = self.robot.v
            
        # collision check with vehicles
        if self.config.args.consider_veh:
            collision_veh = False
            veh_closest_dist = []
            for i in range(self.scen_max_vehicle_num):
                if self.vehs.curr_mask[self.obs_length-1, i] != 0:  # The vehicle is present at this time step
                    dx_veh = self.vehs.px[i] - self.robot.px
                    dy_veh = self.vehs.py[i] - self.robot.py
                    closest_dist_veh = (dx_veh ** 2 + dy_veh ** 2) ** (1 / 2) - self.vehs.radius - self.robot.radius
                    closest_dist_veh = closest_dist_veh.numpy()
                    veh_closest_dist.append(closest_dist_veh)

                    if closest_dist_veh < 0: 
                        # closest_dist_veh threshold value should be ideally different
                        # for vehicles in the same line and those in different lines but
                        # lets keep it simple for now by reducing the safety zone to zero 
                        # to not detected conflicts between cars in different lane
                        collision_veh = True
                        break

        # check if reaching the goal
        if self.robot.kinematics == 'unicycle':
            goal_radius = 2
        else:
            goal_radius = self.robot.radius
        reaching_goal = norm(
            np.array(self.robot.get_position()) - np.array(self.robot.get_goal_position())) < goal_radius


        danger_cond = dmin < self.discomfort_dist
        min_danger_dist = dmin

        if ((self.frame) >= self.time_out_f):
            reward = 0
            done = True
            episode_info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            episode_info = Collision(self.robot.v)
        elif self.config.args.consider_veh and collision_veh:
            reward = self.veh_collision_penalty
            done = True
            episode_info = Collision_Vehicle()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            episode_info = ReachGoal()

        elif danger_cond:
            # # speed independent danger penalty:
            # reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            # # speed dependent danger penalty:
            reward = (1+(self.robot.v/self.robot.v_pref).item()) * \
                     (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            episode_info = Danger(min_danger_dist, speedAtdmin)

        else:
            # potential reward
            if self.robot.kinematics == 'holonomic':
                pot_factor = 2
            elif self.robot.kinematics == 'bicycle':
                pot_factor = 0.5
            else:
                pot_factor = 1 # 0.8
            potential_cur = np.linalg.norm(
                np.array([self.robot.px, self.robot.py]) - np.array(self.robot.get_goal_position()))
            reward = pot_factor * (-abs(potential_cur) - self.potential)
            self.potential = -abs(potential_cur)

            done = False
            episode_info = Nothing()


        if self.robot.kinematics == 'unicycle':
            # add a rotational penalty
            r_spin = -200 * action.r ** 2

            # add a penalty for going backwards
            if action.v < 0:
                r_back = -2 * abs(action.v)
            else:
                r_back = 0.

            reward = reward + r_spin + r_back

        if self.robot.kinematics in {'double_integrator_unicycle', 'bicycle'}:
            # add a rotational penalty
            if self.robot.kinematics == 'double_integrator_unicycle':
                r_spin = -200 * action.u_alpha ** 2  # the coefficient should be adjusted
                reward = reward + r_spin
            elif self.robot.kinematics == 'bicycle':
                # penalizing not only the steering angle change rate 
                # but also any steering angle that is not zero
                r_spin_action  = -2000 * action.steering_change_rate ** 2
                r_spin = -10 * self.robot.phi ** 2
                reward = reward + r_spin_action + r_spin


        if self.phase == 'test':  # there is only one env to render
            self.epo_info = episode_info

        return reward, done, episode_info


    def render(self, mode='human'):

        from matplotlib import pyplot as plt

        ax = self.render_axis
        fig = self.render_figure

        fig.suptitle(f'Scenario #{self.scenario_num}', fontsize=15)
        ax.cla()
        ax.set_xlim(15, 95)  # for HBS
        ax.set_ylim(5, 60)  # for HBS
        ax.set_xlabel('x(m)', fontsize=15)
        ax.set_ylabel('y(m)', fontsize=15)
   
        Robot_obs_traj, _ = self.robot.get_traj_history()
        Peds_pos_obs, Peds_mask_obs = self.peds.get_traj_history()
        other_Vehs_traj, Vehs_mask = self.vehs.get_traj()

        planned_traj = self.robot_planned
        ego_traj = torch.cat((Robot_obs_traj, planned_traj), 0)

        num_peds = Peds_pos_obs.shape[1]
        num_vehs = other_Vehs_traj.shape[1]

        max_marker_size = 6
        min_marker_size = 2
        max_marker_size_ego = 10
        min_alpha = 0.2

        for ped_ind in range(num_peds):
            PedPresFrame = []
            for t in range(self.obs_length):
                if Peds_mask_obs[t, ped_ind] == 1:
                    PedPresFrame.append(t)
                    # plotting the observed trajectory for this time step
                    marker_size = min_marker_size + \
                                    ((max_marker_size-min_marker_size)/self.seq_length * t)
                    alpha_val = min_alpha + ((1-min_alpha)/self.obs_length * t)
                    ax.plot(Peds_pos_obs[t, ped_ind, 0], Peds_pos_obs[t, ped_ind, 1], c='r',
                            marker='o', markersize=marker_size, alpha=alpha_val)
            
            # plotting the raduis and the personal space of the pedestrian at its current position
            if Peds_mask_obs[self.obs_length-1, ped_ind] == 1:
                PedRadius = plt.Circle((Peds_pos_obs[self.obs_length-1, ped_ind, 0], 
                                        Peds_pos_obs[self.obs_length-1, ped_ind, 1]),
                                        self.peds.radius ,fill = False, ec='r', 
                                        linestyle='--', linewidth=0.5)
                ax.add_artist(PedRadius)

                # static PS
                PS_static = plt.Circle((Peds_pos_obs[self.obs_length-1, ped_ind, 0], 
                                        Peds_pos_obs[self.obs_length-1, ped_ind, 1]),
                                        self.discomfort_dist + self.peds.radius,
                                        fill = False, ec='y', linestyle='--', linewidth=0.5)
                ax.add_artist(PS_static)

                # # put pedestrian index on the plot
                # ax.annotate(ped_ind, # this is the text
                # (Peds_pos_obs[-1, ped_ind,0], Peds_pos_obs[-1, ped_ind,1]),
                # textcoords="offset points",
                # xytext=(0,1), # distance from text to points (x,y)
                # ha='center', fontsize=10) # horizontal alignment can be left, right or center

        # plotting the trajectory of other vehicles
        if self.config.args.consider_veh:
            for veh_ind in range(num_vehs):
                VehPresFrame = []
                for t in range(self.obs_length):
                    # plotting the trajecotry up to this time
                    marker_size = min_marker_size + \
                                    ((max_marker_size-min_marker_size)/self.obs_length * t)
                    alpha_val = min_alpha + ((1-min_alpha)/self.obs_length * t)
                    if Vehs_mask[t, veh_ind] == 1:
                        VehPresFrame.append(t)
                        ax.plot(other_Vehs_traj[t, veh_ind, 0], other_Vehs_traj[t, veh_ind, 1],
                                c='c', marker='o', markersize=marker_size, alpha=alpha_val)
                ax.plot(other_Vehs_traj[VehPresFrame, veh_ind, 0],
                        other_Vehs_traj[VehPresFrame, veh_ind, 1], c='c', linewidth=1.0, alpha=0.5)

        # plotting the trajecotry of ego
        for t in range(self.seq_length):
            if t < self.obs_length:
                marker_size_ego = min_marker_size + \
                                     ((max_marker_size_ego-min_marker_size)/self.obs_length * t)
                alpha_val = min_alpha + ((1-min_alpha)/self.obs_length * t)
                ax.plot(ego_traj[t, 0, 0], ego_traj[t, 0, 1], c='k',
                        marker='o', markersize=marker_size_ego, alpha=alpha_val)
            else:
                ax.plot(ego_traj[t, 0, 0], ego_traj[t, 0, 1], c='g',
                        marker='o', markersize=marker_size_ego, alpha=alpha_val)
        ax.plot(ego_traj[:self.obs_length, 0, 0], ego_traj[:self.obs_length, 0, 1], 
                c='k', linewidth=1.0, alpha=0.5)
        ax.plot(ego_traj[self.obs_length:, 0, 0], ego_traj[self.obs_length:, 0, 1], 
                c='g', linewidth=1.0, alpha=0.5)

        # drawing the raduis of the robot
        RobotRadius = plt.Circle((ego_traj[self.obs_length-1, 0, 0], 
                                  ego_traj[self.obs_length-1, 0, 1]),
                                  self.robot.radius ,fill = False, ec='k', 
                                  linestyle='--', linewidth=0.5)
        ax.add_artist(RobotRadius)

        # drawing the visibility range of the robot
        visibility_circle = plt.Circle((ego_traj[self.obs_length-1, 0, 0], 
                                        ego_traj[self.obs_length-1, 0, 1]),
                                       self.robot.sensor_range ,fill = False, 
                                       ec='k', linestyle='--', linewidth=0.5)
        ax.add_artist(visibility_circle)

        # Indicating the start and goal position of the ego for this scenario
        ax.plot(self.start_pos[0], self.start_pos[1], c='m', marker='P', markersize=15)
        ax.plot(self.goal_pos[0], self.goal_pos[1], c='m', marker='*', markersize=20)

        # legends
        ax.plot(-100, -100, c='k', marker='o', label='AV traj')
        ax.plot(-100, -100, c='g', marker='o', label='AV planned traj')
        ax.plot(-100, -100, c='r', marker='o', label='Ped observed traj')
        if self.config.args.consider_veh:
            ax.plot(-100, -100, c='c', marker='o', label='Other veh current traj')
        ax.scatter(-100, -100, c='m', marker='P', label='Start')
        ax.scatter(-100, -100, c='m', marker='*', label='Goal')

        legend = ax.legend(loc="upper left", prop={'size': 9}, ncol=1)
        legend.legendHandles[-1]._sizes = [120]
        legend.legendHandles[-2]._sizes = [110]

        ax.text(78, 8, 'frame: '+ str(self.frame), fontsize=14, color='0.3')

        # Writing the episode info on the plot
        if self.epo_info.__str__() == 'Collision':
            c = 'red'
        elif self.epo_info.__str__() == 'Vehicle Collision':
            c = 'red'
        elif self.epo_info.__str__() == 'Reaching goal':
            c = 'green'
        elif self.epo_info.__str__() == 'Timeout':
            c = 'blue'
        elif self.epo_info.__str__() == 'Intrusion':
            c = 'orange'
        else:
            c = 'black'

        if len(self.epo_info.__str__()) != 0 and self.epo_info.__str__() != 'None':
            ax.text(40, 8, self.epo_info.__str__(), fontsize=20, color=c)
            plt.pause(1)

        plt.pause(0.2)
        save_path = os.path.join(self.config.data.visual_save_path, 'plots')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, f'img_{self.frame}.png')
        plt.savefig(save_path)