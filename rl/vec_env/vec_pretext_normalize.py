import numpy as np
import torch
from .running_mean_std import RunningMeanStd
from . import VecEnvWrapper
from ped_pred.wrapper.move_plan_interface_multi_env_parallel import MovePlanPredInterfaceMultiEnv


class VecPretextNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that processes the observations and rewards, used for PCG predictors
    and returns from an environment.
    config: a Config object
    test: whether we are training or testing
    """

    def __init__(self, venv, ob=False, ret=False, clipob=10., cliprew=10., 
                 gamma=0.99, epsilon=1e-8, config=None, test=False):
        VecEnvWrapper.__init__(self, venv)

        self.config = config
        self.device = torch.device(self.config.training.device)
        if test:
            self.num_envs = 1
        else:
            self.num_envs = self.config.env.num_processes
          
        self.max_human_num = config.sim.max_human_num

        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = torch.zeros(self.num_envs).to(self.device)
        self.gamma = gamma
        self.epsilon = epsilon

        self.buffer_len = self.config.sim.obs_len

        self.predictor = MovePlanPredInterfaceMultiEnv(self.config, num_env=self.num_envs)

    def talk2Env_async(self, data):
        self.venv.talk2Env_async(data)

    def talk2Env_wait(self):
        outs = self.venv.talk2Env_wait()
        return outs

    def step_wait(self):
        obs, rews, done, infos = self.venv.step_wait()

        obs, rews = self.process_obs_rew(obs, done, rews=rews)

        return obs, rews, done, infos

    def _obfilt(self, obs):
        if self.ob_rms and self.config.RLTrain:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                           -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):

        obs = self.venv.reset()

        obs, _ = self.process_obs_rew(obs, np.zeros(self.num_envs))

        return obs

    '''
    1. Process observations: 
    Run inference on pred model with past obs as inputs, 
    fill in the predicted trajectory in O['spatial_edges']
    
    2. Process rewards: 
    Calculate the probablity of colliding with predicted future traj of pedestrians.
    Based on this collision probablity, calculate the prediction reward and add it
    to the original reward
    '''

    def process_obs_rew(self, O, done, rews=0.):
        '''
        O: robot_node: [nenv, 1, 7], 
           spatial_edges: [nenv, max_human_num, 2*(1+predict_steps)],
           temporal_edges: [nenv, 1, 2],
           agent_pos: [nenv, obs_len, max_human_num, 5], 
           agent_mask: [nenv, obs_len, max_human_num]
           robot_plan: [nenv, pred_len, 1, 5]
        '''
        
        # ped_pred: (n_env, num_peds, pred_seq_len, 5)
        # O['robot_pos']: (n_env, pred_len, 1, 5)
        if self.config.sim.predict_method == 'inferred':
            ped_pred, pred_avail_mask, ped_pred_cov = self.predictor.forward(O['ped_pos'], 
                                                                             O['ped_mask'], 
                                                                             O['veh_pos'], 
                                                                             O['veh_mask'], 
                                                                             O['robot_pos'], 
                                                                             O['robot_plan'])
        elif self.config.sim.predict_method == 'truth':
            ped_pred = O['GT_pred_pos'].permute(0, 2, 1, 3) # (n_env, max_num_peds, pred_seq_len, 5)
            pred_avail_mask = O['GT_pred_mask'].permute(0, 2, 1) # (n_env, max_num_peds, pred_len)

        ped_pred = ped_pred.to(self.device)
        pred_avail_mask = pred_avail_mask.to(self.device)
        if self.config.sim.uncertainty_aware and self.config.sim.predict_method == 'inferred':
            ped_pred_cov = ped_pred_cov.to(self.device) # (n_env, max_num_peds, pred_seq_len, 2, 2)

        
        next_obs_ped_pos = torch.zeros_like(O['ped_pos']) # [nenv, obs_len, max_human_num, 5]
        next_obs_ped_pos[:, :-1, :, :] =  O['ped_pos'][:, 1:, :, :]
        if self.config.sim.predict_method == 'truth':
            _pred_avail_mask = pred_avail_mask[:,:,0].unsqueeze(dim=2).repeat(1, 1, 5) # (n_env,max_num_peds,5)
        else: # inferred
            _pred_avail_mask = pred_avail_mask.repeat(1, 1, 5)
        valid_first_pos_pred = torch.logical_and(ped_pred[:, :, 0, :], _pred_avail_mask) # (n_env,max_num_peds,5)
        next_obs_ped_pos[:, -1, :, :] = ped_pred[:, :, 0, :] * valid_first_pos_pred # (n_env,max_num_peds,5)

        next_obs_ped_mask = torch.zeros(O['ped_mask'].shape, dtype=torch.int8) # [nenv, obs_len, max_human_num]
        next_obs_ped_mask[:, :-1, :] = O['ped_mask'][:, 1:, :]
        next_obs_ped_mask[:, -1, :] = O['ped_mask'][:, -1, :]


        # compute the prediction reward
        # add penalties if the collision probablitiy between the robot's state 
        # and pedestrians' future pos is above a threshold

        # [n_env, max_human_num, pred_seq_len, 2]
        relative_pos = ped_pred[:, :, :, :2] - O['robot_node'][:, :, :2].unsqueeze(1)
        if self.config.sim.uncertainty_aware and self.config.sim.predict_method == 'inferred':
            # relative_pos: (n_env, max_num_peds, pred_seq_len, 2)
            relative_pos = relative_pos.unsqueeze(dim=4) # (n_env, max_num_peds, pred_seq_len, 2, 1)
            inv_cov = torch.inverse(ped_pred_cov.reshape(-1, 2, 2))
            inv_cov = inv_cov.reshape(self.num_envs, self.max_human_num, 
                                      self.config.sim.pred_len, 2, 2) # (n_env, max_num_peds, pred_seq_len, 2, 2)
            d_MD2 = torch.matmul(torch.matmul(torch.transpose(relative_pos, 3, 4) , inv_cov), 
                                 relative_pos) # (n_env, max_num_peds, pred_seq_len, 1, 1)
            d_MD2 =d_MD2.squeeze(-1).squeeze(-1) # (n_env, max_num_peds, pred_seq_len)
            den = torch.det(2*torch.pi*ped_pred_cov.reshape(-1, 2, 2))
            den = torch.pow(den, 0.5).reshape(self.num_envs, self.max_human_num, 
                                              self.config.sim.pred_len) # (n_env, max_num_peds, pred_seq_len)
            r_sum = self.config.robot.radius + self.config.peds.radius + \
                                               self.config.reward.discomfort_dist 
            V_s = (4. / 3.) * torch.pi * (r_sum) ** 3 
            
            d_MD_threshold = -2 * torch.log(0.1 * den / V_s) # (n_env, max_num_peds, pred_seq_len)
            collision_idx = d_MD2 < d_MD_threshold # (n_env, max_num_peds, pred_seq_len)
            collision_idx = torch.logical_and(collision_idx, pred_avail_mask) 

        else:
           # collision index baseed on the deterministic distance between the robot and pedestrians future pos
            collision_idx = torch.norm(relative_pos, dim=-1) < self.config.robot.radius + \
                self.config.peds.radius + self.config.reward.discomfort_dist  # [n_env, max_human_num, pred_seq_len]

            # [1,1, pred_len]
            # mask out invalid predictions
            # [nenv, max_human_num, pred_len] AND [nenv, max_human_num, 1]
            collision_idx = torch.logical_and(collision_idx, pred_avail_mask)

        coefficients = 2. ** torch.arange(2, self.config.sim.pred_len + 2, 
                                          device=self.device).reshape(
                                          (1, 1, self.config.sim.pred_len)) # 4, 8, 16, 32, 64, 128

        # [1, 1, pred_len]
        collision_penalties = 2 * self.config.reward.collision_penalty / coefficients

        # [nenv, max_human_num, pred_len]
        reward_future = collision_idx.to(torch.float)*collision_penalties
        # [nenv, max_human_num, predict_steps] -> [nenv, max_human_num*predict_steps] -> [nenv,]
        reward_future, _ = torch.min(reward_future.reshape(self.num_envs, -1), dim=1)
        rews = rews + reward_future.reshape(self.num_envs, 1).cpu().numpy()

        # get observation back to env
        robot_pos = O['robot_node'][:, :, :2].unsqueeze(1)

        # convert from positions in world frame to robot frame
        ped_pred[:, :, :, :2] = ped_pred[:, :, :, :2] - robot_pos

        pred_avail_mask = pred_avail_mask.repeat(1, 1, self.config.sim.pred_len * 2)
        if self.config.sim.uncertainty_aware and self.config.sim.predict_method == 'inferred':
            visibility_mask = O['visible_masks'].unsqueeze(2).repeat(1,
                                                                1, self.config.sim.pred_len * 6)    
            new_spatial_edges_part1 = ped_pred[:, :, :, :2].reshape(self.num_envs, 
                                                                    self.max_human_num, -1)
            new_spatial_edges_part2 = ped_pred_cov.reshape(self.num_envs, self.max_human_num, 
                                                           self.config.sim.pred_len, 4)
            new_spatial_edges_part2 = new_spatial_edges_part2.reshape(self.num_envs, 
                                                                      self.max_human_num, -1)
            new_spatial_edges = torch.cat((new_spatial_edges_part1, new_spatial_edges_part2),
                                           dim=2)
            # taking into account only visible pedestrians to the robot
            O['spatial_edges'][:, :, 6:][visibility_mask] = new_spatial_edges[visibility_mask]
            
            # preparing the ped_pred to be passed to talk2Env for rendering
            rel_prediction = torch.cat((ped_pred[:, :, :, :2], 
                                        ped_pred_cov.reshape(self.num_envs, self.max_human_num,
                                                             self.config.sim.pred_len, 4)), dim=3)
            rel_prediction = rel_prediction.reshape(self.num_envs, self.max_human_num, -1) # last dimesnion is 6*pred_len
            _visibility_mask = O['visible_masks'].unsqueeze(2).repeat(1, 1, 
                                                                    self.config.sim.pred_len * 6) 
            # taking into account only visible pedestrians to the robot
            O['pred_pos'][:, :, 6:][_visibility_mask] = rel_prediction[_visibility_mask]

        else:
            visibility_mask = O['visible_masks'].unsqueeze(2).repeat(1, 1, 
                                                                     self.config.sim.pred_len * 2)    
            new_spatial_edges = ped_pred[:, :, :, :2].reshape(self.num_envs, 
                                                              self.max_human_num, -1)
            # taking into account only visible pedestrians to the robot
            O['spatial_edges'][:, :, 2:][visibility_mask] = new_spatial_edges[visibility_mask]

        # sort all humans by distance to robot
        # [nenv, max_human_num]
        if self.config.sim.uncertainty_aware and self.config.sim.predict_method == 'inferred':
            hr_dist_cur = torch.linalg.norm(O['spatial_edges'][:, :, :2], dim=-1)
            sorted_idx = torch.argsort(hr_dist_cur, dim=1)
            _hr_dist_cur = torch.linalg.norm(O['pred_pos'][:, :, :2], dim=-1)
            _sorted_idx = torch.argsort(_hr_dist_cur, dim=1) 
        else:
            hr_dist_cur = torch.linalg.norm(O['spatial_edges'][:, :, :2], dim=-1)
            sorted_idx = torch.argsort(hr_dist_cur, dim=1)
        for i in range(self.num_envs):
            O['spatial_edges'][i] = O['spatial_edges'][i][sorted_idx[i]]
            O['visible_masks'][i] = O['visible_masks'][i][sorted_idx[i]]
            if self.config.sim.uncertainty_aware and self.config.sim.predict_method == 'inferred':
                O['pred_pos'][i] = O['pred_pos'][i][_sorted_idx[i]]


        obs = {'robot_node': O['robot_node'],
                    'spatial_edges': O['spatial_edges'],
                    'temporal_edges': O['temporal_edges'],
                    'visible_masks': O['visible_masks'],
                    'detected_human_num': O['detected_human_num'],
                    'ped_pos': next_obs_ped_pos,
                    'ped_mask': next_obs_ped_mask,
                    'veh_pos': O['veh_pos'],
                    'veh_mask': O['veh_mask'],
                    'robot_pos': O['robot_pos'],
                    'robot_plan': O['robot_plan'],
                }
        
        if self.config.args.consider_veh:

            obs['spatial_edges_veh'] = O['spatial_edges_veh']
            obs['detected_vehicle_num'] = O['detected_vehicle_num']

        if self.config.sim.predict_method == 'truth':
            obs['GT_pred_pos'] = O['GT_pred_pos']
            obs['GT_pred_mask'] = O['GT_pred_mask']

        if self.config.sim.uncertainty_aware and self.config.sim.predict_method == 'inferred':
            obs['pred_pos'] = O['pred_pos']


        return obs, rews
