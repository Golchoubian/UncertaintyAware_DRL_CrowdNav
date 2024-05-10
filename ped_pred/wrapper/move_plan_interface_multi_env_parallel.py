from ped_pred.pedestrian_trajectory import traj_prediction
from ped_pred.DataLoader import DataLoader
import torch
from ped_pred.helper import getCoef, cov_mat_generation

class MovePlanPredInterfaceMultiEnv(object):

    def __init__(self, config, num_env):

        self.nenv = num_env
        self.config = config

        if num_env>1:
            phase = 'train'
        else:
            phase = 'test'

        self.ped_traj_pred = traj_prediction(config)
        self.dataloader = DataLoader(phase=phase)
    

    def forward(self, ped_pos_all, ped_mask_all, veh_pos_all, veh_mask_all,
                robot_pos_all, robot_planned_traj):
        """
        inputs:
            - ped_pos_all and veh_pos_all:
                # torch tensor
                # (n_env, obs_seq_len, max_num_agent, 5)
                # where 5 includes [x, y, v_x, v_y, timestep]
            - ped_mask_all and veh_mask_all:
                # torch tensor
                # (n_env, obs_seq_len, max_num_agent)
            - robot_pos_all:
                # torch tensor
                # (n_env, obs_seq_len, 1, 5)
                # where 5 includes [x, y, v_x, v_y, timestep]
            - robot_planned_traj:
                # torch tesnor:
                # [n_env, planned_length, 1, 5]

        outputs:
            - output_traj:
                # torch "cpu"
                # (n_env, max_num_peds, pred_seq_len, 5)
                # where 5 includes [x ,y ,v_x, v_y, timestep]
            - output_binary_mask:
                # Specifies whether the ped is present in the last observed timestep to 
                # consider its prediction
                # torch "cpu"
                # (n_env, max_num_peds, 1)
            - output_traj_dist:
                # torch "cpu"
                # (n_env, max_num_peds, pred_seq_len, 5)
                # where 5 includes [mu_x, mu_y, sigma_x, sigma_y, correlation coefficient] 
            - output_traj_cov:
                # torch "cpu"
                # (n_env, max_num_peds, pred_seq_len, 2, 2)
                # where (2,2) includes [cov11, cov12, cov21, cov22] 
        """

        ped_pos_pred = []
        ped_dist_pred = []
        ped_cov_pred = []
        ped_availability_mask = []
        ped_pos_dist = []


        for i in range(self.nenv):

            ped_pos = ped_pos_all[i, :]
            ped_mask = ped_mask_all[i, :]

            veh_pos = veh_pos_all[i, :]
            veh_mask = veh_mask_all[i, :]

            robot_pos = robot_pos_all[i, :]
            robot_plan_env = robot_planned_traj[i, :]

            ob_ped_pos, ob_ped_mask, col_ind_pres_peds = self.filter_curr_ob(ped_pos, ped_mask)
            ob_veh_pos, ob_veh_mask, col_ind_pres_vehs = self.filter_curr_ob(veh_pos, veh_mask)


            pred_pos = torch.zeros((self.config.sim.pred_len, self.config.sim.max_human_num, 5))
            pred_dist = torch.zeros((self.config.sim.pred_len, self.config.sim.max_human_num, 5))
            pred_cov = torch.eye(2).unsqueeze(0).unsqueeze(0).repeat(self.config.sim.pred_len,
                                                                    self.config.sim.max_human_num,
                                                                    1,1) 
                                    # ones for making the inverse possible for not existing peds

            with torch.no_grad():
                # ped_pred: (pred_seq_len, num_peds, 5)
                ped_pred, dist_param = self.ped_traj_pred.forward(ob_ped_pos.cpu(), ob_ped_mask.cpu(),
                                                                ob_veh_pos.cpu(), ob_veh_mask.cpu(),
                                                                robot_pos.cpu(), robot_plan_env.cpu(), 
                                                                self.dataloader.timestamp)
                mux, muy, sx, sy, corr = getCoef(dist_param.cpu())
                scaled_param_dist = torch.stack((mux, muy, sx, sy, corr),2) 
                cov = cov_mat_generation(scaled_param_dist)

                pred_pos[:, col_ind_pres_peds, :] = ped_pred.cpu()
                pred_dist[:, col_ind_pres_peds, :] = dist_param.cpu()
                pred_cov[:, col_ind_pres_peds, :, :] = cov.cpu()
               
            prediction_mask = self.mask_creation(ped_pos, ped_mask) # shape: (pred_seq_len, num_peds)
            # this should be also build based on the ped_mask before filtering
            # to allow concatination later on and the information to match the pred_pos without sorting

            ped_pos_pred.append(pred_pos)
            ped_dist_pred.append(pred_dist)
            ped_cov_pred.append(pred_cov)
            ped_availability_mask.append(prediction_mask)

        # (n_env, obs_seq_len, num_peds, 5)
        output_traj = torch.stack(ped_pos_pred, dim=0)
        output_traj_dist = torch.stack(ped_dist_pred, dim=0)
        output_traj_cov = torch.stack(ped_cov_pred, dim=0)
        # (n_env, num_peds, obs_seq_len, 5)
        output_traj = output_traj.permute(0, 2, 1, 3)
        output_traj_dist = output_traj_dist.permute(0, 2, 1, 3)
        output_traj_cov = output_traj_cov.permute(0, 2, 1, 3, 4) # (n_env, num_peds, obs_seq_len, 2, 2)

        output_binary_mask = torch.stack(ped_availability_mask, dim=0)
        output_binary_mask = torch.unsqueeze(output_binary_mask, 2)

        return output_traj, output_binary_mask, output_traj_cov


    def mask_creation(self, ped_pos, ped_mask):
        # specifying humans that are present at the last time step of the observation

        binary_mask = torch.zeros_like(ped_pos, dtype=torch.bool)
        last_obs_mask = ped_mask[-1, :]
        pred_mask = last_obs_mask

        return pred_mask
    

    def filter_curr_ob(self, pos, mask):
        '''
        This function removes those columns in pos and mask
        that are assosicated to agents that are not present 
        in any of the frames during this current sequence length
        that we are looking at in the scenario 
        '''
        num_frame = pos.shape[0]
        # columns with value of zero are associated to those peds not available in this whole sequence
        num_avail_fram = torch.sum(mask, 0)
        columns_to_keep = (num_avail_fram != 0).nonzero()  # of shape (num_valid_columns, 1)
        # expanding this valid column number to all time rows in the pos and mask (first dimension)
        columns_to_keep_rp = torch.transpose(columns_to_keep, 1, 0).repeat(num_frame, 1)
        mask_filter = torch.gather(mask, dim=1, index=columns_to_keep_rp)
        columns_to_keep_rp2 = columns_to_keep_rp.unsqueeze(2).repeat(1, 1, pos.shape[2])
        pos_filter = torch.gather(pos, dim=1, index=columns_to_keep_rp2)

        col_indexs_to_keep = columns_to_keep_rp[0, :]

        return pos_filter, mask_filter, col_indexs_to_keep
