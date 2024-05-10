
import argparse
import pickle
import os
import torch
from torch.autograd import Variable
from ped_pred.Interaction import getInteractionGridMask, getSequenceInteractionGridMask
from ped_pred.helper import *


class traj_prediction():

    def __init__(self, config):

        parser = argparse.ArgumentParser()

        # Observed length of the trajectory parameter
        parser.add_argument('--obs_length', type=int, default=6,  # 6 for HBS
                            help='Observed length of the trajectory')
        # Predicted length of the trajectory parameter
        parser.add_argument('--pred_length', type=int, default=6,  # 6 for HBS
                            help='Predicted length of the trajectory')
        # cuda support
        parser.add_argument('--use_cuda', action="store_true", default=True,
                            help='Use GPU or not')

        parser.add_argument('--method', type=int, default=4, # this will be overwritten by the config.sim.predict_network
                            help='Method of lstm will be used (1 = social lstm, 3 = vanilla lstm, 4 = collision grid)')

        self.sample_args = parser.parse_args()

        seq_length = self.sample_args.obs_length + self.sample_args.pred_length

        # Define the path for the config file for saved args
        prefix = 'ped_pred/TrainedModel/'
        if config.sim.uncertainty_aware:
            method_name = config.sim.predict_network
            if config.sim.predict_network == 'CollisionGrid':
                self.sample_args.method = 4
                save_directory = os.path.join(prefix, 'uncertainty_aware_model/CollisionGrid/')
            elif config.sim.predict_network == 'VanillaLSTM':
                self.sample_args.method = 3
                save_directory = os.path.join(prefix, 'uncertainty_aware_model/VanillaLSTM/')
            else:
                # report error of unknown network
                raise ValueError('Invalid prediction network type (config.sim.predict_network)')
        else:
            self.sample_args.method = 4
            method_name = 'CollisionGrid'
            save_directory = os.path.join(prefix, 'uncertainty_unaware_model/')

        with open(os.path.join(save_directory, 'config.pkl'), 'rb') as f:
            self.saved_args = pickle.load(f)
        self.saved_args.use_cuda = self.sample_args.use_cuda

        save_tar_name = method_name+"_lstm_model"

        self.net = get_model(self.sample_args.method, self.saved_args, True)

        if self.sample_args.use_cuda:
            self.net = self.net.cuda()

        # Loading the trained model
        checkpoint_path = os.path.join(save_directory, save_tar_name+'.tar')
        if os.path.isfile(checkpoint_path):
            # print('Loading checkpoint')
            checkpoint = torch.load(checkpoint_path)
            model_epoch = checkpoint['epoch']
            self.net.load_state_dict(checkpoint['state_dict'])
            # print('Loaded checkpoint at epoch', model_epoch)
        else: 
            raise ValueError("No checkpoint found at", checkpoint_path)

    def forward(self, x_seq, mask, x_seq_veh, mask_veh, ego_obs, ego_pre_planned_traj, timestamp):


        orig_x_seq = x_seq.clone()
        orig_x_seq_veh = x_seq_veh.clone()

        # grid mask calculation
        if self.saved_args.method == 4:  # collision grid
            grid_seq, grid_TTC_seq = getSequenceInteractionGridMask(x_seq, mask, x_seq, mask,
                                                                    self.saved_args.TTC, self.saved_args.D_min, self.saved_args.num_sector,
                                                                    self.saved_args.use_cuda)

            grid_seq_veh_in_ped, grid_TTC_veh_seq = getSequenceInteractionGridMask(x_seq, mask, x_seq_veh, mask_veh,
                                                                                   self.saved_args.TTC_veh, self.saved_args.D_min_veh,
                                                                                   self.saved_args.num_sector,
                                                                                   self.saved_args.use_cuda,
                                                                                   is_heterogeneous=True, sequence_ego=ego_obs)

        x_seq, first_values_dict = position_change(x_seq, mask)
 
        # uncomment the following seven lines if the uncertainty-aware predictor relies on covaraince matrix as an input
        # # create the covaraince matrix using kalman filter and add it to x_seq
        # GT_filtered_disp, GT_covariance = KF_covariance_generator(x_seq, mask, timestamp) 
        # # add the covariances to x_seq
        # covariance_flat = GT_covariance.reshape(GT_covariance.shape[0], GT_covariance.shape[1], 4)
        # # x_seq up to here: [x, y, vx, vy, timestamp]
        # x_seq = torch.cat((x_seq, covariance_flat), dim=2) 
        # # x_seq: [x, y, vx, vy, timestamp, cov11, cov12, cov21, cov22]


        if self.sample_args.use_cuda:
            x_seq = x_seq.cuda()
            x_seq_veh = x_seq_veh.cuda()

        if self.saved_args.method == 3: # vanilla lstm
            # Extract the observed part of the trajectories
            obs_traj, obs_mask = x_seq.clone(), mask.clone()
            ret_x_seq, dist_param_seq = self.sample(obs_traj, obs_mask, self.sample_args, self.net, self.saved_args,
                                                            first_values_dict, orig_x_seq, None, None, None, None, None, None,
                                                            timestamp, ego_pre_planned_traj)
        elif self.saved_args.method == 4: # collision grid
            # Extract the observed part of the trajectories
            obs_traj, obs_mask, obs_grid, obs_grid_TTC = x_seq.clone(), mask.clone(), grid_seq.copy(), grid_TTC_seq.copy()
            obs_grid_veh_in_ped, obs_grid_TTC_veh = grid_seq_veh_in_ped[:
                                                                        self.sample_args.obs_length], grid_TTC_veh_seq[:self.sample_args.obs_length]

            ret_x_seq, dist_param_seq = self.sample(obs_traj, obs_mask, self.sample_args, self.net, self.saved_args,
                                                        first_values_dict, orig_x_seq, obs_grid, x_seq_veh, mask_veh,
                                                        obs_grid_veh_in_ped, obs_grid_TTC, obs_grid_TTC_veh, timestamp, ego_pre_planned_traj)

        last_obs_frame_mask = mask[-1, :]
        rp_mask = last_obs_frame_mask.unsqueeze(dim=0).repeat(self.sample_args.pred_length, 1)
        extended_mask = torch.cat((mask, rp_mask), 0)


        ret_x_seq = revert_postion_change(ret_x_seq.cpu(), extended_mask, first_values_dict,
                                               orig_x_seq, self.sample_args.obs_length, infer=True)


        dist_param_seq[:, :, 0:2] = revert_postion_change(dist_param_seq[:, :, 0:2].cpu(), extended_mask, first_values_dict,
                                                               orig_x_seq, self.sample_args.obs_length, infer=True)

        return ret_x_seq[self.sample_args.obs_length:, :, :5], dist_param_seq[self.sample_args.obs_length:, :]

    def sample(self, x_seq, mask, args, net, saved_args,
               first_values_dict, orig_x_seq, grid, x_seq_veh, mask_veh,
               grid_veh_in_ped, grid_TTC, grid_TTC_veh, timestamp, ego_pre_planned_traj):

        # Number of peds in the sequence
        numx_seq = x_seq.shape[1]
        if args.method == 4:
            numx_seq_veh = x_seq_veh.shape[1]

        with torch.no_grad():
            # Construct variables for hidden and cell states
            hidden_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
            cell_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
            if args.use_cuda:
                hidden_states = hidden_states.cuda()
                cell_states = cell_states.cuda()

        ret_x_seq = Variable(torch.zeros(args.obs_length+args.pred_length, numx_seq, x_seq.shape[2]))
        dist_param_seq = Variable(torch.zeros(args.obs_length+args.pred_length, numx_seq, 5))

        if args.use_cuda:
            ret_x_seq = ret_x_seq.cuda()
            dist_param_seq = dist_param_seq.cuda()


        # For the observed part of the trajectory
        for tstep in range(args.obs_length-1):

            if (args.method == 3): #  vanilla lstm
               # Do a forward prop
                out_obs, hidden_states, cell_states = net(x_seq[tstep, :, :].view(1, numx_seq, x_seq.shape[2]),
                                                           hidden_states, cell_states, 
                                                           mask[tstep, :].view(1, numx_seq))

            elif (args.method == 4): # collision grid
                # Do a forward prop
                grid_t = grid[tstep]
                grid_veh_in_ped_t = grid_veh_in_ped[tstep]
                grid_TTC_t = grid_TTC[tstep]
                grid_TTC_veh_t = grid_TTC_veh[tstep]
                if args.use_cuda:
                    grid_t = grid_t.cuda()
                    grid_veh_in_ped_t = grid_veh_in_ped_t.cuda()
                    grid_TTC_t = grid_TTC_t.cuda()
                    grid_TTC_veh_t = grid_TTC_veh_t.cuda()
                # first_net_start = time.time()   
                out_obs, hidden_states, cell_states = net(x_seq[tstep, :, :].view(1, numx_seq, x_seq.shape[2]), [grid_t],
                                                          hidden_states, cell_states, mask[tstep, :].view(1, numx_seq),
                                                          x_seq_veh[tstep, :, :].view(
                                                              1, numx_seq_veh, x_seq_veh.shape[2]),
                                                          [grid_veh_in_ped_t], mask_veh[tstep, :],
                                                          [grid_TTC_t], [grid_TTC_veh_t])

            # Extract the mean, std and corr of the bivariate Gaussian
            mux, muy, sx, sy, corr = getCoef(out_obs.cpu())

            dist_param_seq[tstep + 1, :, :] = out_obs.clone()

            # # Sample from the bivariate Gaussian
            # next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, mask[tstep, :])
            # ret_x_seq[tstep + 1, :, 0] = next_x
            # ret_x_seq[tstep + 1, :, 1] = next_y

            # Assigning the mean to the next state instead of sampling from the distrbution.
            next_x_mean = mux.clone().data
            next_y_mean = muy.clone().data
            ret_x_seq[tstep + 1, :, 0] = next_x_mean
            ret_x_seq[tstep + 1, :, 1] = next_y_mean

        if args.method == 4:
            # Last seen grid
            prev_grid = grid[-1].clone()
            prev_grid_veh_in_ped = grid_veh_in_ped[-1].clone()
            prev_TTC_grid = grid_TTC[-1].clone()
            prev_TTC_grid_veh = grid_TTC_veh[-1].clone()

        ret_x_seq[tstep + 1, :, 2:4] = x_seq[-1, :, 2:4]  # vx and vy
        ret_x_seq[tstep + 1, :, 5:9] = x_seq[-1,:,5:9] # covariances of the trjecotries generated by the Kalman filter
        last_observed_frame_prediction = ret_x_seq[tstep + 1, :, :2].clone()
        ret_x_seq[tstep + 1, :, :2] = x_seq[-1,:,:2] # storing the last GT observed frame here to ensure this is used in the next for loop and then 
        # storing the actual prediction in it after the forward network is run for the first step in the prediction length 


        # in prediction part we continue predictig the trajecotry of those agents that were
        # present in the last timestep of the observation period

        last_obs_frame_mask = mask[-1, :]
        rp_mask = last_obs_frame_mask.unsqueeze(dim=0).repeat(args.pred_length, 1)
        extended_mask = torch.cat((mask, rp_mask), 0)

        # For the predicted part of the trajectory
        for tstep in range(args.obs_length-1, args.pred_length + args.obs_length-1):

            # Do a forward prop
            if (args.method == 3):
                outputs, hidden_states, cell_states = net(ret_x_seq[tstep].view(1, numx_seq, ret_x_seq.shape[2]),
                                                            hidden_states, cell_states, 
                                                            last_obs_frame_mask.view(1, numx_seq))
            elif (args.method == 4):
                outputs, hidden_states, cell_states = net(ret_x_seq[tstep].view(1, numx_seq, ret_x_seq.shape[2]), [prev_grid],
                                                          hidden_states, cell_states, last_obs_frame_mask.view(
                                                              1, numx_seq),
                                                          x_seq_veh[tstep, :, :].view(
                                                              1, numx_seq_veh, x_seq_veh.shape[2]),
                                                          [prev_grid_veh_in_ped], mask_veh[tstep, :].view(
                                                              1, numx_seq_veh),
                                                          [prev_TTC_grid], [prev_TTC_grid_veh])
            if tstep == args.obs_length-1: 
                # storing the actual prediction in the last observed frame position
                ret_x_seq[args.obs_length-1, :, :2] = last_observed_frame_prediction.clone()


            # Extract the mean, std and corr of the bivariate Gaussian
            mux, muy, sx, sy, corr = getCoef(outputs.cpu())

            # Storing the paramteres of the distriution for plotting
            dist_param_seq[tstep + 1, :, :] = outputs.clone()

            # # Sample from the bivariate Gaussian
            # next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, last_obs_frame_mask)
            # # Store the predicted position
            # ret_x_seq[tstep + 1, :, 0] = next_x
            # ret_x_seq[tstep + 1, :, 1] = next_y

            # Assigning the mean to the next state instead of sampling from the distrbution.
            next_x_mean = mux.clone().data
            next_y_mean = muy.clone().data
            ret_x_seq[tstep + 1, :, 0] = next_x_mean
            ret_x_seq[tstep + 1, :, 1] = next_y_mean

            # uncomment the following three lines if the uncertainty-aware predictor relies on covaraince matrix as an input
            # scaled_param_dist = torch.stack((mux, muy, sx, sy, corr),2) 
            # cov = cov_mat_generation(scaled_param_dist)
            # ret_x_seq[tstep + 1, :, 5:9] = cov.reshape(cov.shape[0], cov.shape[1], 4).squeeze(0) # covariances of the trjectories generated by the predictor

            if args.method == 4:
                # Preparing a ret_x_seq that is covnerted back to the original frame by reverting back to the absolute coordinate.
                # This will be used for grid calculation
                ret_x_seq_convert = ret_x_seq.clone()

                ret_x_seq_convert = revert_postion_change(ret_x_seq_convert.cpu(), extended_mask,
                                                            first_values_dict, orig_x_seq, saved_args.obs_length, infer=True)

                ret_x_seq_convert[tstep + 1, :, 2] = (ret_x_seq_convert[tstep + 1, :, 0] -
                                                    ret_x_seq_convert[tstep, :, 0]) / timestamp  # vx
                ret_x_seq_convert[tstep + 1, :, 3] = (ret_x_seq_convert[tstep + 1, :, 1] -
                                                    ret_x_seq_convert[tstep, :, 1]) / timestamp  # vy
                # updating the velocity data in ret_x_seq accordingly
                ret_x_seq[tstep + 1, :, 2] = ret_x_seq_convert[tstep + 1, :, 2].clone()
                ret_x_seq[tstep + 1, :, 3] = ret_x_seq_convert[tstep + 1, :, 3].clone()

                converted_pedlist = [i for i in range(numx_seq) if last_obs_frame_mask[i] == 1]
                list_of_x_seq = Variable(torch.LongTensor(converted_pedlist))
            
                # Get their predicted positions
                current_x_seq = torch.index_select(ret_x_seq_convert[tstep+1], 0, list_of_x_seq)
      
                converted_vehlist = [i for i in range(numx_seq_veh) if mask_veh[tstep+1, i] == 1]
                list_of_x_seq_veh = Variable(torch.LongTensor(converted_vehlist))
                current_x_seq_veh = torch.index_select(x_seq_veh[tstep+1].cpu(), 0, list_of_x_seq_veh)

                prev_grid, prev_TTC_grid = getInteractionGridMask(current_x_seq.data.cpu(), current_x_seq.data.cpu(),
                                                                  saved_args.TTC, saved_args.D_min, saved_args.num_sector)
                prev_grid_veh_in_ped, prev_TTC_grid_veh = getInteractionGridMask(current_x_seq.data.cpu(),  current_x_seq_veh.data.cpu(),
                                                                                 saved_args.TTC_veh, saved_args.D_min_veh, saved_args.num_sector,
                                                                                 is_heterogeneous=True,
                                                                                 frame_ego=ego_pre_planned_traj[tstep-(args.obs_length-1)])

                prev_grid = Variable(torch.from_numpy(prev_grid).float())
                prev_grid_veh_in_ped = Variable(torch.from_numpy(prev_grid_veh_in_ped).float())
                prev_TTC_grid = Variable(torch.from_numpy(prev_TTC_grid).float())
                prev_TTC_grid_veh = Variable(torch.from_numpy(prev_TTC_grid_veh).float())

                if args.use_cuda:
                    prev_grid = prev_grid.cuda()
                    prev_grid_veh_in_ped = prev_grid_veh_in_ped.cuda()
                    prev_TTC_grid = prev_TTC_grid.cuda()
                    prev_TTC_grid_veh = prev_TTC_grid_veh.cuda()
                        
        return ret_x_seq, dist_param_seq
