from ped_sim.envs.utils.agent import Agent
import torch


class Peds(Agent):
    # see Agent class in agent.py for details!!!
    def __init__(self, config, section):
        super().__init__(config, section)

    def get_curr_ob(self, f, reset=True):
        '''
        returing a history of the agents trajectory data
        for the last obs_length period given the current frame number
        [f-obs_length:f]
        '''

        self.curr_obs = self.true_pose[f-self.obs_length+1:f+1, :, :]
        self.curr_mask = self.mask[f-self.obs_length+1:f+1, :]

        curr_ob = {'pos': self.curr_obs,
                   'mask': self.curr_mask}
    
        self.update_current_pose()

        return curr_ob


    def update_current_pose(self):
        '''
        updating the current pose of the pedestrian
        which is the pos of the last frame in the curr_obs
        '''
        self.px = self.curr_obs[-1, :, 0]
        self.py = self.curr_obs[-1, :, 1]

        self.vx = self.curr_obs[-1, :, 2]
        self.vy = self.curr_obs[-1, :, 3]

        self.theta = self.calculate_theta(self.vx, self.vy)

    def get_true_prediction(self, f):

        GT_pred_pos = self.true_pose[f+1:f+self.pred_length+1, :, :]
        GT_pred_mask = self.mask[f+1:f+self.pred_length+1, :]

        current_peds_indx = torch.nonzero(self.mask[f, :]).squeeze()#.tolist()
        if current_peds_indx.dim()==0:
            current_peds_indx = [current_peds_indx.item()]
        else:
            current_peds_indx = current_peds_indx.tolist()
        # edit the prediction data to include only the data of pedestrians that
        # are present in the current frame
        # removing the data of pedestrians that will appear in the future for
        # fair comparison of 'truth' and 'inference' cases
        GT_pred_pos_filtered = torch.zeros_like(GT_pred_pos)
        GT_pred_mask_filtered = torch.zeros_like(GT_pred_mask) 
        GT_pred_pos_filtered[:,current_peds_indx,:] = GT_pred_pos[:,current_peds_indx,:]
        GT_pred_mask_filtered[:,current_peds_indx] = GT_pred_mask[:,current_peds_indx] 
        # this ensures that peds that disapear before the end of the prediction horizon will have a mask of 0

        for i in current_peds_indx:
            disapear_indx = torch.where(GT_pred_mask_filtered[:,i]==0)[0].tolist()
            if len(disapear_indx)!=0: # the ped disapears before the end of the prediction horizon
                # store the last existing postion of that ped for the rest of the prediction horizon
                GT_pred_pos_filtered[disapear_indx,i,:] = GT_pred_pos_filtered[disapear_indx[0]-1,i,:]
                


        GT_pred = {'pos': GT_pred_pos_filtered,
                   'mask': GT_pred_mask_filtered}
        
        return GT_pred