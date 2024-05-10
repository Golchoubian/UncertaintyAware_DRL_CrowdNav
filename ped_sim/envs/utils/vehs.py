from ped_sim.envs.utils.agent import Agent

class Vehs(Agent):
    # see Agent class in agent.py for details!!!
    def __init__(self, config, section):
        super().__init__(config, section)

    def get_curr_ob(self, f, reset=True):
        '''
        returing a history of the agents trajectory data
        for the last obs_length period given the current frame number
        [f-obs_length:f]
        As well as the ground truth trajecotry of the vehicles (not including the robot)
        since we do not predict the postion of the other vehicles. 
        So rely on their ground truth data
        '''
        
        self.curr_obs = self.true_pose[f-self.obs_length+1:f+self.pred_length+1,:,:]
        self.curr_mask = self.mask[f-self.obs_length+1:f+self.pred_length+1,:]

        curr_ob = {'pos': self.curr_obs,
                   'mask': self.curr_mask}

        self.update_current_pose()

        return curr_ob
    

    def get_traj(self):

        return self.curr_obs, self.curr_mask
    
    def update_current_pose(self):
        '''
        updating the current pose of the vehicle
        since curr_obs of vehicle is of length obs_length + pred_length
        we need to take the last frame of the obs_length as the current pose
        '''
        self.px = self.curr_obs[self.obs_length-1, :, 0]
        self.py = self.curr_obs[self.obs_length-1, :, 1]

        self.vx = self.curr_obs[self.obs_length-1, :, 2]
        self.vy = self.curr_obs[self.obs_length-1, :, 3]

        self.theta = self.calculate_theta(self.vx, self.vy)
