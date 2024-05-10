
import os
import pandas as pd
import numpy as np
import pickle
from torch.autograd import Variable
import torch
import random



class DataLoader():

    def __init__(self, phase='train'):


        self.downsample_step = 1 # consider the data every x steps
        fps = 2 / self.downsample_step # (CITR: 29.97, DUT:23.98, ETH/UCY: 2.5, HBS: 2, SDD: 30)
        self.timestamp = 1 / fps
        self.phase = phase

        base_dataset = ['ped_pred/Data/HBS/hbs.csv']

        self.base_data_dirs = base_dataset

        # Number of datasets
        self.numDatasets = len(self.base_data_dirs)

        # array for keepinng target ped ids for each sequence
        self.target_ids = []
        self.veh_ids = []

        # Define the path in which the process data would be stored
        self.data_file_train = os.path.join('ped_pred/Data/preprocessed/', "trajectories_train.cpkl")
        self.data_file_test = os.path.join('ped_pred/Data/preprocessed/', "trajectories_test.cpkl")
        self.data_file_vali = os.path.join('ped_pred/Data/preprocessed/', "trajectories_validation.cpkl")

        print('==================== phase is:', self.phase)
        if self.phase == 'train':
            self.data_file = self.data_file_train
        elif self.phase == 'test':
            self.data_file = self.data_file_test
        elif self.phase == "val":
            self.data_file = self.data_file_vali
        else: 
            raise Exception('the phase input for dataloder should be either "train", "test" or "val"')


        if not(os.path.exists(self.data_file)): # all the files will be created at the first run.

            print("Creating pre-processed data from raw data")
            self.frame_preprocess(self.base_data_dirs, self.data_file, self.phase)
            self.load_preprocessed(self.data_file)
        else:
            self.load_preprocessed(self.data_file)



    def frame_preprocess(self, data_dirs, data_file, phase):
        '''
        Function that will pre-process the csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file_tr : The file into which all the pre-processed training data needs to be stored
        data_file_te : The file into which all the pre-processed testing data needs to be stored

        '''
        # all_frame_data would be a list of list of list of numpy arrays corresponding to each dataset and each target vehicle in that dataset
        # Each numpy array will correspond to a frame and would be of size (numPeds, 6)
        # containing pedID, x, y, vx, vy, timstamp
        all_frame_data_ped = []
       
        # frameList_data would be a list of list of lists corresponding to each dataset and each target vehicle in that dataset
        # Each list would contain the frameIds of all the frames in the dataset where the target agent is present
        frameList_data = []
       
        # numPeds_data would be a list of list of lists corresponding to each dataset and each target vehicle in that dataset
        # Each list would contain the number of pedestrians in each frame in the dataset where the target vehicle is present
        numPeds_data = []
        
        #each list includes ped ids of this frame
        pedsList_data = []

        # target ped ids for each sequence
        target_ids = []
        orig_data = []

        # creating the same lists for the vehicle data
        all_frame_data_veh = []
        numVehs_data = []
        vehsList_data = []
        target_ids_veh = []
        orig_data_veh = []

        # all_frame_data_ego is a list of list of list of numpy arrays corresponding to each dataset and each target vehicle in that dataset
        # each numpy array will correspond to a frame and would be of size (1, 6)
        # containing tagetvehID, x, y, vx, vy, timstamp
        all_frame_data_ego = []

        egoList_data = []
        scenario_num_list = []


        scenario_index = 0
        # For each dataset (each video file of each dataset)
        for directory in data_dirs:

            # Load the data from the txt file
            print("Now processing: ", directory)
            
            column_names = ['frame_id','agent_id','pos_x','pos_y', 'vel_x', 'vel_y', 'label', 'timestamp']

            # read train/test/validation file to pandas dataframe and process
            df_orig = pd.read_csv(directory, dtype={'frame_id':'int','agent_id':'int', 'label':'str'}, usecols=column_names)
            df_orig = df_orig[column_names] # changing the order of the columns as specifed inn the "columns_names"

            # downsampling the data
            min_frame = df_orig["frame_id"].min()
            max_frame = df_orig["frame_id"].max()
            considered_frames = np.arange(min_frame, max_frame, self.downsample_step).tolist()
            df = df_orig.loc[df_orig['frame_id'].isin(considered_frames)]


            df_ped_orig = df.loc[(df['label'].isin(['pedestrian', 'ped']))] # !! Check these lables for each new dataset you want to add later
            df_veh_orig = df.loc[(df['label'].isin(['car', 'cart', 'veh', 'vehicle']))]  # !!! Check these lables for each new dataset you want to add later
            self.target_ped_ids = np.array(df_ped_orig.drop_duplicates(subset={'agent_id'}, keep='first', inplace=False)['agent_id'])
            self.target_veh_ids = np.array(df_veh_orig.drop_duplicates(subset={'agent_id'}, keep='first', inplace=False)['agent_id'])


            df_ped = df_ped_orig.copy()
            df_veh = df_veh_orig.copy()


            # convert pandas -> numpy array
            data = np.array(df_ped.drop(['label'], axis=1))
            data_veh = np.array(df_veh.drop(['label'], axis=1))


            # keep original copy of file
            orig_data.append(data)
            orig_data_veh.append(data_veh)

            #swap x and y points (in txt file it is like -> y,x)
            data = np.swapaxes(data,0,1) # Now the frame id will propogate through the columns and the frames are all in the first row
            data_veh = np.swapaxes(data_veh,0,1)

            for veh_id in  self.target_veh_ids:

                numPeds_data.append([])
                all_frame_data_ped.append([])
                pedsList_data.append([])
                numVehs_data.append([])
                all_frame_data_veh.append([])
                vehsList_data.append([])
                all_frame_data_ego.append([])
                egoList_data.append([])

                # get frame numbers
                curr_veh_data = data_veh[:, data_veh[1,:] == veh_id] 
                frameList = curr_veh_data[0, :].tolist()
                frameList.sort() # sorting the frame numbers
                # Number of frames
                numFrames = len(frameList) 

                # Add the list of frameIDs for this vehicle to the frameList_data
                frameList_data.append(frameList) 

                for ind, frame in enumerate(frameList):

                    # Extract all pedestrians in current frame
                    pedsInFrame = data[: , data[0, :] == frame]
                    allvehsInFrame = data_veh[: , data_veh[0, :] == frame]
                    egoInFrame = allvehsInFrame[:, allvehsInFrame[1,:] == veh_id]
                    vehsInFrame = allvehsInFrame[:, allvehsInFrame[1,:] != veh_id]

                    # Extract peds list
                    pedsList = pedsInFrame[1, :].tolist() # Grabing the agent_id of all the pedestrians in this specific frame
                    vehsList = vehsInFrame[1, :].tolist()

                    # Initialize the row of the numpy array
                    pedsWithPos = []
                    vehsWithPos = []

                    egoWithPos = np.transpose(egoInFrame[1:,:]) 

                    # For each ped in the current frame
                    for ped in pedsList:
                        # Extract their x and y positions
                        current_x = pedsInFrame[2, pedsInFrame[1, :] == ped][0]
                        current_y = pedsInFrame[3, pedsInFrame[1, :] == ped][0]
                        current_vx = pedsInFrame[4, pedsInFrame[1, :] == ped][0]
                        current_vy = pedsInFrame[5, pedsInFrame[1, :] == ped][0]
                        timestamp = pedsInFrame[6, pedsInFrame[1, :] == ped][0]

                        # Add their pedID, x, y to the row of the numpy array
                        pedsWithPos.append([ped, current_x, current_y, current_vx, current_vy, timestamp])

                    # For each veh in the current frame
                    for veh in vehsList:
                        # Extract their x and y positions
                        current_x_veh = vehsInFrame[2, vehsInFrame[1, :] == veh][0]
                        current_y_veh = vehsInFrame[3, vehsInFrame[1, :] == veh][0]
                        current_vx_veh = vehsInFrame[4, vehsInFrame[1, :] == veh][0]
                        current_vy_veh = vehsInFrame[5, vehsInFrame[1, :] == veh][0]
                        timestamp_veh = vehsInFrame[6, vehsInFrame[1, :] == veh][0]

                        # Add their vehID, x, y to the row of the numpy array
                        vehsWithPos.append([veh, current_x_veh, current_y_veh, current_vx_veh, current_vy_veh, timestamp_veh])

                  
                    all_frame_data_ego[scenario_index].append(egoWithPos)
                    
                    all_frame_data_ped[scenario_index].append(np.array(pedsWithPos))
                    pedsList_data[scenario_index].append(pedsList)
                    numPeds_data[scenario_index].append(len(pedsList))

                    all_frame_data_veh[scenario_index].append(np.array(vehsWithPos))
                    vehsList_data[scenario_index].append(vehsList)
                    numVehs_data[scenario_index].append(len(vehsList))

                    egoList_data[scenario_index].append([veh_id])

                scenario_num_list.append(scenario_index)
                scenario_index += 1

        
        # =========== creating the data split for train/test/validation =========
        #  
        num_scenarios = len(all_frame_data_ego)
        print(f"Total number of scenarios: {num_scenarios}")
        # removing the last 30 scenarios of the HBS dataset, since they aren't complete trajectories
        num_scenarios = num_scenarios -20
        print(f"Total number of scenarios after filtring some of them: {num_scenarios}")

        # 80% of the data goes for training. The rest for test
        num_train_data = int(num_scenarios * 0.8)
        num_validation = int(num_train_data * 0.2)

        data_split = {'validation': (0,num_validation),
                      'train': (num_validation,num_train_data),
                      'test': (num_train_data, num_scenarios)}
        print("The data split is as follows:")
        print(data_split)
        start, end = data_split[phase][0], data_split[phase][1]

        invalid_ind = [23, 145, 193, 194, 220, 250, 251, 272, 273, 309]
        # remove these scenarios due to overlapping agents
        
        data_ind = list(range(start, end))
        valid_data_ind = sorted(list(set(data_ind) - set(invalid_ind)))


        all_frame_data_ego_phase = [all_frame_data_ego[i] for i in valid_data_ind]
        egoList_data_phase = [egoList_data[i] for i in valid_data_ind]
        all_frame_data_ped_phase = [all_frame_data_ped[i] for i in valid_data_ind]
        frameList_data_phase = [frameList_data[i] for i in valid_data_ind]
        numPeds_data_phase = [numPeds_data[i] for i in valid_data_ind]
        pedsList_data_phase = [pedsList_data[i] for i in valid_data_ind]
        target_ids_phase = target_ids
        orig_data_phase = orig_data
        all_frame_data_veh_phase = [all_frame_data_veh[i] for i in valid_data_ind] 
        numVehs_data_phase = [numVehs_data[i] for i in valid_data_ind]
        vehsList_data_phase = [vehsList_data[i] for i in valid_data_ind] 
        target_ids_veh_phase = target_ids_veh
        orig_data_veh_phase = orig_data_veh
        scenario_num_list_phase = [scenario_num_list[i] for i in valid_data_ind]

        # Save the arrays in the pickle file
        f = open(data_file, "wb")
        pickle.dump((all_frame_data_ego_phase, egoList_data_phase,
                      all_frame_data_ped_phase, frameList_data_phase, numPeds_data_phase,
                      pedsList_data_phase, target_ids_phase, orig_data_phase,
                      all_frame_data_veh_phase, numVehs_data_phase, vehsList_data_phase,
                      target_ids_veh_phase, orig_data_veh_phase, scenario_num_list_phase),
                      f, protocol=2)
        f.close()


    def load_preprocessed(self, data_file): 
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file containig all the trajecotries
        '''
       
        print("Loading all the dataset: ", data_file)

        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()

        # Get all the data from the pickle file
        self.egoData = raw_data[0] # all_frame_data_ego  from the frame_preprocess function
        self.egoList = raw_data[1]
        self.data_ped = raw_data[2] # all_frame_data  from the frame_preprocess function
        self.frameList = raw_data[3] # frameList_data  from the frame_preprocess function
        self.numPedsList = raw_data[4] # numPeds_data  from the frame_preprocess function
        self.pedsList = raw_data[5] # pedsList_data  from the frame_preprocess function
        self.target_ids = raw_data[6] # target_ids  from the frame_preprocess function
        self.orig_data = raw_data[7] # orig_data  from the frame_preprocess function
        
        self.data_veh = raw_data[8]
        self.numVehsList = raw_data[9]
        self.vehsList = raw_data[10]
        self.target_ids_veh = raw_data[11]
        self.orig_data_veh = raw_data[12]
        self.scenario_num_list = raw_data[13]

        # Calculate the overall number of scenarios
        self.num_scenarios = len(self.egoData)
        print('Total number of scenarios:', self.num_scenarios)

        self.num_features = self.egoData[0][0].shape[1] - 1 #  -1 to not consider agent id in the count 

    
    def get_scenario(self, index, shuffle=False, first_time=False, test_case = None):

        if shuffle:
            all_data = list(zip(self.egoData, self.egoList, self.data_ped, self.pedsList,
                                 self.data_veh, self.vehsList, self.scenario_num_list))
            random.shuffle(all_data)
            self.egoData2, self.egoList2, self.data_ped2, self.pedsList2,  \
                self.data_veh2, self.vehsList2, self.scenario_num2 = zip(*all_data)
        if first_time:
            self.egoData2, self.egoList2 = self.egoData.copy(), self.egoList.copy()
            self.data_ped2, self.pedsList2 = self.data_ped.copy(), self.pedsList.copy()
            self.data_veh2, self.vehsList2 = self.data_veh.copy(), self.vehsList.copy()
            self.scenario_num2 = self.scenario_num_list.copy()
  
        egoData, egoList = self.egoData2, self.egoList2
        data_ped, pedsList = self.data_ped2, self.pedsList2
        data_veh, vehsList = self.data_veh2, self.vehsList2
        scenario_num = self.scenario_num2

        if test_case is not None:
            index = scenario_num.index(test_case)

        out = (egoData[index], egoList[index], data_ped[index], pedsList[index], \
                data_veh[index], vehsList[index], scenario_num[index])

        return out


    def convert_proper_array(self, x_seq, pedlist, seq_len):
        '''
        converter function to appropriate format. 
        Instead of direcly using ped ids, we are mapping ped ids to
        array indices using a lookup table for each scenario
        return_mask is a True-False tensor of size (seq_len, num_peds) and
        specifys whether that ped and at that time step is present or not
        This mask will be used instead of the pedList and lookup table
        '''
        # get unique ids from sequence
        unique_ids = pd.unique(np.concatenate(pedlist).ravel().tolist()).astype(int)
        # create a lookup table which maps ped ids -> array indices
        lookup_table = dict(zip(unique_ids, range(0, len(unique_ids))))  

        seq_data = np.zeros(shape=(seq_len, len(lookup_table), self.num_features))
        availability_mask = np.zeros(shape=(seq_len, len(lookup_table)),dtype=np.int8)

        # create new structure of array
        for ind, frame in enumerate(x_seq):
            if len(pedlist[ind]) != 0:
                corr_index = [lookup_table[x] for x in frame[:, 0]] 
                availability_mask[ind, corr_index] = 1
                seq_data[ind, corr_index,:] = frame[:,1:] 

        return_arr = Variable(torch.from_numpy(np.array(seq_data)).float())
        return_mask = Variable(torch.from_numpy(np.array(availability_mask)))

        return return_arr, return_mask

    
    
    def get_len_of_dataset(self):
        # return the number of dataset in the mode
        return len(self.egoData)


    def extend_traj(self, pos_veh, mask_veh, ext_len):

        '''
        Given the data of the last frame of the scenario in the dataset
        we want to extend the trajecotries of the vehicles (beyond existing data in dataset)
        usinga constant velocity model
        This is done only for vehicles since we do not predict their trajectories to rely on that
        '''

        num_veh = mask_veh.shape[1]
        ext_pos = np.zeros(shape=(ext_len, num_veh, self.num_features))
        last_pres_vehs_ind = [i for i in range(num_veh) if mask_veh[-1][i]==1]

        velocity_vec = pos_veh[-1, last_pres_vehs_ind, 2:4]
        last_pos = pos_veh[-1, last_pres_vehs_ind, 0:2]
        for t in range(ext_len):
            next_pos = last_pos + self.timestamp*velocity_vec
            ext_pos[t, last_pres_vehs_ind,0:2] = next_pos
            ext_pos[t, last_pres_vehs_ind,2:4] = velocity_vec
            ext_pos[t, last_pres_vehs_ind,4] = self.timestamp * np.ones(len(last_pres_vehs_ind))
            last_pos = next_pos

        last_mask = mask_veh[-1,:]
        rp_mask = last_mask.unsqueeze(dim=0).repeat(ext_len,1)
        ext_veh_mask = torch.cat((mask_veh, rp_mask), 0)

        ext_pos_return = Variable(torch.from_numpy(np.array(ext_pos)).float())
        ext_pos_return = torch.cat((pos_veh,ext_pos_return), 0)

        return ext_pos_return, ext_veh_mask
    
    
    def extend_traj2(self, pos_veh, VehsList, lookup_veh, ext_len):

        '''
         same as "extend_traj" just written in vectorized form to improve computational cost
        '''

        ext_pos = torch.zeros(ext_len, len(lookup_veh), self.num_features)
        last_pres_vehs_ID =VehsList[-1] 
        last_pres_vehs_ind = [lookup_veh[ID] for ID in last_pres_vehs_ID]

        last_pos = pos_veh[-1, last_pres_vehs_ind, 0:2]
        last_pos_rp = last_pos.unsqueeze(dim=0).repeat(ext_len,1,1)
        velocity_vec = pos_veh[-1, last_pres_vehs_ind, 2:4]
        velocity_vec_rp = velocity_vec.unsqueeze(dim=0).repeat(ext_len,1,1)
        timestep = torch.FloatTensor([*range(1,ext_len+1)])
        timestep_rp = timestep.unsqueeze(dim=1).unsqueeze(2).repeat(1, len(last_pres_vehs_ID), 2)
        new_pos = torch.mul(timestep_rp,velocity_vec_rp) + last_pos_rp
        ext_pos[:,last_pres_vehs_ind,0:2] = new_pos
        ext_pos[:,last_pres_vehs_ind,2:4] = velocity_vec_rp
        ext_pos[:,last_pres_vehs_ind,4] = self.timestamp *torch.ones(ext_len, len(last_pres_vehs_ID))

        ext_pos_return = torch.cat((pos_veh, ext_pos),0)
        ext_VehsList = VehsList + [last_pres_vehs_ID for i in range(ext_len)]

        return ext_pos_return, ext_VehsList