import torch
import torch.nn as nn
from torch.autograd import Variable

class CollisionGridModel(nn.Module):

    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(CollisionGridModel, self).__init__()

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda

        if infer:
            # Test time
            self.seq_length = 2
        else:
            # Training time
            if args.teacher_forcing:
                self.seq_length = args.seq_length
            else:
                self.seq_length = 2

        # Store required sizes
        self.rnn_size = args.rnn_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.maxNumPeds=args.maxNumPeds
        self.gru = args.gru

        self.num_sector = args.num_sector
        self.num_TTC = len(args.TTC)
        self.num_TTC_veh = len(args.TTC_veh)
        self.embedding_size_action = args.embedding_size_action

        # The LSTM cell for pedestrians
        self.cell = nn.LSTMCell(3*self.embedding_size, self.rnn_size)
 
        if self.gru:
            self.cell = nn.GRUCell(3*self.embedding_size, self.rnn_size)


        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        
        # Linear layer to embed the social tensor
        self.tensor_embedding_layer = nn.Linear(self.num_TTC*self.num_sector, self.embedding_size) # uncomment this (new - ped)
        
        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        self.tensor_embedding_layer_veh = nn.Linear(self.num_TTC_veh*self.num_sector, self.embedding_size)

      
    def getSocialTensor(self, grid, hidden_states, veh_tensor = False):
        '''
        Computes the social tensor for a given grid mask and hidden states of all peds
        params:
        grid : Grid masks
        hidden_states : Hidden states of all peds / OR the embedded state of vehicles
        '''
        # Number of peds
        numNodes = grid.size()[0]

        # Construct the variable
        if (veh_tensor == True):
            social_tensor = Variable(torch.zeros(numNodes, self.num_TTC*self.num_sector, self.embedding_size))
        else: # for ped in ped grid.
            social_tensor = Variable(torch.zeros(numNodes, self.num_TTC*self.num_sector, self.rnn_size))
        if self.use_cuda:
            social_tensor = social_tensor.cuda()
        
        # For each ped
        for node in range(numNodes):
            # Compute the social tensor
            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)

        # Reshape the social tensor
        if (veh_tensor == True):
            social_tensor = social_tensor.view(numNodes, self.num_TTC*self.num_sector*self.embedding_size)
        else:
            social_tensor = social_tensor.view(numNodes, self.num_TTC*self.num_sector*self.rnn_size)
        return social_tensor

    def getSocialTensor2(self, grid, hidden_states, grid_TTC):
        '''
        Computes the social tensor for a given grid mask and hidden states of all peds
        params:
        grid : Grid masks
        hidden_states : Hidden states of all peds / OR the embedded state of vehicles
        '''
        # Number of peds
        numNodes = grid.size()[0]
        third_dim = hidden_states.shape[1]

        # Construct the variable
        social_tensor = Variable(torch.zeros(numNodes, self.num_TTC*self.num_sector, third_dim))
        if self.use_cuda:
            social_tensor = social_tensor.cuda()
        
        # For each ped
        for node in range(numNodes):
            # Compute the social tensor
            
            ## With the restriction of having only one agent in each occupancy grid cell,
            # we keep the more critical one, defined as having the lowest TTC
            grid_updated = self.OnePerGridCell(grid[node], grid_TTC[node])
           
            social_tensor[node] = torch.mm(torch.t(grid_updated), hidden_states)

        # Reshape the social tensor
        social_tensor = social_tensor.view(numNodes, self.num_TTC*self.num_sector*third_dim)

        return social_tensor
    

    def OnePerGridCell(self, grid_current, grid_TTC_current):

        grid_current_updated =  torch.zeros_like(grid_current)

        # Specifying the columns (grid cells) that have occupancy in first place
        valid_col_index = torch.nonzero(torch.sum(grid_current, dim=0))

        for col in valid_col_index:
            col_ind = col.item()
            # getting the row indexes of non-zero rows (indexs of neigboring agents lying in this specific grid cell) 
            # if they are more than one agent, get the index of the one with minimum TTC (the most critical one)
            rows = torch.nonzero(grid_current[:,col_ind])
            ttc = grid_TTC_current[rows,col_ind] # a smaller tensor or only ttc of conflicting agents presnet in that cell
            min_index_in_rows = torch.argmin(ttc)
            row_ind = rows[min_index_in_rows]
            grid_current_updated[row_ind, col_ind] = 1
        
        return grid_current_updated
          

    def forward(self, *args):

        '''
        Forward pass for the model
        params:
        input_data: Input positions
        grids: Grid masks
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds
        PedsList: id of peds in each frame for this sequence

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''

        # Construct the output variable
        input_data = args[0]
        grids = args[1]
        hidden_states = args[2]
        cell_states = args[3]

        if self.gru:
            cell_states = None

        mask = args[4]
        input_data_veh = args[5]
        grids_veh = args[6]
        mask_veh = args[7]
        grids_TTC = args[8]
        grids_TTC_veh = args[9]

        numNodes = input_data.shape[1]
        outputs = Variable(torch.zeros((self.seq_length-1) * numNodes, self.output_size))
        if self.use_cuda:            
            outputs = outputs.cuda()

        # For each frame in the sequence
        for framenum,frame in enumerate(input_data):

            # Peds present in the current frame
            curr_mask = mask[framenum,:]
            if torch.sum(curr_mask) == 0:
                # If no peds, then go to the next frame
                continue


            # List of nodes
            list_of_nodes = []
            for i in range(curr_mask.shape[0]):
                if curr_mask[i] == 1:
                    list_of_nodes.append(i)

            corr_index = Variable((torch.LongTensor(list_of_nodes)))
            if self.use_cuda:            
                corr_index = corr_index.cuda()

            # Select the corresponding input positions
            nodes_current = frame[list_of_nodes,:2] # Getting only the x and y of each pedestrian for the input.
            if self.args.input_size == 6: # the model also gets covariance as input
                # adding pedestrians covariates to the input
                nodes_current = torch.cat((nodes_current, frame[list_of_nodes,5:9]), 1) 
           
            # Get the corresponding grid masks
            grid_current = grids[framenum] 
            grid_current_veh = grids_veh[framenum] 
            grid_TTC_current = grids_TTC[framenum]
            grid_TTC_current_veh = grids_TTC_veh[framenum]

            if self.use_cuda:
                grid_current = grid_current.cuda()
                grid_current_veh = grid_current_veh.cuda()
                grid_TTC_current = grid_TTC_current.cuda()
                grid_TTC_current_veh = grid_TTC_current_veh.cuda()


            # Get the corresponding hidden and cell states
            hidden_states_current = torch.index_select(hidden_states, 0, corr_index)


            if not self.gru:
                cell_states_current = torch.index_select(cell_states, 0, corr_index)

            # getting the max value of each grid cell for each ego agent (max since we are using (TTC_threshod - TTC) instead of the pure TTC.
            # So bigger values are more prone to collide)
            social_tensor = grid_TTC_current.max(1)[0] # this should become of size: num_agent * num_sector

            # Embed inputs
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))
            # Embed the social tensor
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))

            if (grid_TTC_current_veh.shape[1] == 0): # meaning we do not have a vehicle in this sequence
                social_tensor_veh = torch.zeros((grid_TTC_current_veh.shape[0],grid_TTC_current_veh.shape[2]))
                if self.use_cuda:
                    social_tensor_veh = social_tensor_veh.cuda()
            else:
                social_tensor_veh = grid_TTC_current_veh.max(1)[0] # same as for pedestrians
            
            tensor_embedded_veh = self.dropout(self.relu(self.tensor_embedding_layer_veh(social_tensor_veh)))

            # Concat input
            concat_embedded = torch.cat((input_embedded, tensor_embedded,tensor_embedded_veh), 1)


            if not self.gru:
                # One-step of the LSTM
                h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))
            else:
                h_nodes = self.cell(concat_embedded, (hidden_states_current))

            # Compute the output
            outputs[framenum*numNodes + corr_index.data] = self.output_layer(h_nodes)

            # Update hidden and cell states
            hidden_states[corr_index.data] = h_nodes
            if not self.gru:
                cell_states[corr_index.data] = c_nodes

        # Reshape outputs
        outputs_return = Variable(torch.zeros((self.seq_length-1), numNodes, self.output_size))
        if self.use_cuda:
            outputs_return = outputs_return.cuda()
        for framenum in range(self.seq_length-1): # !!! also -1 here
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum*numNodes + node, :]

        return outputs_return, hidden_states, cell_states