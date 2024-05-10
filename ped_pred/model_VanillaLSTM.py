import torch
import torch.nn as nn
from torch.autograd import Variable

class VLSTMModel(nn.Module):

    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(VLSTMModel, self).__init__()

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

        # The LSTM cell 
        self.cell = nn.LSTMCell(self.embedding_size, self.rnn_size) 

        if self.gru:
            self.cell = nn.GRUCell(self.embedding_size, self.rnn_size) 


        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
       
        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

            
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
        hidden_states = args[1]
        cell_states = args[2]

        if self.gru:
            cell_states = None

        mask = args[3]


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
            nodes_current = frame[list_of_nodes,:2] # Getting only the x and y of each pedestrian for the input. Leaving th vx and vy
            if self.input_size == 6:
                # adding pedestrians covariates to the input
                nodes_current = torch.cat((nodes_current, frame[list_of_nodes,5:9]), 1) 

            # Get the corresponding hidden and cell states
            hidden_states_current = torch.index_select(hidden_states, 0, corr_index)

            if not self.gru:
                cell_states_current = torch.index_select(cell_states, 0, corr_index)

    
            # Embed inputs
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))

            if not self.gru:
                # One-step of the LSTM
                h_nodes, c_nodes = self.cell(input_embedded, (hidden_states_current, cell_states_current))
            else:
                h_nodes = self.cell(input_embedded, (hidden_states_current))


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
        for framenum in range(self.seq_length-1):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum*numNodes + node, :]

        return outputs_return, hidden_states, cell_states