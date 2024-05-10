import numpy as np
import torch
import itertools
import math
from torch.autograd import Variable


def Time2Conflict(current_position, current_velocity, other_position, other_velocity, Social_dis):
    '''
    This function calculates the time that the two agent get closer to each other than an Social_dis threshold by solving a second degree equation
    Social_dis: The distance below which we consider the case a conflict. This distance differ for a ped-ped interaction compared to ped_veh
    '''

    rel_position = other_position - current_position
    rel_velocity = other_velocity - current_velocity
    PdotV = rel_position[0]*rel_velocity[0] + rel_position[1]*rel_velocity[1]
    rel_p_norm2 = rel_position[0]**2 + rel_position[1]**2
    rel_v_norm2 = rel_velocity[0]**2 + rel_velocity[1]**2

    delta = PdotV**2 - (rel_v_norm2*(rel_p_norm2 - Social_dis**2))

    if (rel_v_norm2 == 0 or delta < 0):
        t_conflict = -1  # they do not conflict. There conflict time it -inf or +inf
    elif (rel_p_norm2**0.5 < Social_dis):
        # We are already in conflict 
        #(c in the equation ax^2+bx+c=0 gets negative and the roots are 
        # one positive and one negative, but are meaningless for us as we are
        # already in conflict)
        t_conflict = 0
    else:  # both roots (times) are either positive or negative
        t1 = (-PdotV - delta**0.5) / rel_v_norm2
        t2 = (-PdotV + delta**0.5) / rel_v_norm2
        if t2 > 0:
            t_conflict = t1  
            # both roots are positive but the smaller one should be considered. The bigger one is the time we get out of the conflict
        else:
            # both roots are negative so the agents never get into conflict ( were is conflict preveiously at a time that has now passed)
            t_conflict = -1

    return t_conflict


# def approach_dir_sector(current_position, current_velocity, other_position, num_sector):
#     '''
#     Considering that the circle around each ego agent is divided equally into n sectors,
#     this function calculates the number of the sector the other agent stands in. 
#     The numbering of sectors is counter clockwise, starting from the diretion of the the
#     ego agent's current heading specified through its velocity vector
#     current_position: the position vector of the ego agent (x,y)
#     current_velocity: the velocity vector of the ego agent (Vx,Vy)
#     other_position: the position vector of the neighbour agent (x,y)
#     other_velocity: the velocity vector of the neighbour agent (Vx,Vy)
#     num_sector: the number of secotrs the circle is divived to
#     '''

#     rel_position = other_position - current_position

#     # calculating the angle (theta) between the current agent's velocity vector and 
#     # the relative position to the neigbour agent (calculating using atan2)

#     # dot is propotional to cos(theta)
#     dot = current_velocity[0]*rel_position[0] + current_velocity[1]*rel_position[1] 
#     # det (|V P|) is propotional to sin(theta)
#     det = current_velocity[0]*rel_position[1] - rel_position[0] * \
#         current_velocity[1]
#     angle = math.atan2(det, dot) * (180/math.pi)  # the value is between -180 and 180

#     if angle < 0:
#         angle = angle + 360

#     approach_cell = int(angle / (360/num_sector))

#     return approach_cell


def approach_angle_sector(current_velocity, other_velocity, num_sector):
    '''
    Considering that the apprach angle between the conflicting agent can be divided equally into n sectors, this function calculates to which discretized
    sector the approach angle belongs. The approach angle is calculated with respect to the ego agent's velocity vector in counterclockwise direction.
    The angles are claculated between 0 and 360 deg and the numbering of the sectros starts with the interval that has the lowest angles
    current_velocity: the velocity vector of the ego agent (Vx,Vy)
    other_velocity: the velocity vector of the neighbour agent (Vx,Vy)
    num_sector: the number of secotrs the circle is divived to
    '''

    # calculating the angle (theta) between the current agent's velocity vector and
    #  the neighbor agent's velocoty vector (calculating using atan2)

    # dot is propotional to cos(theta)
    dot = current_velocity[0]*other_velocity[0] + current_velocity[1] * \
        other_velocity[1] 
    # det (|V_e V_n|) is propotional to sin(theta)
    det = current_velocity[0]*other_velocity[1] - other_velocity[0] * \
        current_velocity[1]
    angle = math.atan2(det, dot) * (180/math.pi)  # the value is between -180 and 180

    if angle < 0:
        angle = angle + 360

    approach_cell = int(angle / (360/num_sector))
    if angle == 360:  # an edge case. To ensure that we do not get out of bound in matrix dimension later on
        approach_cell = num_sector - 1

    return approach_cell


def getInteractionGridMask(frame, frame_other, TTC_min, d_min, num_sector, is_heterogeneous=False, is_occupancy=False, frame_ego=None):
    '''
    This function computes the binary mask that represents the
    occupancy of each ped in the other's grid
    params:
    frame : This will be a num_typeI x 3 matrix with each row being [x, y, vx, vy]
    TTC_min: Is the list of number of TTC thresholds that is considered. Form smaler to bigger time
    num_sector: Is the number of equal sectors the circle around each aget is divided to for approach angle consideration
    is_heterogeneous: A flag used specifying wether the inetractions between ped-ped is being considered or ped-veh
    is_occupancy: A flag using for calculation of accupancy map

    '''

    num_agent = frame.shape[0]
    num_agent_other = frame_other.shape[0]
    num_TTC = len(TTC_min)

    if frame_ego is not None:
        num_agent_other += 1  # adding the ego to the neiboring agents count
        frame_ego_np = frame_ego.data.numpy()

    if is_occupancy:
        frame_mask = np.zeros((num_agent, num_TTC*num_sector))
    else:
        # last neiboring agent in the grid tensor creation is the ego vehicle
        frame_mask = np.zeros((num_agent, num_agent_other, num_TTC*num_sector))
        TTC_mask = np.zeros((num_agent, num_agent_other, num_TTC*num_sector))

    frame_np = frame.data.numpy()
    frame_other_np = frame_other.data.numpy()

    # instead of 2 inner loop, we check all possible 2-permutations which is 2 times faster.
    list_indices = list(range(0, num_agent))
    list_indices_other = list(range(0, num_agent_other))

    for real_frame_index, other_real_frame_index in itertools.product(list_indices, list_indices_other):

        if (is_heterogeneous == False and real_frame_index == other_real_frame_index):
            # In case of homogeneous agents as the input for frame and frame_other is
            # the same then we want to skip considering one agent with itself
            continue

        current_position = frame_np[real_frame_index, 0:2]
        current_velocity = frame_np[real_frame_index, 2:4]

        # dealing with the ego vehicle
        if ((frame_ego is not None) and (other_real_frame_index == frame_other.shape[0])):
            other_position = frame_ego_np[0, 0:2]
            other_velocity = frame_ego_np[0, 2:4]
        else:
            other_position = frame_other_np[other_real_frame_index, 0:2]
            other_velocity = frame_other_np[other_real_frame_index, 2:4]

        t_conflict = Time2Conflict(current_position, current_velocity, other_position, other_velocity, d_min)

        if t_conflict == -1:  # not conlficting
            continue

        TTC_min.sort()
        time_cell = -1
        for i, t_threshold in enumerate(TTC_min):
            # The output of Time2Conflict() if not -1 should be >=0. But we check this again anyways
            if ((t_conflict >= 0) and (t_conflict <= t_threshold)):
                time_cell = i
                break  # finding the first time_threshold that passes the condition

        if time_cell == -1:  # the t_conflcit is not critical at this step
            continue

        approach_cell = approach_angle_sector(current_velocity, other_velocity, num_sector)

        if is_occupancy:
            frame_mask[real_frame_index, time_cell*num_sector+approach_cell] = 1
        else:
            frame_mask[real_frame_index, other_real_frame_index, time_cell*num_sector+approach_cell] = 1
            # The ones already within d_min distance from the ego (t_conflict=0) will get the highest number (attenstion) in the tensor
            TTC_mask[real_frame_index, other_real_frame_index, time_cell *
                     num_sector+approach_cell] = (TTC_min[0] - t_conflict)

    return frame_mask, TTC_mask


def getSequenceInteractionGridMask(sequence, avail_mask, sequence_veh, avail_mask_veh,
                                   TTC_min, d_min, num_sector, using_cuda,
                                   is_heterogeneous=False, is_occupancy=False, sequence_ego=None):
    '''
    Get the grid masks for all the frames in the sequence
    params:
    sequence : A torch tensor of shape SL x MNP x 5
    pedlist_seq: A list of list containing agent IDs presnet at each frame   
    sequence_veh:  A numpy array of shape SL x MNP x 5 for neigboring agents whether pedestrians or vehicles
    vehlist_seq: A list of list containing the negiboring agents IDs at each frame
    lookup_seq: A dictionary 
    TTC_min : Scalar value representing the TTC threshold below which an interaction will be considered
    d_min : Scalar value indicating the distance threshold between two agents below which a conflict occurs  
    using_cuda: Boolean value denoting if using GPU or not
    is_occupancy: A flag using for calculation of accupancy map
    is_heterogeneous: A flag for indicating the interaction type being ped-ped or ped-veh
    '''

    sl = len(sequence)
    sequence_mask = []
    sequence_mask_TTC = []

    for i in range(sl):

        converted_pedlist = []
        for ind in range(len(avail_mask[i])):
            if avail_mask[i][ind] == 1:  # if the agent is present in this frame
                converted_pedlist.append(ind)
        list_of_x_seq = Variable(torch.LongTensor(converted_pedlist))
        # Get the portion of the sequence[i] that has only pedestrians present in this specific frame
        current_x_seq = torch.index_select(sequence[i], 0, list_of_x_seq)

        converted_vehlist = []
        for ind in range(len(avail_mask_veh[i])):
            if avail_mask_veh[i][ind] == 1:
                converted_vehlist.append(ind)
        list_of_x_seq_veh = Variable(torch.LongTensor(converted_vehlist))
        current_x_seq_veh = torch.index_select(sequence_veh[i], 0, list_of_x_seq_veh)

        if sequence_ego is not None:
            current_x_seq_ego = sequence_ego[i]
        else:
            current_x_seq_ego = None

        mask_np, mask_TTC_np = getInteractionGridMask(
            current_x_seq, current_x_seq_veh, TTC_min, d_min, num_sector, is_heterogeneous, is_occupancy, current_x_seq_ego)
        mask = Variable(torch.from_numpy(mask_np).float())
        mask_TTC = Variable(torch.from_numpy(mask_TTC_np).float())

        if using_cuda:
            mask = mask.cuda()
            mask_TTC = mask_TTC.cuda()
        sequence_mask.append(mask)
        sequence_mask_TTC.append(mask_TTC)

    return sequence_mask, sequence_mask_TTC
