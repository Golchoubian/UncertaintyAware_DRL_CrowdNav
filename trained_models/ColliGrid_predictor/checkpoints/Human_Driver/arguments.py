import argparse
import torch


def get_args():

    parser = argparse.ArgumentParser()

    # the saving directory for train.py
    parser.add_argument(
        '--output_dir', type=str, default='trained_models/my_model')

    # resume training from an existing checkpoint or not
    parser.add_argument(
        '--resume', default=False, action='store_true')
    # if resume = True, load from the following checkpoint
    parser.add_argument(
        '--load-path', default='trained_models/my_model/checkpoints/75200.pt',
        help='path of weights for resume training')

    parser.add_argument(
        '--overwrite',
        default=True,
        action='store_true',
        help="whether to overwrite the output directory in training")

    parser.add_argument(
        '--num_threads',
        type=int,
        default=1,
        help="number of threads used for intraop parallelism on CPU")

    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")

    # only works for gpu only (although you can make it work on cpu after some minor fixes)
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')

    parser.add_argument(
        '--seed', type=int, default=425, help='random seed (default: 1)')

    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training processes to use (default: 16)')  
        # Number of parallel environments for collecting robot experience

    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=2,
        help='number of batches for ppo (default: 32)')

    parser.add_argument(
        '--num-steps',
        type=int,
        default=30,
        help='number of forward steps in A2C (default: 5)')

    # Mahsa: PPO parameters
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=5,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.0,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--lr', type=float, default=4e-5, help='learning rate (default: 4e-5)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')

    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')

    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=40e6, 
        help='number of environment steps to train (default: 40e6)')

    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=True,
        help='use a linear schedule on the learning rate')
    
    parser.add_argument(
        '--lr-decay-start-epoch',
        type=int,
        default=5000,
        help='starting the learning rate atfer this amount of update')

    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')

    # Sequence length
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Sequence length')  # same as algo_args.num_steps

    # ===========================================================
    #                   params of prediction model
    # ===========================================================
    parser.add_argument('--pred_length', type=int, default=6,
                        help='prediction length')
    parser.add_argument('--obs_length', type=int, default=6,
                        help='Observed length of the trajectory')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size of the dataset. Its 1 for inference')
    # ===========================================================
    # ===========================================================

    # "PedSimPred-v0" when using prediction
    # "PedSim-v0" when not using prediction
    parser.add_argument(
        '--env-name',
        default='PedSim-v0',
        help='name of the environment')
     
    # use uncertainty-aware prediction or not
    parser.add_argument('--uncertainty_aware', type=bool, default=False)

    # sort all humans and squeeze them to the front or not
    parser.add_argument('--sort_humans', type=bool, default=True)

    # use self attn in human states or not
    parser.add_argument('--use_self_attn', type=bool, default=True,
                        help='Attention size')
    
    # use self attn in vehicle states or not
    parser.add_argument('--use_self_attn_veh', type=bool, default=True,
                        help='Attention size')

    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=True,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')

    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')

    parser.add_argument(
        '--save-interval',
        type=int,
        default=200,
        help='save interval, one save per n updates (default: 100)')

    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')

    # for srnn only
    # RNN size
    parser.add_argument('--human_node_rnn_size', type=int, default=128,
                        help='Size of Human Node RNN hidden state')
    parser.add_argument('--human_human_edge_rnn_size', type=int, default=256,
                        help='Size of Human Human Edge RNN hidden state')

    # Input and output size
    parser.add_argument('--human_node_input_size', type=int, default=3,
                        help='Dimension of the node features')
    parser.add_argument('--human_human_edge_input_size', type=int, default=2,
                        help='Dimension of the edge features')
    parser.add_argument('--human_node_output_size', type=int, default=256,
                        help='Dimension of the node output')

    # Embedding size
    parser.add_argument('--human_node_embedding_size', type=int, default=64,
                        help='Embedding size of node features')
    parser.add_argument('--human_human_edge_embedding_size', type=int, default=64,
                        help='Embedding size of edge features')

    # Attention vector dimension
    parser.add_argument('--attention_size', type=int, default=64,
                        help='Attention size')
    
    parser.add_argument('--consider-veh',
                        default=False,
                        action='store_true',
                        help='whether to consider robot-vehicle spatial egdes in the model or not')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
