# training serveral configuration on cluster by mpi4py
# run by mpiexec -np 2 python train_cluster.py
import context
from core.train_batch import train_model
import os
from mpi4py import MPI
import torch
import sys
import argparse

sys.path.append(os.getcwd())

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
n_device = torch.cuda.device_count()

# Instantiate the parser
parser = argparse.ArgumentParser()

parser.add_argument('--sig_s', default=25.0, type=float,
                    help='sigma_s in training prior distribution')
parser.add_argument('--n_model', default=1, type=int,
                    help='Trianing n models with the same hyperparameters')
parser.add_argument('--model_base_dir', default='../core/model_short_res_rep/', type=str,
                    help='Trianing n models with the same hyperparameters')

arg = parser.parse_args()

sig = round( float(arg.sig_s), 1 )
sig = 3.0
n_models = int(arg.n_model)
model_base = arg.model_base_dir + 'model_' + str(round(sig, 1)) + '/'

#################### Parameters
is_cuda = True
rule_name = "color_reproduction_delay_unit" # RNN types: can be "color_reproduction_delay_unit", or 'color_reproduction_delay_tri'. See explanation in default.py

hp_replace = {'prod_interval': [0, 1000], 'l2_jac': -1, 'sigma_rec': 0.2, 'sigma_x': 0.2, 'num_unit': 12, \
              'n_input': 13, 'n_output': 12, 'n_rnn': 256, 'bias_method': 'vonmises', 'sig': sig, \
              'min_trials': 2e5, \
              'bias_centers': [40., 130., 220., 310.], 'response_duration': 40, 'pulse_duration_go': 40
             } # n_input must = n_output + 1. Check default for the meaning of each parameters.
#################### Training
for i in range(rank, n_models, size):
    ## We use progressive training
    # step 1: prod_interval = [0, 0], no noise, no penalty, uniform prior distribution
    hp_replace_temp = hp_replace.copy()

    hp_replace_temp['prod_interval'] = [0, 0]
    hp_replace_temp['sigma_rec'] = 0.
    hp_replace_temp['sigma_x'] = 0
    hp_replace_temp['l2_jac'] = -0.1 # negative means no penalty
    hp_replace_temp['l2_weight'] = -0.1
    hp_replace_temp['l2_firing_rate'] = -0.1
    hp_replace_temp['bias_method'] = 'uniform'
    hp_replace_temp['learning_rate'] = 0.0005 # In the first step we need larger learning rate to find the solution.

    cost = train_model(rule_name, i, device_id=i % n_device, hp_replace=hp_replace_temp, is_cuda=is_cuda, final_name='progressive_' + str(0), model_base=model_base)

    # step 2: prod_interval = [0, 1000], adjust the learning rate
    hp_replace_temp['prod_interval'][1] = hp_replace_temp['prod_interval'][1] + 1000
    hp_replace_temp['learning_rate'] = 0.00005
    cost = train_model(rule_name, i, device_id=i % n_device, hp_replace=hp_replace_temp, is_cuda=is_cuda, final_name='progressive_' + str(1), model_base=model_base)

    # step 3: add penalty and noise, but still uniform distribution
    hp_replace_temp = hp_replace.copy()
    hp_replace_temp['bias_method'] = 'uniform'
    cost = train_model(rule_name, i, device_id=i % n_device, hp_replace=hp_replace_temp, is_cuda=is_cuda, final_name='noise', model_base=model_base)

    # step 4: use original hp_replace
    cost = train_model(rule_name, i, device_id=i % n_device, hp_replace=hp_replace, is_cuda=is_cuda, final_name='noise_delta', model_base=model_base)
    torch.cuda.empty_cache()
