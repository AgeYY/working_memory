# Import libraries
import numpy as np
import shutil
import os
import sys
import torch
from mpi4py import MPI
import context
from core.train_batch import train_model

# Append the current directory to the system path
sys.path.append(os.getcwd())

# Configuration setup
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
n_device = torch.cuda.device_count()

# Hyperparameters configuration
sig_list = [17.5, 90.0]
n_models = 10
model_base_dir = '../core/model/'
noise_step = 0.4 # in step 3, we increase the sigma_rec noise progressively by noise_step
is_cuda = True
rule_name = "color_reproduction_delay_unit"
hp_replace = {
    'prod_interval': [0, 1000],
    'l2_jac': -1,
    'sigma_rec': 1.2, # 0.6 is a bit small
    'sigma_x': 0.,
    'num_unit': 12,
    'n_input': 13,
    'n_output': 12,
    'n_rnn': 256,
    'bias_method': 'vonmises',
    'stop_color_error': 1,
    'stop_noise_color_error': 30,
    'stop_delta_color_error': 30,
    'min_trials': 2e5,
    'bias_centers': [40., 130., 220., 310.]
    }
#################### Hyperparameters finished

def copy_files(src_folder, dest_folder):
    """
    Copy all files from src_folder to dest_folder.
    
    :param src_folder: Source folder path
    :param dest_folder: Destination folder path
    """
    
    # Check if source folder exists
    if not os.path.exists(src_folder):
        print(f"Source folder {src_folder} does not exist.")
        return

    # Create destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Iterate over all files in source folder and copy them
    for filename in os.listdir(src_folder):
        file_path = os.path.join(src_folder, filename)
        if os.path.isfile(file_path):
            shutil.copy(file_path, dest_folder)

# Training Utility Functions
def step_1(hp_replace, rule_name, i, device_id, model_base_uniform, is_cuda):
    hp_replace_temp = hp_replace.copy()
    hp_replace_temp.update({
        'prod_interval': [0, 0],
        'sigma_rec': 0.,
        'sigma_x': 0,
        'l2_jac': -0.1,  # negative means no penalty
        'l2_weight': -0.1,
        'l2_firing_rate': -0.1,
        'bias_method': 'uniform',
        'learning_rate': 0.0005
    })
    cost = train_model(rule_name, i, device_id=device_id, hp_replace=hp_replace_temp, is_cuda=is_cuda, final_name='progressive_' + str(0), model_base=model_base_uniform)
    return cost

def step_2(hp_replace, rule_name, i, device_id, model_base_uniform, is_cuda):
    hp_replace_temp = hp_replace.copy()
    hp_replace_temp.update({
        'sigma_rec': 0.,
        'sigma_x': 0,
        'l2_jac': -0.1,  # negative means no penalty
        'l2_weight': -0.1,
        'l2_firing_rate': -0.1,
        'bias_method': 'uniform',
    })
    cost = train_model(rule_name, i, device_id=device_id, hp_replace=hp_replace_temp, is_cuda=is_cuda, final_name='progressive_' + str(1), model_base=model_base_uniform)
    return cost

def step_3(hp_replace, rule_name, i, device_id, model_base_uniform, is_cuda, noise_step):
    hp_replace_temp = hp_replace.copy()
    hp_replace_temp.update({
        'bias_method': 'uniform',
    })

    err_tol = 1e-5
    noise_arr = np.arange(noise_step, hp_replace_temp['sigma_rec'], noise_step)
    if np.abs(noise_arr[-1] - hp_replace_temp['sigma_rec']) >= err_tol: # add end point
        noise_arr = np.append(noise_arr, hp_replace_temp['sigma_rec'])

    for noise in noise_arr:
        hp_replace_temp['sigma_rec'] = noise
        cost = train_model(rule_name, i, device_id=device_id, hp_replace=hp_replace_temp, is_cuda=is_cuda, final_name='noise', model_base=model_base_uniform)
    return cost

def step_4(hp_replace, rule_name, i, device_id, model_base_uniform, model_base_bias, is_cuda):
    # copy model in model_base_uniform to model_base_bias
    uniform_model_path = os.path.join(model_base_uniform, rule_name, 'model_'+str(i)) # confirm model path in train_batch.py
    bias_model_path = os.path.join(model_base_bias, rule_name, 'model_'+str(i))
    copy_files(uniform_model_path, bias_model_path)

    cost = train_model(rule_name, i, device_id=device_id, hp_replace=hp_replace, is_cuda=is_cuda, final_name='noise_delta', model_base=model_base_bias)
    return cost

# Training uniform and bias model
def perform_training_uniform(i, n_models, size, hp_replace, is_cuda, rule_name, model_base_uniform, noise_step):
    device_id = i % n_device

    step_1(hp_replace, rule_name, i, device_id, model_base_uniform, is_cuda)
    step_2(hp_replace, rule_name, i, device_id, model_base_uniform, is_cuda)
    step_3(hp_replace, rule_name, i, device_id, model_base_uniform, is_cuda, noise_step)

def perform_training_bias(i, n_models, size, hp_replace, is_cuda, rule_name, model_base_uniform, model_base_bias):
    device_id = i % n_device
    step_4(hp_replace, rule_name, i, device_id, model_base_uniform, model_base_bias, is_cuda)


###################### Main function
model_base_uniform = model_base_dir + 'model_uniform/'

# train the uniform base model
for i in range(rank, n_models, size):
    perform_training_uniform(i, n_models, size, hp_replace, is_cuda, rule_name, model_base_uniform, noise_step)
    torch.cuda.empty_cache()

# train with different sig
for sig in sig_list:
    hp_replace['sig'] = sig
    model_base_bias = model_base_dir + 'model_' + str(round(sig, 1)) + '/'
    for i in range(rank, n_models, size):
        perform_training_bias(i, n_models, size, hp_replace, is_cuda, rule_name, model_base_uniform, model_base_bias)

        torch.cuda.empty_cache()
