# compare the simulation of RNN and DDM during the delay
import context
import seaborn as sns
import matplotlib.pyplot as plt
from core.agent import Agent, Agent_group
import numpy as np
from core.tools import mean_se, save_dic, load_dic, find_nearest
from core.ddm import fit_ddm
from core.bay_drift import get_bay_drift
import core.tools as tools
from mpi4py import MPI
import sys

def rad2deg(arr, shift=False):
    '''
    arr ranges from -pi to pi and be converted to 0 to 360
    '''
    if shift:
        return (arr + np.pi) / 2 / np.pi * 360
    else:
        return arr / 2 / np.pi * 360

#################### READ CONFIGURATION
try:
    model_dir = sys.argv[1]
    rule_name = sys.argv[2]
    sub_dir = sys.argv[3]
except:
    model_dir = '../core/model_local/color_reproduction_delay_unit_vonmise_cp7_np2_2c/'
    rule_name = 'color_reproduction_delay_unit'
    sub_dir = '/noise_delta'

try:
    if sys.argv[4] == 'Y': # set false so it will not generate data
        gen_data = True
    else:
        gen_data = False
except:
    gen_data = False

#################### READ CONFIGURATION
#################### Parameters

c_center = np.array([40, 130, 220, 310]) # 4 peaks
#c_center = np.array([90, 270]) # for 2 peaks
sigma_s = 10.0 / 360 * 2 * np.pi
#sigma_n = 0.0061 # for 2 peaks, np2
n_epochs = 300
batch_size=16
fs_order = 10
learning_rate = 1e-5
center_prob = 0.5
out_dir = './figs/fig_data/rnn_noise.json'
repeat = 12 # n_sub * repeat must be multiple of number of threads

c_center = c_center / 360 * 2 * np.pi - np.pi # convert to rad
n_center = len(c_center)

########## Parallel
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
node_name = MPI.Get_processor_name()


if gen_data:
    group = Agent_group(model_dir, rule_name, sub_dir=sub_dir, is_cuda=False)

    color_bin_group, rnn_drift_group, noise_group = [], [], []
    n_sub = len(group.group)

    for i in range(rank, n_sub * repeat, size):
        sub = group.group[i % n_sub]
        #color_bin, rnn_drift, rnn_noise, noise_loss = fit_ddm(sub, bin_width=10)
        color_bin, rnn_drift, rnn_noise, noise_loss = fit_ddm(sub, bin_width=5)
        color_bin_group.append(color_bin); rnn_drift_group.append(rnn_drift); noise_group.append(rnn_noise);

    color_bin_group, rnn_drift_group, noise_group = np.concatenate(color_bin_group), np.concatenate(rnn_drift_group), np.array(noise_group)

    rec_rnn_color, rec_rnn_drift, rec_noise = None, None, None
    if rank == 0:
        n_color = color_bin.size
        rec_rnn_color = np.empty([n_sub * n_color * repeat])
        rec_rnn_drift = np.empty([n_sub * n_color * repeat])
        rec_noise = np.empty([n_sub * repeat])

    comm.Gather(color_bin_group, rec_rnn_color, root=0)
    comm.Gather(rnn_drift_group, rec_rnn_drift, root=0)
    comm.Gather(noise_group, rec_noise, root=0)

    if rank == 0:
        tools.save_dic({'rnn_color': rec_rnn_color, 'rnn_drift': rec_rnn_drift, 'noise': rec_noise}, out_dir)

if rank == 0:
    data = tools.load_dic(out_dir)

    plt.figure(0)
    noise = data['noise']
    print('median = {} \t mean = {} \n'.format(np.median(noise), np.mean(noise)))
    plt.hist(noise)
    #plt.show()

    color = data['rnn_color']; drift = data['rnn_drift'];

    plt.figure(1)
    plt.plot(color, drift)
    #plt.show()
