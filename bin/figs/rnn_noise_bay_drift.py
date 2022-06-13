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
import argparse # I should use a global parse for files in fig/

# Instantiate the parser
parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default="../core/model/model_25.0/color_reproduction_delay_unit/", type=str,
                    help='models')
parser.add_argument('--rule_name', default='color_reproduction_delay_unit', type=str,
                    help='RNN and architeture type, fix to the default throught out this paper')
parser.add_argument('--sub_dir', default="/noise_delta", type=str,
                    help='example model')
parser.add_argument('--prod_interval', default=1000, type=int,
                    help='delay epoch length')
parser.add_argument('--file_label', default='', type=str,
                    help='the figure filename would be name + file_label.pdf')
parser.add_argument('--gen_data', default=True, type=bool,
                    help='generate data or not')
parser.add_argument('--sigma_s', default=25.0, type=float,
                    help='sigma_s of the prior distribution')

arg = parser.parse_args()

model_dir = arg.model_dir
rule_name = arg.rule_name
sub_dir = arg.sub_dir
prod_intervals = arg.prod_interval
file_label = arg.file_label
gen_data = arg.gen_data
sigma_s = arg.sigma_s

out_dir = './figs/fig_data/rnn_noise_bay_' + file_label + '.json'
fig_name = 'rnn_bay_drift_' + file_label + '.pdf'

def rad2deg(arr, shift=False):
    '''
    arr ranges from -pi to pi and be converted to 0 to 360
    '''
    if shift:
        return (arr + np.pi) / 2 / np.pi * 360
    else:
        return arr / 2 / np.pi * 360

#################### READ CONFIGURATION
#################### Parameters

bin_width = 5
c_center = np.array([40, 130, 220, 310]) # 4 peaks
center_prob = 0.5

sigma_s = sigma_s / 360 * 2 * np.pi
n_epochs = 200
batch_size=16
fs_order = 10
learning_rate = 1e-6

c_center = c_center / 360 * 2 * np.pi - np.pi # convert to rad
n_center = len(c_center)

########## Parallel
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
node_name = MPI.Get_processor_name()


if gen_data:
    group = Agent_group(model_dir, rule_name, sub_dir=sub_dir, is_cuda=False)

    color_bin_group, rnn_drift_group, noise_group, bay_drift_group = [], [], [], []
    n_sub = len(group.group) # number of subject should be a multiple of thread size. This is bad, but already enough for this computation.

    for i in range(rank, n_sub, size):
        sub = group.group[i]
        color_bin, rnn_drift, rnn_noise, noise_loss = fit_ddm(sub, bin_width=bin_width, batch_size=300, sigma_init=0.05)
        print("final loss for this rnn: ", rnn_noise)
        sys.stdout.flush()

        if(np.abs(rnn_noise) < 1e-5): # to avoid numeric error in get_bay_drift
            n_sub = n_sub - 1
            continue

        bay_drift = get_bay_drift(color_bin, c_center, sigma_s, np.abs(rnn_noise), batch_size=batch_size, n_epochs=n_epochs, learning_rate=learning_rate, fs_order=fs_order)

        color_bin_group.append(color_bin); rnn_drift_group.append(rnn_drift); noise_group.append(rnn_noise);
        bay_drift_group.append(bay_drift)

        sub.state = []
        sub.fir_rate = []

    if len(color_bin_group) > 0:
        color_bin_group, rnn_drift_group, noise_group, bay_drift_group = np.concatenate(color_bin_group).astype(float), np.concatenate(rnn_drift_group).astype(float), np.array(noise_group).astype(float), np.concatenate(bay_drift_group).astype(float)
    else:
        color_bin_group, rnn_drift_group, noise_group, bay_drift_group = np.array([]), np.array([]), np.array([]), np.array([])

    rec_rnn_color, rec_rnn_drift, rec_noise, rec_bay_drift = None, None, None, None
    if rank == 0:
        n_color = color_bin.size
        rec_rnn_color = np.empty([n_sub * n_color], dtype=float)
        rec_rnn_drift = np.empty([n_sub * n_color], dtype=float)
        rec_noise = np.empty([n_sub], dtype=float)
        rec_bay_drift = np.empty([n_sub * n_color], dtype=float)

    sendcounts_rnn_color = np.array(comm.gather(len(color_bin_group), root=0)) # count the length of array in each thread
    sendcounts_rnn_drift = np.array(comm.gather(len(rnn_drift_group), root=0))
    sendcounts_noise= np.array(comm.gather(len(noise_group), root=0))
    sendcounts_bay_drift = np.array(comm.gather(len(bay_drift_group), root=0))

    comm.Gatherv(sendbuf=color_bin_group, recvbuf=(rec_rnn_color, sendcounts_rnn_color), root=0)
    comm.Gatherv(sendbuf=rnn_drift_group, recvbuf=(rec_rnn_drift, sendcounts_rnn_drift), root=0)
    comm.Gatherv(sendbuf=noise_group, recvbuf=(rec_noise, sendcounts_noise), root=0)
    comm.Gatherv(sendbuf=bay_drift_group, recvbuf=(rec_bay_drift, sendcounts_bay_drift), root=0)

    if rank == 0:
        tools.save_dic({'rnn_color': rec_rnn_color, 'rnn_drift': rec_rnn_drift, 'noise': rec_noise, 'bay_drift': rec_bay_drift}, out_dir)

if rank == 0:
    data = tools.load_dic(out_dir)

    rnn_color, rnn_drift, bay_drift = np.array(data['rnn_color']), np.array(data['rnn_drift']), np.array(data['bay_drift'])

    rnn_color, rnn_drift, bay_drift = rad2deg(rnn_color, shift=True), rad2deg(rnn_drift), rad2deg(bay_drift)
    c_center = rad2deg(c_center, shift=True)
    ########## Plot curves

    m=3.5
    color_rnn, mean_drift_rnn, se_drift_rnn = mean_se(rnn_color, rnn_drift, remove_outlier=True, m=m)
    color_rnn, mean_drift_bay, se_drift_bay = mean_se(rnn_color, bay_drift, remove_outlier=True, m=m)

    plt.figure()
    plt.scatter(rnn_color, rnn_drift)
    #plt.show()

    sns.set_theme()
    sns.set_style("ticks")
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_axes([0.2, 0.3, 0.63, 0.6])

    ax.plot(color_rnn, mean_drift_rnn, label='RNN')
    ax.fill_between(color_rnn, mean_drift_rnn - se_drift_rnn, mean_drift_rnn + se_drift_rnn, alpha=0.4)

    #ax.plot(bay_color, bay_drift, label='Bayesian')
    ax.plot(color_rnn, mean_drift_bay, label='Bayesian')
    ax.fill_between(color_rnn, mean_drift_bay - se_drift_bay, mean_drift_bay + se_drift_bay, alpha=0.4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Current color (degree)')
    ax.set_ylabel('Drift force (degree/ms)')
    
    ax.axhline(y = 0, linestyle = '--', linewidth = 3, color = 'grey')

    for cc_i in c_center:
        ax.scatter(cc_i, 0, color = 'red', s=100)
        
    plt.legend()
    plt.ticklabel_format(axis='y', style='sci', useMathText=True)

    fig.savefig('./figs/fig_collect/' + fig_name, format='pdf')

    plt.figure(2)
    noise = np.array(data['noise'])
    print('median = {} \t mean = {} \n'.format(np.median(np.abs(noise)), np.mean(np.abs(noise))))
    plt.hist(noise, bins=20)

    #plt.show()
