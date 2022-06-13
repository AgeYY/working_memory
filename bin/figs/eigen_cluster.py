# compute the fixpoints eigenvalues
import context
import sys
from core.rnn_decoder import RNN_decoder
from core.agent import Agent, Agent_group
import numpy as np
from core.manifold.fix_point import Fix_point_finder, Hidden0_helper
from core.data_plot.manifold_ploter import Manifold_ploter as MPloter
import matplotlib.pyplot as plt
from core.color_manager import Degree_color
from sklearn.decomposition import PCA
import core.tools as tools
from core.data_plot.plot_tool import color_curve_plot
from core.manifold.ultimate_fix_point import ultimate_find_fixpoints
import torch
from core.manifold.state_analyzer import State_analyzer
import seaborn as sns
from mpi4py import MPI
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--sub_dir', default='/noise_delta', type=str, help='mdoel sub dir')
parser.add_argument('--file_label', default='', type=str, help='file name for storing the data and figure')
parser.add_argument('--gen_data', default=False, action='store_true', help='compute the data or not')

arg = parser.parse_args()

file_label = arg.file_label
gen_data = arg.gen_data

out_path = './figs/fig_data/fixpoint_eigen_' + file_label + '.json'
fig_out_path_att = './figs/fig_collect/fixpoint_eigen_att_' + file_label + '.pdf'
fig_out_path_sad = './figs/fig_collect/fixpoint_eigen_sad_' + file_label + '.pdf'
rule_name = 'color_reproduction_delay_unit'

batch_size = 300
n_epochs = 20000

lr=1
speed_thre = None # speed lower than this we consider it as fixpoints, slow points otherwise
milestones = [6000, 12000, 18000]
alpha=0.7
initial_type='delay_ring'
sigma_init = 0 # Specify the noise adding on initial searching points
common_colors = [40, 130, 220, 310]
sub_max = 50

model_dir_label = ["90.0", "30.0", '27.5', "25.0", '22.5', "20.0", '17.5', '15.0', '12.5', "10.0", "3.0"]
model_dir_list = ['../core/model/model_' + lb + '/color_reproduction_delay_unit/' for lb in model_dir_label]

def get_eigv(model_dir, sub_dir):
    group = Agent_group(model_dir, rule_name, sub_dir=sub_dir)

    att_ev_real_avg_sub = []
    att_ev_imag_avg_sub = []
    sad_ev_real_avg_sub = []
    sad_ev_imag_avg_sub = []

    count = 0
    for sub in group.group:
        count += 1
        if count > sub_max:
            break

        att_ev_real, att_ev_imag = [], []
        sad_ev_real, sad_ev_imag = [], []
        att_status = []

        model_dir = sub.model_dir
        fixpoint_output = ultimate_find_fixpoints(model_dir, rule_name, batch_size=batch_size, n_epochs=n_epochs, lr=lr, speed_thre=speed_thre, milestones=milestones, initial_type=initial_type, sigma_init=sigma_init, witheigen=True, prod_intervals=0) # find the angle of fixpoints

        for i, ev in enumerate(fixpoint_output['eigval']): # iterate every fixpoint
            idx_max = np.argmax(ev) # find the largest eigenvalue
            if fixpoint_output['att_status'][i]: # if its an attractor
                att_ev_real.append(np.real(ev[idx_max])) # collect the eigenvalue
                att_ev_imag.append(np.imag(ev[idx_max])) # collect the eigenvalue
            else:
                sad_ev_real.append(np.real(ev[idx_max])) # collect the eigenvalue
                sad_ev_imag.append(np.imag(ev[idx_max])) # collect the eigenvalue

        att_ev_real_avg_sub.append(np.mean(att_ev_real))
        att_ev_imag_avg_sub.append(np.mean(att_ev_imag))
        sad_ev_real_avg_sub.append(np.mean(sad_ev_real))
        sad_ev_imag_avg_sub.append(np.mean(sad_ev_imag))

    return att_ev_real_avg_sub, att_ev_imag_avg_sub, sad_ev_real_avg_sub, sad_ev_imag_avg_sub

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if gen_data:
    data_dic = {}
    for i in range(rank, len(model_dir_list), size):
        att_ev_real, att_ev_imag, sad_ev_real, sad_ev_imag = get_eigv(model_dir_list[i], sub_dir='/noise_delta')
        data_dic['att_ev_real_' + model_dir_label[i]] = att_ev_real
        data_dic['att_ev_imag_' + model_dir_label[i]] = att_ev_imag
        data_dic['sad_ev_real_' + model_dir_label[i]] = sad_ev_real
        data_dic['sad_ev_imag_' + model_dir_label[i]] = sad_ev_imag

    tot_dic = comm.gather(data_dic, root=0)

    if rank == 0:
        tot_data_dic = {k: v for l in tot_dic for k, v in l.items()}
        tools.save_dic(tot_data_dic, out_path)

if rank==0:
    alpha=0.5
    ci='sd'
    data = tools.load_dic(out_path)

    def data_to_sns(data, kwd_list=['real', 'att']):
        data_kwd = {'name': [], 'val': []}
        for key in data:
            yes = True
            for kwd in kwd_list:
                if not (kwd in key):
                    yes = False
                    break
            if yes:
                data_kwd['val'] = np.concatenate((data_kwd['val'], np.array(data[key], dtype=float)))
                #data_kwd['name'] = data_kwd['name'] + [key] * len(data[key])
                if key[-4:] == '_3.0':
                    data_kwd['name'].extend([float(key[-3:])] * len(data[key]))
                else:
                    data_kwd['name'].extend([float(key[-4:])] * len(data[key]))
        return data_kwd

    #key_list = ['att_ev_real_90.0', 'att_ev_real_25.0', 'att_ev_real_20.0', 'att_ev_real_10.0', 'att_ev_real_3.0']
    #key_list = ['att_ev_real_90.0', 'att_ev_real_25.0']
    #n_att =[]
    #for key in key_list:
    #    n_att.append(len(data[key]))

    #plt.figure()
    #plt.plot(range(len(n_att)), n_att)
    #plt.show()

    data_real = data_to_sns(data, kwd_list=['real', 'att'])
    data_imag = data_to_sns(data, kwd_list=['imag', 'att'])
    data_sad_real = data_to_sns(data, kwd_list=['real', 'sad'])
    data_sad_imag = data_to_sns(data, kwd_list=['imag', 'sad'])

    import seaborn as sns
    sns.set_style("ticks")
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(3,3))

    def set_fig(ax1, ax2):
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(direction='in')

        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.tick_params(direction='in', left=False)

        ax1.set_xlim(0, 32)
        ax1.set_xticks([0, 8, 16, 22, 30])
        ax1.set_xlabel(r'$\sigma_s$ of the prior distribution')
        ax2.set_xlim(88, 92)

    sns.lineplot(x='name', y='val', data=data_real, err_style='bars', marker='o', ci=ci, estimator=np.mean, ax=ax1)
    sns.lineplot(x='name', y='val', data=data_real, ax=ax2, err_style='bars', ci=ci, marker="o", estimator=np.mean)

    #sns.lineplot(x='name', y='val', data=data_imag, err_style='bars', marker='o', ci=ci, estimator=np.mean, ax=ax1)
    #sns.lineplot(x='name', y='val', data=data_imag, ax=ax2, err_style='bars', ci=ci, marker="o", estimator=np.mean)

    set_fig(ax1, ax2)

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    fig.savefig(fig_out_path_att, format='pdf')
    #plt.show()

    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(3,3))
    sns.lineplot(x='name', y='val', data=data_sad_real, err_style='bars', marker='o', ci=ci, estimator=np.mean, ax=ax1)
    sns.lineplot(x='name', y='val', data=data_sad_real, ax=ax2, err_style='bars', ci=ci, marker="o", estimator=np.mean)

    #sns.lineplot(x='name', y='val', data=data_sad_imag, err_style='bars', marker='o', ci=ci, estimator=np.mean, ax=ax1)
    #sns.lineplot(x='name', y='val', data=data_sad_imag, ax=ax2, err_style='bars', ci=ci, marker="o", estimator=np.mean)

    set_fig(ax1, ax2)

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    fig.savefig(fig_out_path_sad, format='pdf')
    #plt.show()


