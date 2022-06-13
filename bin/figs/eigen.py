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

try:
    model_dir = sys.argv[1]
    rule_name = sys.argv[2]
    sub_dir = sys.argv[3]
    out_path = sys.argv[5]
    fig_out_path = sys.argv[6]
except:
    rule_name = 'color_reproduction_delay_unit'
    model_dir = '../core/model_local/color_reproduction_delay_unit/'
    sub_dir = '/noise_delta'
    out_path = './figs/fig_data/fixpoint_color.json'
    fig_out_path = './figs/fig_collect/dis_attractor.pdf'

try:
    if sys.argv[4] == 'Y': # set false so it will not generate data
        gen_data = True
    else:
        gen_data = False
except:
    gen_data = False

hidden_size = 256
prod_intervals_mplot = 800 # for ploting delay trajectories, not for searching fixpoints
batch_size = 500
n_epochs = 5000

lr=1
speed_thre = None # speed lower than this we consider it as fixpoints, slow points otherwise
milestones = [2000, 6000, 9000]
alpha=0.7
initial_type='delay_ring'
sigma_init = 0 # Specify the noise adding on initial searching points
common_colors = [40, 130, 220, 310]

#model_label = ["90", "25", "3"]
#model_dir_list = ["../core/model/color_reproduction_delay_unit_90/",
#                  "../core/model/color_reproduction_delay_unit_25/",
#                  "../core/model/color_reproduction_delay_unit_3/",]

model_dir_label = ["90", "30", "25", "20", "10", "3"]
model_dir_list = ["../core/model/model_90/color_reproduction_delay_unit/",
                  "../core/model/model_30/color_reproduction_delay_unit/",
                  "../core/model/model_25/color_reproduction_delay_unit/",
                  "../core/model/model_20/color_reproduction_delay_unit/",
                  "../core/model/model_10/color_reproduction_delay_unit/",
                  "../core/model/model_3/color_reproduction_delay_unit/",]
#model_dir_label = ["20"]
#model_dir_list = ["../core/model/model_20/color_reproduction_delay_unit/",]

def get_eigv(model_dir, sub_dir):
    group = Agent_group(model_dir, rule_name, sub_dir=sub_dir)

    att_ev_real, att_ev_imag = [], []
    sad_ev_real, sad_ev_imag = [], []
    att_status = []

    for sub in group.group:
        model_dir = sub.model_dir
        fixpoint_output = ultimate_find_fixpoints(model_dir, rule_name, batch_size=batch_size, n_epochs=n_epochs, lr=lr, speed_thre=speed_thre, milestones=milestones, initial_type=initial_type, sigma_init=sigma_init, witheigen=True, prod_intervals=0) # find the angle of fixpoints
        # decode the angles to color
        rnn_de = RNN_decoder()
        rnn_de.read_rnn_agent(sub)

        for i, ev in enumerate(fixpoint_output['eigval']): # iterate every fixpoint
            idx_max = np.argmax(ev) # find the largest eigenvalue
            if fixpoint_output['att_status'][i]: # if its an attractor
                att_ev_real.append(np.real(ev[idx_max])) # collect the eigenvalue
                att_ev_imag.append(np.imag(ev[idx_max])) # collect the eigenvalue
            else:
                sad_ev_real.append(np.real(ev[idx_max])) # collect the eigenvalue
                sad_ev_imag.append(np.imag(ev[idx_max])) # collect the eigenvalue
    return att_ev_real, att_ev_imag, sad_ev_real, sad_ev_imag

if gen_data:
    data_dic = {}
    for i, model_dir in enumerate(model_dir_list):
        att_ev_real, att_ev_imag, sad_ev_real, sad_ev_imag = get_eigv(model_dir, sub_dir='/noise_delta')
        data_dic['att_ev_real_' + model_dir_label[i]] = att_ev_real
        data_dic['att_ev_imag_' + model_dir_label[i]] = att_ev_imag
        data_dic['sad_ev_real_' + model_dir_label[i]] = sad_ev_real
        data_dic['sad_ev_imag_' + model_dir_label[i]] = sad_ev_imag

    tools.save_dic(data_dic, out_path)

alpha=0.5
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
            data_kwd['name'] = data_kwd['name'] + [key] * len(data[key])
    return data_kwd
key_list = ['att_ev_real_90', 'att_ev_real_25', 'att_ev_real_20', 'att_ev_real_10', 'att_ev_real_3']
n_att =[]
for key in key_list:
    n_att.append(len(data[key]))

plt.figure()
plt.plot(range(len(n_att)), n_att)
#plt.show()

data_real = data_to_sns(data, kwd_list=['real', 'att'])
data_imag = data_to_sns(data, kwd_list=['imag', 'att'])
data_sad_real = data_to_sns(data, kwd_list=['real', 'sad'])
data_sad_imag = data_to_sns(data, kwd_list=['imag', 'sad'])
import seaborn as sns
ax = sns.boxplot(x='name', y='val', data=data_real)
ax.set_xlabel('Sigma in the prior distribution')
ax.set_ylabel('The real component of largest eigenvalue in Attractors')
#plt.show()
ax = sns.boxplot(x='name', y='val', data=data_imag)
#plt.show()
ax = sns.boxplot(x='name', y='val', data=data_sad_real)
#plt.show()
ax = sns.boxplot(x='name', y='val', data=data_sad_imag)
#plt.show()

#
#data_sns = {'sig': [], 'real': [], 'imag': []}
#
#data_sns['real'] = np.concatenate((data['att_ev_real_90'], data['att_ev_real_25'], data['att_ev_real_10'], data['att_ev_real_3']))
#
#for key in data:
#    data_sns['sig'] = data_sns['sig'] + [key] * len(data[key])

#att_list = ['att_ev_real_' + '90', 'att_ev_real_' + '25']
#for item in att_list:
#    sns.displot(data=data, x=item)
#plt.show()

#att_ev_real = np.array(data['att_ev_real_90'], dtype=float)
#att_ev_imag = np.array(data['att_ev_imag_90'], dtype=float)
#sad_ev_real = np.array(data['sad_ev_real_90'], dtype=float)
#sad_ev_imag = np.array(data['sad_ev_imag_90'], dtype=float)
#
#plt.figure()
#plt.hist(att_ev_real, bins=100, alpha=alpha)
#plt.hist(sad_ev_real, alpha=alpha)
#
#att_ev_real = np.array(data['att_ev_real_25'], dtype=float)
#att_ev_imag = np.array(data['att_ev_imag_25'], dtype=float)
#sad_ev_real = np.array(data['sad_ev_real_25'], dtype=float)
#sad_ev_imag = np.array(data['sad_ev_imag_25'], dtype=float)
#
#plt.hist(att_ev_real, alpha=alpha)
#plt.hist(sad_ev_real, alpha=alpha)
#
#att_ev_real = np.array(data['att_ev_real_10'], dtype=float)
#att_ev_imag = np.array(data['att_ev_imag_10'], dtype=float)
#sad_ev_real = np.array(data['sad_ev_real_10'], dtype=float)
#sad_ev_imag = np.array(data['sad_ev_imag_10'], dtype=float)
#
#plt.hist(att_ev_real, alpha=alpha)
#plt.hist(sad_ev_real, alpha=alpha)
#plt.show()

#sns.set()
#sns.set_style("ticks")
#
#fig = plt.figure(figsize=(3, 3))
#ax = fig.add_axes([0.23, 0.2, 0.6, 0.7])
#
#bins = np.histogram_bin_edges(att_colors, bins=36, range=(0, 360))
#
#sns.histplot(att_colors, ax=ax, bins=bins, stat='count')
#for cm in common_colors:
#    sns.histplot(att_colors[(cm-10 < att_colors) * (att_colors < cm+10)], ax=ax, bins=bins, stat='count', color='red')
#
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
#ax.set_xlabel('Decoded color of attractors')
#ax.set_xticks([0, 90, 180, 270, 360])
#
#fig.savefig(fig_out_path, format='pdf')
#
#plt.show()
