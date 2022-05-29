# compute the fixpoints eigenvalues
import context
import seaborn as sns
from core.rnn_decoder import RNN_decoder
from core.agent import Agent, Agent_group
import numpy as np
import matplotlib.pyplot as plt
import core.tools as tools
from core.manifold.ultimate_fix_point import ultimate_find_fixpoints
import seaborn as sns

rule_name = 'color_reproduction_delay_unit'
sub_dir = '/noise_delta'
out_path = '../bin/figs/fig_data/eigenvalue.json'
fig_out_path = '../bin/figs/fig_collect/dis_attractor.pdf'

gen_data = False

hidden_size = 256
batch_size = 500
n_epochs = 5000

lr=1
speed_thre = None # speed lower than this we consider it as fixpoints, slow points otherwise
milestones = [2000, 6000, 9000]
alpha=0.5
initial_type='delay_ring'
sigma_init = 0 # Specify the noise adding on initial searching points
common_colors = [40, 130, 220, 310]
key = ["90", "30", "25", "20", "10", "3"]
#model_label = ["90", "25", "3"]

model_dir_list = []
for ky in key:
    model_dir_list.append("../core/model/model_" + ky + "/color_reproduction_delay_unit/")

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

########## the number of attractors
#key_list = ['att_ev_real_90', 'att_ev_real_25', 'att_ev_real_20', 'att_ev_real_10', 'att_ev_real_3']
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

ax = sns.boxplot(x='name', y='val', data=data_real)

ax.set_xlabel(r'$\sigma_s$ of the prior distribution')
ax.set_ylabel('The real component of largest eigenvalue in attractors')
plt.show()

#ax = sns.boxplot(x='name', y='val', data=data_imag)
#plt.show()
#ax = sns.boxplot(x='name', y='val', data=data_sad_real)
#plt.show()
#ax = sns.boxplot(x='name', y='val', data=data_sad_imag)
#plt.show()
