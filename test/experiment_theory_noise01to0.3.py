# compute the the color dispersion
import context
import os
from core.color_error import Color_error
from core.color_input import Color_input
import numpy as np
import matplotlib.pyplot as plt
from core.agent import Agent, Agent_group
from core.manifold.state_analyzer import State_analyzer
from brokenaxes import brokenaxes
import pickle
import math
from matplotlib.lines import Line2D


def removeOutliers(a, outlierConstant=1):
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant # 1.5
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    return a[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))]


Noises = ['0.10', '0.12', '0.14', '0.16', '0.18', '0.20', '0.22', '0.24', '0.26', '0.28', '0.30', '0.32']
Noise_values = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32]

######## Calculation
'''
dispersion_all = []
density_all = []
regular_all = []
exp_error_all = []
prod_int = 800 # duration of the delay
input_color = 40 # the input will be fixed to 40 degree (common color) or 85 degree (uncommon color)
batch_size = 5000
delta = 2 # d color / d phi = ( (color + delta) - (color - delta) ) / ( phi(color + delta) - phi(color - delta) )
#prior_sig = 90.0 # width of the piror
sigma_rec = None; sigma_x = None # set the noise to be default (training value)

rule_name = 'color_reproduction_delay_unit'
sub_dir = 'noise_delta/'

for i,noise in enumerate(Noises):
    model_dir_parent = "../core/model_noise/noise_"+noise+"/model_17.5/color_reproduction_delay_unit/"
    dispersion_n = []
    density_n = []
    regular_n = []
    exp_error_n= []

    for m in range(50):
        print(noise, m)
        model_dir = 'model_{}/'.format(m)  # example RNN
        f = os.path.join(model_dir_parent, model_dir, sub_dir)
        sub = Agent(f, rule_name)  # this is the outside agent creating data
        ### obtain angle of common color phi_c
        sa = State_analyzer(decoder_type='RNN_decoder')
        sa.read_rnn_file(f,rule_name)  # I strongly recommand using read_rnn_file instead of creating a agent outside (read_rnn_agent). Agent used within a state_analyzer should not be used outside.

        ### obtain angle phi_i at the end of delay in repeated trials
        input_color_list = np.ones(batch_size) * input_color  # repeatly run common color trials
        sub.do_exp(prod_intervals=prod_int, sigma_rec=sigma_rec, sigma_x=sigma_x, ring_centers=input_color_list)
        end_of_delay_state = sub.state[sub.epochs['interval'][1]]  # shape is [batch_size, hidden_size]
        phii = sa.angle(end_of_delay_state, fit_pca=True)  # Anyway, remember to fit_pca the first time use angle

        ### Dynamic dispersion
        color_error = Color_error()
        color_error.add_data(phii, np.mean(phii))
        sub_error = color_error.calculate_error()
        sqe_phi = (sub_error) ** 2
        # sqe_phi = (phii - np.mean(phii)) ** 2  # second method
        sqe_phi = removeOutliers(sqe_phi)
        dispersion = np.mean(sqe_phi)
        print('dynamic dispersion: ', dispersion)

        ### color density
        phi = sa.angle_color(np.array([input_color - delta, input_color + delta]), input_var='color')
        color_error = Color_error()
        color_error.add_data([phi[1]], [phi[0]])
        sub_error = color_error.calculate_error()[0]
        dc_dphi = 2.0 * delta / sub_error
        # dc_dphi = 2.0 * delta / (phi[1] - phi[0])
        density = (dc_dphi) ** 2
        print('color density: ', density)

        ### regularization term
        phii_mean = np.mean(phii)
        phic = sa.angle_color(np.array([input_color]), input_var='color')[0]

        # regular_term = abs(dc_dphi) * (phii_mean - phic) ** 2 / 2 / math.sqrt(dispersion) # method 1

        color_error = Color_error()
        color_error.add_data([phii_mean], [phic])
        sub_error = color_error.calculate_error()[0]
        regular_term = density * (sub_error ** 2)  # method 2
        # regular_term = density * ((phii_mean - phic) ** 2) # method 2
        print('regularization term: ', regular_term)

        ### theoretical prediction
        # print('theory: ', math.sqrt(dispersion * density) + regular_term) # method 1
        print('theory: ', math.sqrt(dispersion * density + regular_term)) # method 2

        ### experimental prediction
        sub.do_exp(prod_intervals=prod_int, sigma_rec=sigma_rec, sigma_x=sigma_x, ring_centers=input_color_list)

        color_error = Color_error()
        color_error.add_data(sub.behaviour['report_color'], sub.behaviour['target_color'])
        sub_error = color_error.calculate_error()  # circular substraction
        sqe_exp = sub_error ** 2  # first method
        sqe_exp = removeOutliers(sqe_exp)
        mse_exp = np.mean(sqe_exp)
        print('exp: ', math.sqrt(mse_exp),end='\n\n')

        dispersion_n.append(dispersion)
        density_n.append(density)
        regular_n.append(regular_term)
        exp_error_n.append(mse_exp)


    dispersion_all.append(dispersion_n)
    density_all.append(density_n)
    regular_all.append(regular_n)
    exp_error_all.append(exp_error_n)

with open('../bin/figs/fig_data/dynamic_dispersion_{}_noise.txt'.format(input_color),'wb') as fp:
    pickle.dump(dispersion_all,fp)
with open('../bin/figs/fig_data/color_density_{}_noise.txt'.format(input_color),'wb') as fp:
    pickle.dump(density_all,fp)
with open('../bin/figs/fig_data/regularization_{}_noise.txt'.format(input_color),'wb') as fp:
    pickle.dump(regular_all,fp)
with open('../bin/figs/fig_data/experimental_error_{}_noise.txt'.format(input_color),'wb') as fp:
    pickle.dump(exp_error_all,fp)
# '''

######### Plot dispersion
# '''
with open('../bin/figs/fig_data/dynamic_dispersion_40_noise.txt','rb') as fp:
    dispersion_all = pickle.load(fp)

dispersion_unadapted = removeOutliers(np.array(dispersion_all[-1]))
unadapted_mean = np.mean(dispersion_unadapted)
unadapted_std = np.std(dispersion_unadapted)
unadapted_means = [unadapted_mean] * len(Noise_values)
upper = [unadapted_mean + unadapted_std] * len(Noise_values)
lower = [unadapted_mean - unadapted_std] * len(Noise_values)

fig, ax = plt.subplots(figsize=(4,3.5))
# ax.plot(Noise_values,unadapted_means,'k',linestyle='--',alpha=0.5)
# ax.fill_between(Noise_values, lower, upper, color='grey',alpha=0.2)
ax.boxplot(dispersion_all[:-1],showfliers=False,positions=Noise_values[:-1],patch_artist = True,widths=0.01,boxprops=dict(facecolor='lightblue', color='blue'))
ax.set_ylabel('Dynamic Dispersion',fontsize=13)
ax.set_xlabel('Noise',fontsize=15)
ax.set_xticks([0.1,0.16,0.22,0.28,0.34])
ax.set_xticklabels(['0.1','0.16','0.22','0.28','0.34'])
ax.set_xlim((0.08,0.34))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('../bin/figs/fig_collect/dynamic_dispersion_common_noise.svg',format='svg',bbox_inches='tight')
plt.show()
# '''

######## Plot color occupancy
'''
with open('../bin/figs/fig_data/color_density_40_noise.txt','rb') as fp:
    density_all = pickle.load(fp)

density_unadapted = removeOutliers(np.array(density_all[-1]))
unadapted_mean = np.mean(density_unadapted)
unadapted_std = np.std(density_unadapted)
unadapted_means = [unadapted_mean] * len(Noise_values)
upper = [unadapted_mean + unadapted_std] * len(Noise_values)
lower = [unadapted_mean - unadapted_std] * len(Noise_values)

fig, ax = plt.subplots(figsize=(4,3.5))
ax.boxplot(density_all[:-1],showfliers=False,positions=Noise_values[:-1],patch_artist = True,widths=0.01,boxprops=dict(facecolor='lightblue', color='blue'))
ax.set_ylabel('Color density',fontsize=13)
ax.set_xlabel('Noise',fontsize=15)
ax.set_xticks([0.1,0.16,0.22,0.28,0.34])
ax.set_xticklabels(['0.1','0.16','0.22','0.28','0.34'])
ax.set_xlim((0.08,0.34))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('../bin/figs/fig_collect/color_density_common_noise.svg',format='svg',bbox_inches='tight')
plt.show()
# '''


# Compare theoritical and experimental results (side-by-side boxes)
'''
with open('../bin/figs/fig_data/color_density_40_noise.txt','rb') as fp:
    density_all = np.array(pickle.load(fp))
with open('../bin/figs/fig_data/dynamic_dispersion_40_noise.txt','rb') as fp:
    dispersion_all = np.array(pickle.load(fp))
with open('../bin/figs/fig_data/experimental_error_40_noise.txt', 'rb') as fp:
    exp_error_all = np.array(pickle.load(fp))
with open('../bin/figs/fig_data/regularization_40_noise.txt', 'rb') as fp:
    regular_all = np.array(pickle.load(fp))



exp_error_box = [np.sqrt(exp_error_all[i]) for i in range(exp_error_all.shape[0])]

# theo_error = np.sqrt(density_all*dispersion_all) + regular_all # method 1
theo_error = np.sqrt(density_all * dispersion_all + regular_all) # method 2
theo_error_box = [list(theo_error[i]) for i in range(theo_error.shape[0])]



fig, ax = plt.subplots(figsize=(6,3.5))

positions_1 = [x+0.0025 for x in Noise_values]
positions_2 = [x-0.0025 for x in Noise_values]

ax.boxplot(theo_error_box[:-1],showfliers=False,positions=positions_1[:-1],patch_artist = True,widths=0.005,boxprops=dict(facecolor='lightblue', color='darkblue'),medianprops = dict(color = "darkblue", linewidth = 1.5))
ax.boxplot(exp_error_box[:-1],showfliers=False,positions=positions_2[:-1],patch_artist = True,widths=0.005,boxprops=dict(facecolor='salmon', color='darkred'),medianprops = dict(color = "darkred", linewidth = 1.5))
ax.set_ylabel('Memory Error',fontsize=13)
ax.set_xlabel('Noise',fontsize=15)
ax.set_xticks([0.1,0.16,0.22,0.28,0.34])
ax.set_xticklabels(['0.1','0.16','0.22','0.28','0.34'])
ax.set_xlim((0.08,0.32))
ax.plot([],[],color='lightblue',linewidth=8,label='Theory')
ax.plot([],[],color='salmon',linewidth=8,label='Experiment')
ax.legend(loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('../bin/figs/fig_collect/exp_theo_comparison_noise.svg',format='svg',bbox_inches='tight')
plt.show()
# '''




