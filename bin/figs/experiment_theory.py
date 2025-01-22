# compute the the color dispersion
import context
import os
from core.color_error import Color_error
from core.color_input import Color_input
from core.tools import removeOutliers
import numpy as np
import matplotlib.pyplot as plt
from core.agent import Agent, Agent_group
from core.manifold.state_analyzer import State_analyzer
from core.ploter import error_bar_plot
from brokenaxes import brokenaxes
import pickle
import math
from matplotlib.lines import Line2D

plt.rc('ytick', labelsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('axes', linewidth=2)


model_names = ['3.0','10.0','12.5','15.0','17.5','20.0', '22.5','25.0','27.5', '30.0','90.0']
sigmas = [3.0, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 90.0]

######## Calculation (uncomment this part to generate new data)
# '''
dispersion_all = []
density_all = []
regular_all = []
exp_error_all = []
mean_bias_all = []

# Simulation parameters
prod_int = 800  # Duration of the delay
input_color = 40  # The input will be fixed to 40 degree (common color) or 85 degree (uncommon color)
batch_size = 500  # Number of trials
delta = 2  # d color / d phi = ( (color + delta) - (color - delta) ) / ( phi(color + delta) - phi(color - delta) )
# prior_sig = 90.0  # width of the piror
sigma_rec = None; sigma_x = None  # Set the noise to be default (training value)
rule_name = 'color_reproduction_delay_unit'
# model_dir = 'model_1/' # example RNN
sub_dir = 'noise_delta/'

# Iterate over all sigmas to compute metrics for RNN
for i,prior_sig in enumerate(sigmas):
    model_dir_parent = "../core/model/model_" + str(prior_sig) + "/color_reproduction_delay_unit/"
    dispersion_sig = []
    density_sig = []
    regular_sig = []
    exp_error_sig = []
    mean_bias_sig = []

    # Iterate over 50 RNNs for each sigma
    for m in range(50):
        print(prior_sig,m)
        model_dir = 'model_{}/'.format(m)  # example RNN
        f = os.path.join(model_dir_parent, model_dir, sub_dir)
        sub = Agent(f, rule_name)  # this is the outside agent creating data

        ### obtain angle of common color phi_c
        # sa = State_analyzer(decoder_type='RNN_decoder')
        sa = State_analyzer()
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
        mean_bias = sub_error.copy()
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

        dispersion_sig.append(dispersion)
        density_sig.append(density)
        regular_sig.append(regular_term)
        exp_error_sig.append(mse_exp)
        mean_bias_sig.append(mean_bias)


    dispersion_all.append(dispersion_sig)
    density_all.append(density_sig)
    regular_all.append(regular_sig)
    exp_error_all.append(exp_error_sig)
    mean_bias_all.append(mean_bias_sig)

with open('../bin/figs/fig_data/dynamic_dispersion_{}.txt'.format(input_color),'wb') as fp:
    pickle.dump(dispersion_all,fp)
with open('../bin/figs/fig_data/color_density_{}.txt'.format(input_color),'wb') as fp:
    pickle.dump(density_all,fp)
with open('../bin/figs/fig_data/regularization_{}.txt'.format(input_color),'wb') as fp:
    pickle.dump(regular_all,fp)
with open('../bin/figs/fig_data/mean_bias_{}.txt'.format(input_color),'wb') as fp:
    pickle.dump(mean_bias_all,fp)
with open('../bin/figs/fig_data/experimental_error_{}.txt'.format(input_color),'wb') as fp:
    pickle.dump(exp_error_all,fp)
# '''

######### Plot dispersion
# '''
with open('../bin/figs/fig_data/dynamic_dispersion_40.txt','rb') as fp:
    dispersion_all = pickle.load(fp)

dispersion_unadapted = removeOutliers(np.array(dispersion_all[-1]))
unadapted_mean = np.mean(dispersion_unadapted)
unadapted_std = np.std(dispersion_unadapted)
unadapted_means = [unadapted_mean] * len(sigmas)
upper = [unadapted_mean + unadapted_std] * len(sigmas)
lower = [unadapted_mean - unadapted_std] * len(sigmas)

fig = plt.figure(figsize=(4,3.5))
bax = brokenaxes(xlims=((0, 35), (85, 95)), hspace=.05)

bax.boxplot(dispersion_all,showfliers=False,positions=sigmas,patch_artist = True,widths=1.5,boxprops=dict(facecolor='tab:blue', color='k'))
bax.set_ylabel('Dynamic Dispersion \n (angle degree$^2$)',fontsize=16)
bax.set_xlabel('$\sigma_s$', fontsize=16)
bax.axs[0].set_xticks([10,20,30])
bax.axs[0].set_xticklabels(['10.0','20.0','30.0'])
bax.axs[1].set_xticks([90])
bax.axs[1].set_xticklabels(['90.0'])

bax.legend(loc='lower right')
fig.tight_layout()
plt.savefig('../bin/figs/fig_collect/dynamic_dispersion_common.svg',format='svg',bbox_inches='tight')
plt.show()
'''

######## Plot color occupancy
# '''
with open('../bin/figs/fig_data/color_density_40.txt','rb') as fp:
    density_all = pickle.load(fp)

density_unadapted = removeOutliers(np.array(density_all[-1]))
unadapted_mean = np.mean(density_unadapted)
unadapted_std = np.std(density_unadapted)
unadapted_means = [unadapted_mean] * len(sigmas)
upper = [unadapted_mean + unadapted_std] * len(sigmas)
lower = [unadapted_mean - unadapted_std] * len(sigmas)

fig = plt.figure(figsize=(4,3.5))
bax = brokenaxes(xlims=((0, 35), (85, 95)), hspace=.05)
bax.boxplot(density_all,showfliers=False,positions=sigmas,patch_artist = True,widths=1.5,boxprops=dict(facecolor='tab:blue', color='k'))
bax.set_ylabel('Squared reciprocal of \n the Angular Occupation \n (color degree$^2$ / angle degree$^2$)',fontsize=15)
bax.set_xlabel(r'$\sigma_s$',fontsize=15)
bax.axs[0].set_xticks([10,20,30])
bax.axs[0].set_xticklabels(['10.0','20.0','30.0'])
bax.axs[1].set_xticks([90])
bax.axs[1].set_xticklabels(['90.0'])

bax.legend(loc='lower right')
fig.tight_layout()
plt.savefig('../bin/figs/fig_collect/color_density_common.svg',format='svg',bbox_inches='tight')
plt.show()
# '''

# Plot regularization term (mean bias correction)
# '''
with open('../bin/figs/fig_data/regularization_40.txt', 'rb') as fp:
    regular_all = np.array(pickle.load(fp))
regular_mean = list(np.mean(regular_all,axis=1))
regular_std = list(np.std(regular_all,axis=1) / math.sqrt(len(sigmas)))
regular_box = [list(regular_all[i]) for i in range(regular_all.shape[0])]

fig = plt.figure(figsize=(4,3.5))
bax = brokenaxes(xlims=((0, 35), (85, 95)), hspace=.05)
bxplt = bax.boxplot(regular_box,showfliers=False,positions=sigmas,patch_artist = True,widths=1.5, boxprops=dict(facecolor="tab:blue", color='k'))
bax.set_ylabel('Mean Bias Correction \n (color degree$^2$)',fontsize=13)
bax.set_xlabel(r'$\sigma_s$',fontsize=15)
bax.axs[0].set_xticks([10,20,30])
bax.axs[0].set_xticklabels(['10.0','20.0','30.0'])
bax.axs[1].set_xticks([90])
bax.axs[1].set_xticklabels(['90.0'])
fig.tight_layout()
plt.show()
# '''

# Compare theoritical and experimental results
# '''
with open('../bin/figs/fig_data/color_density_40.txt','rb') as fp:
    density_all = np.array(pickle.load(fp))
with open('../bin/figs/fig_data/dynamic_dispersion_40.txt','rb') as fp:
    dispersion_all = np.array(pickle.load(fp))
with open('../bin/figs/fig_data/experimental_error_40.txt', 'rb') as fp:
    exp_error_all = np.array(pickle.load(fp))
with open('../bin/figs/fig_data/regularization_40.txt', 'rb') as fp:
    regular_all = np.array(pickle.load(fp))
with open('../bin/figs/fig_data/mean_bias_40.txt', 'rb') as fp:
    mean_bias_all = np.array(pickle.load(fp))

exp_error_box = [np.sqrt(exp_error_all[i]) for i in range(exp_error_all.shape[0])]

theo_error = np.sqrt(density_all * dispersion_all + regular_all) # method 2
theo_error_box = [list(theo_error[i]) for i in range(theo_error.shape[0])]

no_decode_theo_error = np.sqrt(dispersion_all + mean_bias_all**2) # the result of setting d color / d angle = 1
no_decode_theo_error_box = [list(no_decode_theo_error[i]) for i in range(no_decode_theo_error.shape[0])]

fig = plt.figure(figsize=(4.5,4))
# ax = fig.add_subplot(111)
bax = brokenaxes(xlims=((0, 35), (85, 95)), hspace=.05)
error_mode, mean_mode = 'quantile', 'median'
error_bar_plot(sigmas, theo_error_box, fig=fig, ax=bax, color='tab:blue', label='Theory', error_mode=error_mode, mean_mode=mean_mode)
error_bar_plot(sigmas, no_decode_theo_error_box, fig=fig, ax=bax, color='tab:green', label='Theory \n Angular Occupation = 1', error_mode=error_mode, mean_mode=mean_mode)
error_bar_plot(sigmas, exp_error_box, fig=fig, ax=bax, color='tab:red', label='Experiment', error_mode=error_mode, mean_mode=mean_mode)

bax.set_ylabel('Memory Error \n (color degree)',fontsize=13)
bax.set_xlabel(r'$\sigma_s$',fontsize=15)
bax.axs[0].set_xticks([10,20,30])
bax.axs[0].set_xticklabels(['10.0','20.0','30.0'])
bax.axs[1].set_xticks([90])
bax.axs[1].set_xticklabels(['90.0'])
bax.legend(loc='upper right')
fig.tight_layout()
plt.show()
