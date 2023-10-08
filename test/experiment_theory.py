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
from brokenaxes import brokenaxes
import pickle
import math
from matplotlib.lines import Line2D


model_names = ['3.0','10.0','12.5','15.0','17.5','20.0', '22.5','25.0','27.5', '30.0','90.0']
sigmas = [3.0, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 90.0]

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
# model_dir = 'model_1/' # example RNN
sub_dir = 'noise_delta/'

for i,prior_sig in enumerate(sigmas):
    model_dir_parent = "../core/model/model_" + str(prior_sig) + "/color_reproduction_delay_unit/"
    dispersion_sig = []
    density_sig = []
    regular_sig = []
    exp_error_sig = []

    for m in range(50):
        print(prior_sig,m)
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

        dispersion_sig.append(dispersion)
        density_sig.append(density)
        regular_sig.append(regular_term)
        exp_error_sig.append(mse_exp)


    dispersion_all.append(dispersion_sig)
    density_all.append(density_sig)
    regular_all.append(regular_sig)
    exp_error_all.append(exp_error_sig)

with open('../bin/figs/fig_data/dynamic_dispersion_{}.txt'.format(input_color),'wb') as fp:
    pickle.dump(dispersion_all,fp)
with open('../bin/figs/fig_data/color_density_{}.txt'.format(input_color),'wb') as fp:
    pickle.dump(density_all,fp)
with open('../bin/figs/fig_data/regularization_{}.txt'.format(input_color),'wb') as fp:
    pickle.dump(regular_all,fp)
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

bax.plot(sigmas,unadapted_means,'k',linestyle='--',alpha=0.5)
bax.fill_between(sigmas, lower, upper, color='grey',alpha=0.2)
bax.boxplot(dispersion_all,showfliers=False,positions=sigmas,patch_artist = True,widths=1.5,boxprops=dict(facecolor='lightblue', color='blue'))
bax.set_ylabel('Dynamic Dispersion',fontsize=13)
bax.set_xlabel(r'$\sigma_s$',fontsize=15)
bax.axs[0].set_xticks([10,20,30])
bax.axs[0].set_xticklabels(['10.0','20.0','30.0'])
bax.axs[1].set_xticks([90])
bax.axs[1].set_xticklabels(['90.0'])

bax.plot([],[],color='lightblue',linewidth=8,label='Adapted')
bax.plot([],[],color='lightgrey',linewidth=8,label='Unadapted')
bax.legend(loc='lower right')
plt.savefig('../bin/figs/fig_collect/dynamic_dispersion_common.svg',format='svg',bbox_inches='tight')
plt.show()
# '''

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
bax.plot(sigmas,unadapted_means,'k',linestyle='--',alpha=0.5)
bax.fill_between(sigmas, lower, upper, color='grey',alpha=0.2)
bax.boxplot(density_all,showfliers=False,positions=sigmas,patch_artist = True,widths=1.5,boxprops=dict(facecolor='lightblue', color='blue'))
bax.set_ylabel('Color density',fontsize=13)
bax.set_xlabel(r'$\sigma_s$',fontsize=15)
bax.axs[0].set_xticks([10,20,30])
bax.axs[0].set_xticklabels(['10.0','20.0','30.0'])
bax.axs[1].set_xticks([90])
bax.axs[1].set_xticklabels(['90.0'])

bax.plot([],[],color='lightblue',linewidth=8,label='Adapted')
bax.plot([],[],color='lightgrey',linewidth=8,label='Unadapted')
bax.legend(loc='lower right')
plt.savefig('../bin/figs/fig_collect/color_density_common.svg',format='svg',bbox_inches='tight')
plt.show()
# '''


# Plot regularization term
# '''
with open('../bin/figs/fig_data/regularization_40.txt', 'rb') as fp:
    regular_all = np.sqrt(np.array(pickle.load(fp)))

regular_mean = list(np.mean(regular_all,axis=1))
regular_std = list(np.std(regular_all,axis=1) / math.sqrt(len(sigmas)))
regular_box = [list(regular_all[i]) for i in range(regular_all.shape[0])]

fig = plt.figure(figsize=(4,3.5))
bax = brokenaxes(xlims=((0, 35), (85, 95)), hspace=.05)
bxplt = bax.boxplot(regular_box,showfliers=False,positions=sigmas,patch_artist = True,widths=1.5)
bax.set_ylabel('',fontsize=13)
bax.set_xlabel(r'$\sigma_s$',fontsize=15)
bax.axs[0].set_xticks([10,20,30])
bax.axs[0].set_xticklabels(['10.0','20.0','30.0'])
bax.axs[1].set_xticks([90])
bax.axs[1].set_xticklabels(['90.0'])
plt.show()
# '''


# plot experimental results
# '''
with open('../bin/figs/fig_data/experimental_error_40.txt', 'rb') as fp:
    exp_error_common = np.array(pickle.load(fp))
exp_error_box_common = [np.sqrt(exp_error_common[i]) for i in range(exp_error_common.shape[0])]

unadapted_mean = np.mean(np.sqrt(exp_error_common[-1]))
# unadapted_ste = np.std(np.sqrt(exp_error_common[-1])) / math.sqrt(len(exp_error_common[-1]))
unadapted_std = np.std(np.sqrt(exp_error_common[-1]))
unadapted_means = [unadapted_mean] * len(sigmas)
upper = [unadapted_mean + unadapted_std] * len(sigmas)
lower = [unadapted_mean - unadapted_std] * len(sigmas)

fig = plt.figure(figsize=(4,3.5))
bax = brokenaxes(xlims=((0, 35), (85, 95)), hspace=.05)
bax.plot(sigmas,unadapted_means,'k',linestyle='--',alpha=0.5)
bax.fill_between(sigmas, lower, upper, color='grey',alpha=0.2)
bxplt1 = bax.boxplot(exp_error_box_common,showfliers=False,positions=sigmas,patch_artist = True,widths=1.5,boxprops=dict(facecolor='lightblue', color='blue'),medianprops = dict(color = "blue", linewidth = 1.5))

bax.set_ylabel('Memory Error',fontsize=13)
bax.set_xlabel(r'$\sigma_s$',fontsize=15)
bax.axs[0].set_xticks([10,20,30])
bax.axs[0].set_xticklabels(['10.0','20.0','30.0'])
bax.axs[1].set_xticks([90])
bax.axs[1].set_xticklabels(['90.0'])

bax.plot([],[],color='lightblue',linewidth=8,label='Adapted')
bax.plot([],[],color='lightgrey',linewidth=8,label='Unadapted')
bax.legend(loc='lower right')
plt.savefig('../bin/figs/fig_collect/experimental_error.svg',format='svg',bbox_inches='tight')
plt.show()
# '''


# Compare theoritical and experimental results (side-by-side boxes)
# '''
with open('../bin/figs/fig_data/color_density_40.txt','rb') as fp:
    density_all = np.array(pickle.load(fp))
with open('../bin/figs/fig_data/dynamic_dispersion_40.txt','rb') as fp:
    dispersion_all = np.array(pickle.load(fp))
with open('../bin/figs/fig_data/experimental_error_40.txt', 'rb') as fp:
    exp_error_all = np.array(pickle.load(fp))
with open('../bin/figs/fig_data/regularization_40.txt', 'rb') as fp:
    regular_all = np.array(pickle.load(fp))



exp_error_box = [np.sqrt(exp_error_all[i]) for i in range(exp_error_all.shape[0])]

# theo_error = np.sqrt(density_all*dispersion_all) + regular_all # method 1
theo_error = np.sqrt(density_all * dispersion_all + regular_all) # method 2
theo_error_box = [list(theo_error[i]) for i in range(theo_error.shape[0])]

unadapted_theo_mean = np.mean(removeOutliers(theo_error[-1]))
unadapted_theo_std = np.std(removeOutliers(theo_error[-1]))
unadapted_theo_means = [unadapted_theo_mean] * len(sigmas)
unadapted_upper = [unadapted_theo_mean + unadapted_theo_std] * len(sigmas)
unadapted_lower = [unadapted_theo_mean - unadapted_theo_std] * len(sigmas)


fig = plt.figure(figsize=(6,3.5))
bax = brokenaxes(xlims=((0, 35), (85, 95)), hspace=.05)
bax.plot(sigmas,unadapted_theo_means,'k',linestyle='--',alpha=0.5)
bax.fill_between(sigmas, unadapted_lower, unadapted_upper, color='grey',alpha=0.2,label='Theory unadapted')

positions_1 = [x+0.5 for x in sigmas]
positions_2 = [x-0.5 for x in sigmas]

bxplt1 = bax.boxplot(theo_error_box,showfliers=False,positions=positions_1,patch_artist = True,widths=1,boxprops=dict(facecolor='lightblue', color='darkblue'),medianprops = dict(color = "darkblue", linewidth = 1.5))
bxplt2 = bax.boxplot(exp_error_box,showfliers=False,positions=positions_2,patch_artist = True,widths=1,boxprops=dict(facecolor='salmon', color='darkred'),medianprops = dict(color = "darkred", linewidth = 1.5))
bax.set_ylabel('Memory Error',fontsize=13)
bax.set_xlabel(r'$\sigma_s$',fontsize=15)
bax.axs[0].set_xticks([10,20,30])
bax.axs[0].set_xticklabels(['10.0','20.0','30.0'])
bax.axs[1].set_xticks([90])
bax.axs[1].set_xticklabels(['90.0'])
bax.plot([],[],color='lightblue',linewidth=8,label='Theory adapted')
bax.plot([],[],color='salmon',linewidth=8,label='Experiment')
bax.legend(loc='upper left')
plt.savefig('../bin/figs/fig_collect/exp_theo_comparison.svg',format='svg',bbox_inches='tight')
plt.show()
# '''


# Compare theoritical and experimental results (box and dot line)
'''
with open('../bin/figs/fig_data/color_density_40.txt','rb') as fp:
    density_all = np.array(pickle.load(fp))
with open('../bin/figs/fig_data/dynamic_dispersion_40.txt','rb') as fp:
    dispersion_all = np.array(pickle.load(fp))
with open('../bin/figs/fig_data/experimental_error_40.txt', 'rb') as fp:
    exp_error_all = np.array(pickle.load(fp))
with open('../bin/figs/fig_data/regularization_40.txt', 'rb') as fp:
    regular_all = np.array(pickle.load(fp))

# theo_error = np.sqrt(density_all*dispersion_all) + regular_all # method 1
theo_error = np.sqrt(density_all * dispersion_all + regular_all) # method 2
theo_error_box = [theo_error[i] for i in range(theo_error.shape[0])]

unadapted_theo_mean = np.mean(removeOutliers(theo_error[-1]))
unadapted_theo_std = np.std(removeOutliers(theo_error[-1]))
unadapted_theo_means = [unadapted_theo_mean] * len(sigmas)
unadapted_upper = [unadapted_theo_mean + unadapted_theo_std] * len(sigmas)
unadapted_lower = [unadapted_theo_mean - unadapted_theo_std] * len(sigmas)

exp_error_mean = [np.mean(removeOutliers(np.sqrt(exp_error_all[i]))) for i in range(exp_error_all.shape[0])]
exp_error_std = [np.std(removeOutliers(np.sqrt(exp_error_all[i]))) for i in range(exp_error_all.shape[0])]

fig = plt.figure(figsize=(5.5,4.5))
bax = brokenaxes(xlims=((0, 35), (85, 95)), hspace=.05)

bax.plot(sigmas,unadapted_theo_means,'k',linestyle='--',alpha=0.5)
bax.fill_between(sigmas, unadapted_lower, unadapted_upper, color='grey',alpha=0.2,label='Theory unadapted')

bxplt = bax.boxplot(theo_error_box,showfliers=False,positions=sigmas,patch_artist = True,widths=1.5,boxprops=dict(facecolor='lightblue', color='blue'),medianprops = dict(color = "blue", linewidth = 1.5))
bax.errorbar(x=sigmas, y=exp_error_mean,yerr=exp_error_std,fmt='r.-',linewidth=1.5, markersize=12,label='Experiment',alpha=0.5)
bax.set_ylabel('Memory Error',fontsize=13)
bax.set_xlabel(r'$\sigma_s$',fontsize=15)
bax.axs[0].set_xticks([10,20,30])
bax.axs[0].set_xticklabels(['10.0','20.0','30.0'])
bax.axs[1].set_xticks([90])
bax.axs[1].set_xticklabels(['90.0'])
bax.plot([],[],color='lightblue',linewidth=8,label='Theory adapted')
bax.legend(loc='lower right')
plt.savefig('../bin/figs/fig_collect/exp_theo_comparison2.svg',format='svg',bbox_inches='tight')
plt.show()
# '''


