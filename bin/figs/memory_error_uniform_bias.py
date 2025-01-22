# compute the variance of angle and color in one example trial
import context
import os
from core.color_error import Color_error
from core.color_input import Color_input
import numpy as np
import matplotlib.pyplot as plt
from core.agent import Agent, Agent_group
from core.manifold.state_analyzer import State_analyzer
#from brokenaxes import brokenaxes
import pickle
import math
from core.tools import find_nearest, mean_se
from scipy.stats import mannwhitneyu, ttest_rel
import pandas as pd
import seaborn as sns
from core.ploter import plot_layer_boxplot_helper


# Function to remove outliers from data
def removeOutliers(a, outlierConstant=1.5):
    """
    Remove outliers from an array using the interquartile range (IQR) method.

    Parameters:
        a (array-like): Input array from which outliers will be removed.
        outlierConstant (float): Multiplier for the IQR to define outlier thresholds.

    Returns:
        array-like: Filtered array with outliers removed.
    """
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant # 1.5
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    return a[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))[0]]


# Function to calculate memory error (mean squared error) for a prior distribution
def color_mse(input_color,f,prior_sig,batch_size=5000):
    """
       Calculate the memory error (MSE) between input and output colors for a specific input distribution.

       Parameters:
           input_color (str or float): Type of input color ('common', 'uncommon', 'random') or specific value.
           f (str): File path to the RNN model.
           prior_sig (float): Standard deviation of the prior distribution.
           batch_size (int): Number of trials to generate.

       Returns:
           float: Mean squared error of the memory error.
    """

    if input_color == 'common':
        input_color_set = 40 * np.ones(batch_size)  # Common color 40
    elif input_color == 'uncommon':
        input_color_set = 85 * np.ones(batch_size)  # Uncommon color 85
    elif input_color == 'random':
        color_sampling = Color_input()   # Generate random input colors.
        color_sampling.add_samples()
        color_sampling.prob(method='vonmises', bias_centers=[40, 130, 220, 310], sig=prior_sig)
        input_color_set = color_sampling.out_color_degree(batch_size=batch_size)
    else:
        input_color_set = input_color * np.ones(batch_size)  # A customized input color

    prod_int = 800 # duration of the delay
    rule_name = 'color_reproduction_delay_unit'
    sub = Agent(f, rule_name)  # Initialize the RNN model.
    sigma_rec = None;sigma_x = None  # Use default noise values.

    # Perform the experiment
    sub.do_exp(prod_intervals=prod_int, sigma_rec=sigma_rec, sigma_x=sigma_x, ring_centers=input_color_set)

    # Calculate memory error (MSE)
    color_error = Color_error()
    color_error.add_data(sub.behaviour['report_color'], sub.behaviour['target_color'])
    sub_error = color_error.calculate_error()  # Circular substraction
    sub_error_sq = removeOutliers(sub_error ** 2)  # Remove outliers before calculating MSE.
    mse_sub = np.mean(sub_error_sq)
    return mse_sub


# Calculate memory errors for Biased and Uniform RNNs across various prior distributions
# '''
sigmas = [3.0, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 90.0]  # Prior distribution sigmas.
model_names = [str(s) for s in sigmas]

error_adapted = []  # Errors for Biased (adapted) RNNs.
error_unadapted = []   # Errors for Uniform (unadapted) RNNs.

sub_dir = 'noise_delta/'
unadapted_model_dir_parent = "../core/model/model_90.0/color_reproduction_delay_unit/"
for prior_sig in sigmas:
    adapted_model_dir_parent = "../core/model/model_" + str(prior_sig) + "/color_reproduction_delay_unit/"
    error_adapted_sig = []
    error_unadapted_sig = []
    for j in range(50): # Iterate over 50 RNNs
        print(prior_sig,j)
        model_dir = 'model_{}/'.format(j)  # example RNN
        adapted_file = os.path.join(adapted_model_dir_parent, model_dir, sub_dir)
        unadapted_file = os.path.join(unadapted_model_dir_parent, model_dir, sub_dir)

        # Calculate memory errors for common input colors
        # adapted_mse = color_mse('random', f=adapted_file, prior_sig=prior_sig)
        # unadapted_mse = color_mse('random', f=unadapted_file, prior_sig=prior_sig)
        adapted_mse = color_mse('common', f=adapted_file, prior_sig=prior_sig)
        unadapted_mse = color_mse('common', f=unadapted_file, prior_sig=prior_sig)

        error_adapted_sig.append(adapted_mse)
        error_unadapted_sig.append(unadapted_mse)
    error_adapted.append(error_adapted_sig)
    error_unadapted.append(error_unadapted_sig)

with open('../bin/figs/fig_data/mse_adapted.txt','wb') as fp:
    pickle.dump(error_adapted,fp)
with open('../bin/figs/fig_data/mse_unadapted.txt','wb') as fp:
    pickle.dump(error_unadapted,fp)
'''

################# Plot the figure for example sigma
# '''
# Load calculated memory errors for plotting
with open('../bin/figs/fig_data/mse_adapted.txt', 'rb') as fp:
    error_adapted = np.array(pickle.load(fp))
with open('../bin/figs/fig_data/mse_unadapted.txt', 'rb') as fp:
    error_unadapted = np.array(pickle.load(fp))

e_adapted = list(np.sqrt(error_adapted[2])) # sigma = 12.5 (third one)
e_unadapted = list(np.sqrt(error_unadapted[2])) # sigma = 12.5

# # Perform statistical test (Mann-Whitney U-test)
U1, pvalue = mannwhitneyu(e_adapted, e_unadapted, method="exact")
print(U1,pvalue)

# Paired T-test
# T, pvalue = ttest_rel(e_adapted,e_unadapted)
# print(T, pvalue)

formatted_pvalues = f'p={pvalue:.2e}'

#################### Plot data
# Remove outliers for plotting
e_unadapted, e_adapted = removeOutliers(np.array(e_unadapted)), removeOutliers(np.array(e_adapted))

data = {'Uniform RNN': e_unadapted, 'Biased RNN': e_adapted}
layer_order = {key: i for i, key in enumerate(data)}
jitter_color_order = {'Biased RNN': '#d62728', 'Uniform RNN': '#1f77b4'}

fig, ax = plot_layer_boxplot_helper(data, layer_order, jitter_color=jitter_color_order)

plt.savefig('../bin/figs/fig_collect/Figure_1D.svg',format='svg',bbox_inches='tight')

plt.show()

# '''


################# Plot the figure for all sigma
'''
# prior sigmas
sigmas = [3.0, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 90.0]

# Load data
with open('../bin/figs/fig_data/mse_adapted.txt', 'rb') as fp:
    error_adapted = np.array(pickle.load(fp))
with open('../bin/figs/fig_data/mse_unadapted.txt', 'rb') as fp:
    error_unadapted = np.array(pickle.load(fp))
error_box_adapted = [np.sqrt(error_adapted[i]) for i in range(error_adapted.shape[0])]
error_box_unadapted = [np.sqrt(error_unadapted[i]) for i in range(error_unadapted.shape[0])]

fig = plt.figure(figsize=(6,3.5))
bax = brokenaxes(xlims=((0, 35), (85, 95)), hspace=.05)

positions_1 = [x+0.5 for x in sigmas]
positions_2 = [x-0.5 for x in sigmas]

bxplt1 = bax.boxplot(error_box_adapted,showfliers=False,positions=positions_1,patch_artist = True,widths=1,boxprops=dict(facecolor='lightblue', color='darkblue'),medianprops = dict(color = "darkblue", linewidth = 1.5))
bxplt2 = bax.boxplot(error_box_unadapted,showfliers=False,positions=positions_2,patch_artist = True,widths=1,boxprops=dict(facecolor='lightgrey', color='darkgrey'),medianprops = dict(color = "darkgrey", linewidth = 1.5))
bax.set_ylabel('Memory Error',fontsize=13)
bax.set_xlabel(r'$\sigma_s$',fontsize=15)
bax.axs[0].set_xticks([10,20,30])
bax.axs[0].set_xticklabels(['10.0','20.0','30.0'])
bax.axs[1].set_xticks([90])
bax.axs[1].set_xticklabels(['90.0'])
bax.plot([],[],color='lightblue',linewidth=8,label='Adapted')
bax.plot([],[],color='lightgrey',linewidth=8,label='Unadapted')
bax.legend(loc='lower right')
plt.savefig('../bin/figs/fig_collect/Figure_1G.svg',format='svg',bbox_inches='tight')
plt.show()
# '''









