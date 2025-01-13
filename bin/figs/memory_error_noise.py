# compute the the color dispersion
import context
import os
from core.color_error import Color_error
from core.color_input import Color_input
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from core.agent import Agent, Agent_group
from core.manifold.state_analyzer import State_analyzer
from brokenaxes import brokenaxes
import pickle
import math
from matplotlib.lines import Line2D

mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15

# Function to remove outliers from a dataset, based on interquartile range
def removeOutliers(a, outlierConstant=1):
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant # 1.5
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    return a[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))]


# Generate a batch of input colors sampled from different distributions
# 'common', 'uncommon', 'random', or a fixed custom color
def gen_input_colors(input_color,prior_sig=17.5,batch_size=5000):
    if input_color == 'common':
        input_color_set = 40 * np.ones(batch_size)
    elif input_color == 'uncommon':
        input_color_set = 85 * np.ones(batch_size)
    elif input_color == 'random':  # randomly sampled from prior distribution
        color_sampling = Color_input()
        color_sampling.add_samples()
        color_sampling.prob(method='vonmises', bias_centers=[40, 130, 220, 310], sig=prior_sig)
        input_color_set = color_sampling.out_color_degree(batch_size=batch_size)
    else:
        input_color_set =input_color * np.ones(batch_size)  # any customized color
    return input_color_set

# Define noise levels used in the experiment
Noises = ['0.10', '0.12', '0.14', '0.16', '0.18', '0.20', '0.22', '0.24', '0.26', '0.28', '0.30', '0.32']
Noise_values = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32]

######## Uncomment the following section to calculate experimental memory error
'''
exp_error_all = []
prod_int = 800  # duration of the delay
batch_size = 5000  # Number of trials for each RNN
prior_sig = 17.5  # Width of the piror
sigma_rec = None; sigma_x = None # Set the noise to be default (training value)

rule_name = 'color_reproduction_delay_unit'
sub_dir = 'noise_delta/'

# Iterate through noise levels
for i,noise in enumerate(Noises):
    model_dir_parent = "../core/model_noise/noise_"+noise+"/model_"+str(prior_sig)+"/color_reproduction_delay_unit/"
    exp_error_n= []

    # Iterate through 50 RNNs for each noise level
    for m in range(50):
        print(noise, m)
        model_dir = 'model_{}/'.format(m)  # example RNN
        f = os.path.join(model_dir_parent, model_dir, sub_dir)
        sub = Agent(f, rule_name)  # this is the outside agent creating data


        # Generate random input colors sampled from the environmental prior
        input_colors = gen_input_colors('random')
        
        # Compute experimental memory error by comparing target and reported colors
        sub.do_exp(prod_intervals=prod_int, sigma_rec=sigma_rec, sigma_x=sigma_x, ring_centers=input_colors)
        color_error = Color_error()
        color_error.add_data(sub.behaviour['report_color'], sub.behaviour['target_color'])
        sub_error = color_error.calculate_error()  # Xircular substraction
        sqe_exp = sub_error ** 2  
        sqe_exp = removeOutliers(sqe_exp)  # Remove outlier errors
        mse_exp = np.mean(sqe_exp)
        print('exp: ', math.sqrt(mse_exp),end='\n\n')

        exp_error_n.append(mse_exp)

    exp_error_all.append(exp_error_n)
with open('../bin/figs/fig_data/all_memory_error_noise.txt','wb') as fp:
    pickle.dump(exp_error_all,fp)
# '''


# Compare theoretical and experimental results (side-by-side boxes)
# '''
# Load previously calculated memory errors for visualization
with open('../bin/figs/fig_data/all_memory_error_noise.txt', 'rb') as fp:
    exp_error_all = np.array(pickle.load(fp))[:-1]

# Compute mean and standard deviation of memory errors across RNNs
exp_error_mean = np.mean(np.sqrt(exp_error_all), axis=1)
exp_error_std = np.std(np.sqrt(exp_error_all), axis=1)

fig, ax = plt.subplots(figsize=(3.2,3))
ax.errorbar(x=Noise_values[:-1], y=exp_error_mean,yerr=exp_error_std,fmt='r.-',linewidth=2.0, markersize=10,alpha=1, label='Experiment')

ax.set_ylabel('Memory Error\n of random input color \n(color degree)',fontsize=15)
ax.set_xlabel('Noise',fontsize=15)
ax.set_xticks([0.1,0.2,0.3])
ax.set_xticklabels(['0.1','0.2','0.3'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.legend(frameon=False,fontsize=12)
plt.savefig('../bin/figs/fig_collect/memory_error_all_noise.svg',format='svg',bbox_inches='tight')
plt.show()
# '''




