# compute the variance of angle and color in one example trial
import context
import os
from core.color_error import Color_error
from core.color_input import Color_input
import numpy as np
import matplotlib.pyplot as plt
from core.agent import Agent, Agent_group
from core.manifold.state_analyzer import State_analyzer
import pickle
import math
from core.tools import find_nearest, mean_se

def removeOutliers(a, outlierConstant=1.5):
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant # 1.5
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    return a[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))]

def color_mse(input_color,f,prior_sig,batch_size=5000, sigma_rec=None, sigma_x=None):

    if input_color == 'common':
        input_color_set = 40 * np.ones(batch_size)
    elif input_color == 'uncommon':
        input_color_set = 85 * np.ones(batch_size)
    elif input_color == 'random':
        color_sampling = Color_input()
        color_sampling.add_samples()
        color_sampling.prob(method='vonmises', bias_centers=[40, 130, 220, 310], sig=prior_sig)
        input_color_set = color_sampling.out_color_degree(batch_size=batch_size)
    else:
        input_color_set = input_color * np.ones(batch_size)

    prod_int = 800 # duration of the delay
    rule_name = 'color_reproduction_delay_unit'
    sub = Agent(f, rule_name)

    sub.do_exp(prod_intervals=prod_int, sigma_rec=sigma_rec, sigma_x=sigma_x, ring_centers=input_color_set)
    # Check for NaN values in report and target colors

    color_error = Color_error()
    color_error.add_data(sub.behaviour['report_color'], sub.behaviour['target_color'])
    sub_error = color_error.calculate_error() # circular substraction
    sub_error_sq = removeOutliers(sub_error ** 2)
    mse_sub = np.mean(sub_error_sq)

    return mse_sub


sigma_rec = None
sigma_x = None
n_model = 50
input_color_range = range(85,176,3)

################# MSE of multiple input colors for biased and uniform prior models
# '''
# Biased prior model
mse_color_biased = []
model_dir_parent = "../core/model/model_" + str(12.5) + "/color_reproduction_delay_unit/"
sub_dir = 'noise_delta/'
for i in range(n_model):
    model_dir = 'model_{}/'.format(i)  # example RNN
    file = os.path.join(model_dir_parent, model_dir, sub_dir)
    mse_color_biased_model = []
    for input_color in input_color_range:
        print('Biased Model {m}, color {d}'.format(m=i,d=input_color))
        mse_sub = color_mse(input_color, f=file,prior_sig=12.5, sigma_rec=sigma_rec, sigma_x=sigma_x)
        mse_color_biased_model.append(mse_sub)
    mse_color_biased.append(mse_color_biased_model)
with open('../bin/figs/fig_data/mse_color_12.5.txt','wb') as fp:
    pickle.dump(mse_color_biased,fp)

# Uniform prior
mse_color_uniform = []
rule_name = 'color_reproduction_delay_unit'
model_dir_parent = "../core/model/model_" + str(90.0) + "/color_reproduction_delay_unit/"
sub_dir = 'noise_delta/'
for i in range(n_model):
    model_dir = 'model_{}/'.format(i)  # example RNN
    file = os.path.join(model_dir_parent, model_dir, sub_dir)
    mse_color_uniform_model = []
    for input_color in input_color_range:
        print('Uniform Model {m}, color {d}'.format(m=i,d=input_color))
        mse_sub = color_mse(input_color, f=file,prior_sig=90.0, sigma_rec=sigma_rec, sigma_x=sigma_x)
        mse_color_uniform_model.append(mse_sub)
    mse_color_uniform.append(mse_color_uniform_model)
with open('../bin/figs/fig_data/mse_color_90.0.txt','wb') as fp:
    pickle.dump(mse_color_uniform,fp)

# '''



################# Plot the figure
# Load data
with open('../bin/figs/fig_data/mse_color_12.5.txt', 'rb') as fp:
    mse_color_biased=np.array(pickle.load(fp))
with open('../bin/figs/fig_data/mse_color_90.0.txt', 'rb') as fp:
    mse_color_uniform=np.array(pickle.load(fp))

# memory error
memory_error_biased = np.sqrt(mse_color_biased)
memory_error_uniform = np.sqrt(mse_color_uniform)
# Colors for plot
colors = input_color_range

## MSE
#mean_mse_biased = np.mean(mse_color_biased,axis=0)
#se_mse_biased = np.std(mse_color_biased,axis=0) / math.sqrt(len(colors))
#
#mean_mse_uniform = np.mean(mse_color_uniform,axis=0)
#se_mse_uniform = np.std(mse_color_uniform,axis=0) / math.sqrt(len(colors))

## memory error
mean_error_biased = np.mean(memory_error_biased,axis=0)
se_error_biased = np.std(memory_error_biased,axis=0) / math.sqrt(len(colors))

mean_error_uniform = np.mean(memory_error_uniform,axis=0)
se_error_uniform = np.std(memory_error_uniform,axis=0) / math.sqrt(len(colors))

# Prior distribution
color_sampling = Color_input()
color_sampling.add_samples()

color_sampling.prob(method='vonmises', bias_centers=[40, 130, 220, 310], sig=12.5)
rand_colors_biased = color_sampling.out_color_degree(batch_size=20000)

color_sampling.prob(method='vonmises', bias_centers=[40, 130, 220, 310], sig=90.0)
rand_colors_uniform = color_sampling.out_color_degree(batch_size=20000)


fig,ax = plt.subplots(2,1,sharex=True,figsize=(5,7))


ax[0].hist(rand_colors_uniform, bins=list(np.arange(0, 360, 4)),density=True,label='Uniform',alpha=0.9,color='b')
ax[0].hist(rand_colors_biased, bins=list(np.arange(0, 360, 4)),density=True,label='Biased',alpha=0.9,color='r')
ax[0].axvline(130,color='k',linestyle='--',label='Common color')
ax[0].set_xlim((85,175))
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
# ax[0].set_xlabel('Color',fontsize=15)
ax[0].set_ylabel('Prior Probability Density',fontsize=15)
ax[0].legend()


ax[1].plot(colors, mean_error_uniform, 'b.-',label='Uniform')
ax[1].fill_between(colors, mean_error_uniform-se_error_uniform, mean_error_uniform+se_error_uniform,alpha=0.5,color='b')
ax[1].plot(colors, mean_error_biased, 'r.-',label='Biased')
ax[1].fill_between(colors, mean_error_biased-se_error_biased, mean_error_biased+se_error_biased,alpha=0.5,color='r')


ax[1].axvline(130,color='k',linestyle='--')
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].set_xlabel('Color',fontsize=15)
ax[1].set_ylabel('Memory Error \n for a Fixed Input Color',fontsize=15)
plt.legend()
plt.tight_layout()
plt.savefig('../bin/figs/fig_collect/color_mse_example.svg',format='svg')
plt.show()







