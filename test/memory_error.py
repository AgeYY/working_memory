import context
import os
from core.color_error import Color_error
from core.color_input import Color_input
import numpy as np
import matplotlib.pyplot as plt
from core.agent import Agent, Agent_group

def removeOutliers(a, outlierConstant=1.5):
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant # 1.5
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    return a[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))]

############ compute the average error (averaged on prior)

def memory_error(prod_int, prior_sig, batch_size, color_sampler):
    rule_name = 'color_reproduction_delay_unit'
    model_dir_parent = "../core/model/model_" + str(prior_sig) + "/color_reproduction_delay_unit/"
    sub_dir = '/noise_delta'

    std_sub_list = []
    for filename in os.listdir(model_dir_parent):
        f = os.path.join(model_dir_parent, filename + sub_dir)
        sub = Agent(f, rule_name)

        input_color_set = color_sampler.out_color_degree(batch_size)
        #plt.hist(input_color_set, bins=40)
        #plt.show()

        sub.do_exp(prod_intervals=prod_int, sigma_rec=sigma_rec, sigma_x=sigma_x, ring_centers=input_color_set)

        color_error = Color_error()
        color_error.add_data(sub.behaviour['report_color'], sub.behaviour['target_color'])
        sub_error = color_error.calculate_error() # circular substraction
        mse_error = sub_error**2
        mse_error = removeOutliers(mse_error)
        mse_sub = np.mean(mse_error)

        std_sub_list.append(np.sqrt(mse_sub))
    return std_sub_list

#################### Main
prior_sig_bias = 15.0
prior_sig_uniform = 90.0
prod_int = 800 # duration of the delay
batch_size = 5000 # number of sampled trials
sigma_rec = None; sigma_x = None # set the noise to be default (training value)
#################### Parameters

color_sampler = Color_input()
color_sampler.add_samples(n_degree=360) # x_mesh for the prior function. Bad name
color_sampler.prob(bias_centers=[40, 130, 220, 310], method='vonmises', sig=prior_sig_bias) # create prior

std_sub_list_bias = memory_error(prod_int, prior_sig_bias, batch_size, color_sampler)
std_sub_list_uniform = memory_error(prod_int, prior_sig_uniform, batch_size, color_sampler)

mean_std_bias = np.mean(std_sub_list_bias)
mean_std_uniform = np.mean(std_sub_list_uniform)
print('memory error (std) of on average of all RNNs and all inputs with prior = {} is {}'.format(prior_sig_bias, mean_std_bias))
print('memory error (std) of on average of all RNNs and all inputs with prior = {} is {}'.format(prior_sig_uniform, mean_std_uniform))

from core.ploter import plot_layer_boxplot_helper
score = {str(prior_sig_bias): std_sub_list_bias, str(prior_sig_uniform): std_sub_list_uniform}
layer_order = {str(prior_sig_bias): 0, str(prior_sig_uniform): 1}

fig, ax = plot_layer_boxplot_helper(score, layer_order)
plt.show()

