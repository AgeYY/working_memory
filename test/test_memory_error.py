import context
import os
from core.color_error import Color_error
from core.color_input import Color_input
import numpy as np
import matplotlib.pyplot as plt
from core.agent import Agent, Agent_group


prod_int = 800 # duration of the delay
input_color = 85 # the input will be fixed to 40 degree (common color) or 85 degree (uncommon color)
prior_sig = 15.0 # width of the piror
batch_size = 20 # repeat 40 degree for batch_size trials
sigma_rec = None; sigma_x = None # set the noise to be default (training value)

rule_name = 'color_reproduction_delay_unit'
model_dir_parent = "../core/model/model_" + str(prior_sig) + "/color_reproduction_delay_unit/"
sub_dir = '/noise_delta'

########### compute the error for common color (or uncommon color)
mse_sub_list = []
for filename in os.listdir(model_dir_parent):
    f = os.path.join(model_dir_parent, filename + sub_dir)
    sub = Agent(f, rule_name)
    input_color_set = np.ones(batch_size) * input_color

    sub.do_exp(prod_intervals=prod_int, sigma_rec=sigma_rec, sigma_x=sigma_x, ring_centers=input_color_set)

    color_error = Color_error()
    color_error.add_data(sub.behaviour['report_color'], sub.behaviour['target_color'])
    sub_error = color_error.calculate_error() # circular substraction

    mse_sub = np.linalg.norm(sub_error)**2 / len(sub_error)
    mse_sub_list.append(mse_sub)

mean_mse = np.mean(mse_sub_list)
print('memory error (mse) of all RNNs with prior = {} at input color = {} is {}'.format(prior_sig, input_color, mean_mse))


############ compute the average error (averaged on prior)
batch_size = 20 # number of colors sampled from the prior
color_sampler = Color_input()
color_sampler.add_samples(n_degree=360) # x_mesh for the prior function. Bad name
color_sampler.prob(bias_centers=[40, 130, 220, 310], method='vonmises', sig=prior_sig) # create prior

mse_sub_list = []
for filename in os.listdir(model_dir_parent):
    f = os.path.join(model_dir_parent, filename + sub_dir)
    sub = Agent(f, rule_name)

    input_color_set = color_sampler.out_color_degree(batch_size)

    sub.do_exp(prod_intervals=prod_int, sigma_rec=sigma_rec, sigma_x=sigma_x, ring_centers=input_color_set)

    color_error = Color_error()
    color_error.add_data(sub.behaviour['report_color'], sub.behaviour['target_color'])
    sub_error = color_error.calculate_error() # circular substraction

    mse_sub = np.linalg.norm(sub_error)**2 / len(sub_error)
    mse_sub_list.append(mse_sub)

mean_mse = np.mean(mse_sub_list)
print('memory error (mse) of all RNNs with prior = {} at on average of all input is {}'.format(prior_sig, mean_mse))
