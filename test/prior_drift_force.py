# From prior distribution of information into drift force according to the theory of Zachary 2018
import context

import numpy as np
import matplotlib.pyplot as plt
from core.bay_drift_loss_torch import prior_func # prior function consists of four vonmises functions.
import core.tools as tools

def drift_force(x, p, sigma):
    '''
    x (array): equally spaced mesh points, e.g. [-pi, -pi + 0.1, ..., pi]. Unit is rad
    p (array): in the same shape of x, prior probability distribution density function.
    sigma (float): ddf should be written as d\theta = - du/d\theta dt + \sigma d\epsilon(t)
    output:
      force = - du/d\theta = \sigma^2 / 2 * p'(\theta) / p(\theta)
    '''
    delta = x[1] - x[0]
    dpx = p[2:] - p[:-2] #  central difference to approximate the derivative, check https://pythonnumericalmethods.berkeley.edu/notebooks/chapter20.02-Finite-Difference-Approximating-Derivatives.html
    dpx = dpx / 2.0 / delta
    x_drop, p_drop = x[1:-1], p[1:-1] # drop first and last item to align with the size of dpx

    force = sigma**2 * dpx / 2.0 / p_drop
    return x_drop, force

## Simple test on gaussian prior.
#def prior(x, a=10):
#    '''
#    x (array): mesh points from [-a, a]
#    '''
#    d = 4 # sd is a d fraction of total interval
#    p = np.exp(- x**2 / 2 / (a / d)**2)
#
#    return p
#a = 10
#x = np.linspace(-a, a, 50)
#p = prior(x, a=a)
#
#sigma = 1
#x_drop, force = drift_force(x, p, sigma)
#
#plt.figure()
#plt.plot(x_drop, force)
#plt.title('test gaussian case')
#plt.show()

def deg2rad(x):
    return x / 360.0 * 2 * np.pi

def rad2deg(arr, shift=False):
    '''
    arr ranges from -pi to pi and be converted to 0 to 360
    '''
    if shift:
        return (arr + np.pi) / 2 / np.pi * 360
    else:
        return arr / 2 / np.pi * 360

##### load RNN data
keys = ['90.0', '30.0', '27.5', '25.0', '22.5', '20.0', '17.5', '15.0', '12.5']
file_label = keys[5]
out_dir = '../bin/figs/fig_data/rnn_noise_bay_' + file_label + '.json'
fig_name = 'rnn_bay_drift_' + file_label + '.pdf'
data = tools.load_dic(out_dir)
# get rnn_drift force
color_mesh, rnn_drift, bay_drift = np.array(data['rnn_color']), np.array(data['rnn_drift']), np.array(data['bay_drift'])

##### get theoretical drift force
a = np.pi
x = np.linspace(-a, a, 500)
sigma_s = deg2rad(float(file_label)) # width of prior distribution
bias_center = np.array([40., 130., 220., 310.,])
bias_center = deg2rad(bias_center) - np.pi
p = prior_func(x, bias_center, sigma_s) # get the prior distribution

rnn_noise = np.array(data['noise']) # load noise, which is sigma_n in paper
theo_drift = [] # each RNN, although trained with same prior, yet resulting noise (sigma_n) would be different. We fit each sigma_n seperately
x_mesh = []
for sigma_n in rnn_noise:
    x_drop, drift_temp = drift_force(x, p, sigma_n)
    theo_drift.append(drift_temp.copy())  # get the drift force predicted by previous theory
    x_mesh.append(x_drop)
x_mesh, theo_drift = np.array(x_mesh), np.array(theo_drift)

##### reunit and calculate mean and se
m=3.5
color_mesh, rnn_drift, bay_drift= rad2deg(color_mesh, shift=True), rad2deg(rnn_drift), rad2deg(bay_drift)
x_mesh, theo_drift = rad2deg(x_mesh, shift=True).flatten(), rad2deg(theo_drift).flatten()
color_mesh_ppd, mean_drift_rnn, se_drift_rnn = tools.mean_se(color_mesh, rnn_drift, remove_outlier=True, m=m)
color_mesh_ppd, mean_drift_bay, se_drift_bay = tools.mean_se(color_mesh, bay_drift, remove_outlier=True, m=m)
x_mesh_ppd, mean_drift_theo, se_drift_theo = tools.mean_se(x_mesh, theo_drift, remove_outlier=True, m=m)

##### Plotting figure
plt.figure()
plt.plot(color_mesh_ppd, mean_drift_rnn, label='rnn drift force')
plt.fill_between(color_mesh_ppd, mean_drift_rnn - se_drift_rnn, mean_drift_rnn + se_drift_rnn, alpha=0.4)

plt.plot(color_mesh_ppd, mean_drift_bay, label='bayesian drift force (ours)')
plt.fill_between(color_mesh_ppd, mean_drift_bay - se_drift_bay, mean_drift_bay + se_drift_bay, alpha=0.4)

plt.plot(x_mesh_ppd, mean_drift_theo, label='previous theoretical drift force')
plt.fill_between(x_mesh_ppd, mean_drift_theo - se_drift_theo, mean_drift_theo + se_drift_theo, alpha=0.4)

plt.axhline(y=0, linestyle='--')
plt.legend()
plt.title('prior sigma_s is ' + file_label)
plt.show()
