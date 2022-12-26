# From prior distribution of information into drift force according to the theory of Zachary 2018
import context

import numpy as np
import matplotlib.pyplot as plt
from core.bay_drift_loss_torch import prior_func

def prior(x, a=10):
    '''
    x (array): mesh points from [-a, a]
    '''
    ## assume uniform prior function in [-a, a]
    #p = 1.0 / 2.0 / a
    # p = p * np.ones(x.shape)

    # assume gaussian, unnormalized only for simple test
    d = 4 # sd is a d fraction of total interval
    p = np.exp(- x**2 / 2 / (a / d)**2)

    return p

def drift_force(x, p, sigma):
    '''
    x (array): equally spaced mesh points, e.g. [0, 0.1, 0.2, ..., 0.9, 1.0]
    p (array): in the same shape of x, prior probability distribution density function
    sigma (float): ddf should be written as d\theta = - du/d\theta dt + \sigma d\epsilon(t)
    output:
      force = - du/d\theta = \sigma^2 / 2 * p'(\theta) / p(\theta)
    '''
    delta = x[1] - x[0]
    dpx = np.diff(p) / delta
    p_drop = p[:-1] # drop the last number so the length is align with dpx

    force = sigma**2 * dpx / 2.0 / p_drop
    return force

a = 10
x = np.linspace(-a, a, 50)
p = prior(x, a=a)

sigma = 1
force = drift_force(x, p, sigma)
x_drop = x[:-1]

plt.figure()
plt.plot(x_drop, force)
plt.title('test gaussian case')
plt.show()

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

a = np.pi
x = np.linspace(-a, a, 500)
sigma_s = deg2rad(12.5)
bias_center = np.array([40., 130., 220., 310.,])
bias_center = deg2rad(bias_center) - np.pi
p = prior_func(x, bias_center, sigma_s)

sigma = 0.006
force = drift_force(x, p, sigma)
x_drop = x[:-1]

x_drop, force = rad2deg(x_drop, shift=True), rad2deg(force)
x = rad2deg(x, shift=True)

plt.figure()
#plt.plot(x, p, label='prior distribution')
plt.plot(x_drop, force, label='drift force')
plt.axhline(y=0, linestyle='--')
plt.legend()
plt.show()
