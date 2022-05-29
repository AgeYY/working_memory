import context
import numpy as np
import torch
import torch.optim as optim
import core.tools as tools
import core.network as network
import torch.nn as nn
from core.agent import Agent
from core.data_plot.manifold_ploter import Manifold_ploter as MPloter
import matplotlib.pyplot as plt
from core.manifold.fix_point import Fix_point_finder, Hidden0_helper
from core.manifold.state_analyzer import State_analyzer
import pandas as pd

model_dir = '../core/model_local/color_reproduction_delay_unit/model_16/noise_delta'
rule_name = 'color_reproduction_delay_unit'
prod_intervals = 0
n_colors = 1400
batch_size = n_colors
pca_degree = np.linspace(0, 360, n_colors, endpoint=False) # Plot the trajectories of these colors
n_epochs = 30000
sigma_init = 10
milestones = [10000, 20000, 25000, 30000, 35000]
lr = 0.01
sigma_rec=0; sigma_x = 0
data_out_dir = './data/fixpoints.json'
gen_data = True
alpha = 0.7 # alpha for delay trajectories

def gen_points(sub):
    '''
    generate fix points and ghost points
    '''
    sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x)

    ## find the fixed points
    ffinder = Fix_point_finder(model_dir, rule_name)

    input_size = ffinder.net.input_size
    input_rnn = np.zeros(input_size)

    # use different stratage to find the initial hidden state
    hidden_size = sub.state.shape[-1]
    hhelper = Hidden0_helper(hidden_size)
    hidden0 = hhelper.center_state(sub, sigma_init=sigma_init, batch_size=batch_size)
    #hidden0 = hhelper.random_zero(sigma_init=sigma_init)
    #hidden0 = hhelper.noisy_ring(sub, batch_size)
    
    result_points, hidden_init = ffinder.search(input_rnn, n_epochs=n_epochs, batch_size=batch_size, hidden0=hidden0, milestones=milestones)

    tools.save_dic({'result_points': result_points, 'fir_rate': sub.fir_rate}, data_out_dir)

sub = Agent(model_dir, rule_name)

if gen_data:
    gen_points(sub)

result_points = tools.load_dic(data_out_dir)['result_points']
result_points = np.array(result_points, dtype=float)


##### Plot the firing rate
#sub.do_exp(prod_intervals=2000, ring_centers=np.linspace(0, 360, 40, endpoint=False), sigma_rec=sigma_rec, sigma_x=sigma_x)
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#mplot = MPloter()
#mplot.load_data(sub.fir_rate, sub.epochs, sub.behaviour['target_color'])
#
## convert to firing rate
#fixedpoints = ffinder.net.act_fcn(torch.tensor(fixedpoints))
#
#mplot.pca_with_fix_point(fixedpoints, start_time= sub.epochs['interval'][0], end_time=sub.epochs['interval'][1], ax = ax)
#plt.show()

sa = State_analyzer(prod_intervals=prod_intervals, pca_degree=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x)
sa.read_rnn_agent(sub)

#angle of fixedpoints
angle = sa.angle(result_points, fit_pca=True)

unique_angle_arg = tools.arg_select_unique(angle, 5)

# show the angle
angle_unique = angle[unique_angle_arg]
#print(angle_unique)
#tools.circular_hist(ax, angle_unique / 360 * 2 * np.pi, bins=80)
#plt.show()

result_points_unique = result_points[unique_angle_arg]

vel = sa.velocity(result_points_unique)
speed = np.linalg.norm(vel, axis=1)

low_speed_arg = speed < 1e-4
fixedpoints = result_points_unique[low_speed_arg]

#dens = sa.encode_space_density(fixedpoints, gen_table=True)
#print(dens)

arr_status, jacs, eigvec, eigval = sa.attractor(fixedpoints)
print(arr_status)

# Plot distribution of eigenvalues of one fixpoint in a 2-d real-imaginary plot
plt.figure()
plt.scatter(np.real(eigval[0]), np.imag(eigval[0]))
plt.plot([0, 0], [-1, 1], '--')
plt.xlabel('Real')
plt.ylabel('Imaginary')

##### Plot speed of fixedpoints
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
ax.scatter(angle_unique / 360 * 2 * np.pi, -np.log10(speed))

##### Plot delay trajectories and the fixpoints
sub.do_exp(prod_intervals=2000, ring_centers=np.linspace(0, 360, 40, endpoint=False), sigma_rec=sigma_rec, sigma_x=sigma_x)
mplot = MPloter()
mplot.load_data(sub.fir_rate, sub.epochs, sub.behaviour['target_color'])
mplot._pca_fit(2, start_time=sub.epochs['interval'][0], end_time=sub.epochs['interval'][1])
fig_2d = plt.figure(figsize=(5, 3))
axext_2d = fig_2d.add_subplot(111)

_, ax = mplot.pca_2d_plot(start_time=sub.epochs['interval'][0], end_time=sub.epochs['interval'][1], ax = axext_2d, alpha=alpha, do_pca_fit=False)
fixedpoints_act = sub.model.act_fcn(torch.tensor(fixedpoints))
fixedpoints_2d = mplot.pca.transform(fixedpoints_act)
axext_2d.scatter(fixedpoints_2d[:, 0], fixedpoints_2d[:, 1], color='black')

plt.show()
