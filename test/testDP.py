# solve the fokker plank equation numerically given the drift field
import context
import matplotlib.pyplot as plt
from core.agent import Agent
import numpy as np
from core.diff_drift import Diff_Drift, plot_traj
import torch
from torch import nn
from core.ddm import DDM
from core.RNN_dataset import RNN_Dataset
from core.tools import collate_fn
from torch.utils.data import DataLoader
from scipy.optimize import minimize_scalar


model_dir = '../core/model_local/color_reproduction_delay_unit_vonmise_cp7_np2_2c/model_13/noise_delta_p2'
rule_name = 'color_reproduction_delay_unit'

prod_intervals = 800
n_colors = 720
pca_degree = np.linspace(0, 360, n_colors, endpoint=False) # Plot the trajectories of these colors
dt = 20
bin_width = 5
batch_size = 1600
learning_rate = 0.005
n_epoch = 1000

sub = Agent(model_dir, rule_name)
ddf = Diff_Drift()
ddf.read_rnn_agent(sub)

color_bin, v_bin = ddf.drift(bin_width=bin_width)
color_bin, v_bin = color_bin / 360.0 * 2 * np.pi - np.pi, v_bin / 360.0 * 2 * np.pi # convert unit to rad

plt.plot(color_bin, v_bin)
plt.show()

ddm = DDM()

ddm.set_drift(color_bin, v_bin)
ddm.prepare()

rnn_ds = RNN_Dataset(batch_size)
rnn_ds.set_sub(sub)
ds_loader = DataLoader(rnn_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

#optimizer = torch.optim.Adam(ddm.parameters(), lr=learning_rate)

#count = 0
#for init_color, report_color, delay_t in ds_loader:
#    for i in range(n_epoch):
#        loss = ddm.loss(init_color, delay_t, report_color)
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#
#        count += 1
#        if count % 1 == 0:
#            current = batch_size * count
#            print(ddm.sigma_diff)
#            print(f"loss: {loss:>7f} \t trials: {count * batch_size:>5d}")
#
#print(ddm.sigma_diff)

#plt.figure(2)
#p_evo = ddm.fokker_planck_p_mat(0, 2000)
#plt.imshow(p_evo, aspect= p_evo.shape[1] / p_evo.shape[0] )
#plt.colorbar()
#plt.show()

init_color, report_color, delay_t = rnn_ds[0]

def llk_noise_func_wrapper(sigma_diff):
    '''
    one variable function. input is sigma_diff, output is the likelihood
    '''

    with torch.no_grad():
        ddm.sigma_diff = sigma_diff
        loss = ddm.loss(init_color, delay_t, report_color)
        return loss.numpy()

#init_color, report_color, delay_t = rnn_ds[0]
#print(rnn_ds[0])
#loss = llk_noise_func_wrapper(0.005)
#print(loss)

#n_mesh = 100
#sigma_diff_mesh = np.linspace(0, 0.1, n_mesh)
#loss_mesh = np.zeros(n_mesh)
#for i, sigma_diff in enumerate(sigma_diff_mesh):
#    loss_mesh[i] = llk_noise_func_wrapper(sigma_diff)
#
#plt.plot(sigma_diff_mesh, loss_mesh)
#plt.xlabel('sigma_diff')
#plt.ylabel('likelihood')
#plt.show()

res = minimize_scalar(llk_noise_func_wrapper, bracket=[0, 1], tol=1e-4)
print(res.x, res.fun)
