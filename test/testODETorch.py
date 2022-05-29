import context
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import core.bay_drift_drifter as drifter
import core.bay_drift_dataset as dft_dataset
import core.bay_drift_loss_torch as dft_loss
from torch.utils.data import DataLoader

def stay_loss(xt, x0=0, t_mask=None, t_mesh=None):
    '''
    this loss function requires xt to stay in position x0 as long as possible
    '''
    if t_mask is None:
        return torch.norm(xt[:, -1] - x0)
    else:
        return torch.norm((xt[t_mask] - x0))

c_center = np.array([-np.pi, -np.pi / 2, 0, np.pi/2])
#c_center = np.array([-np.pi / 2, np.pi/2])
n_center = len(c_center)
sigma_s = np.ones(n_center) * 0.15
sigma_n = 0.004 # slow to attractor
#sigma_n = 0.05 # Srong attractor
n_epochs = 500
batch_size=16
dataset_size=1 # total number of trial is n_epochs * batch_size
fs_order = 10
learning_rate = 1e-5
device = torch.device('cpu')
x0 = 1
baseline = 1.0

input_set = dft_dataset.Noisy_s(c_center, sigma_s, sigma_n, batch_size=batch_size, dataset_size=dataset_size, device=device, baseline=baseline) # 1 batch with batchsize 2

input_loader = DataLoader(input_set, batch_size=1, shuffle=False, collate_fn=dft_dataset.collate_fn)

dfter = drifter.Drifter(fs_order=fs_order, device=device)
post = dft_loss.Posterior(c_center, sigma_s, sigma_n, baseline=baseline)
post.norm_table()

optimizer = torch.optim.Adam(dfter.parameters(), lr=learning_rate)

count = 0
for i in range(n_epochs):
    for prior_s, delay_t, noisy_s in input_loader:

        t, xt = dfter(noisy_s, delay_t)
        #loss = stay_loss(xt, x0=noisy_s, t_mask=None, t_mesh=dfter.t_mesh)
        #loss = stay_loss(xt, x0=0, t_mask=dfter.t_mask, t_mesh=dfter.t_mesh)
        loss = post.loss_tch(xt[dfter.t_mask], noisy_s, delay_t)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count += 1
        if count % 100 == 0:
            current = batch_size * count
            print(f"loss: {loss:>7f} \t trials: {count * batch_size:>5d}")

plt.figure(0)
t, xt = t.detach().cpu().numpy(), xt.detach().cpu().numpy()
for i in range(noisy_s.shape[0]):
    plt.plot(t[i], xt[i])

plt.figure(1)
x_mesh = np.linspace(-np.pi, np.pi, 50)
x_mesh = torch.from_numpy(x_mesh)

plt.plot(x_mesh.numpy(), dfter.drift_force(x_mesh).detach().numpy())

# plot the prior and likelihood distribution
plt.figure(2)
mesh = np.linspace(-np.pi, np.pi, 100)
prior_y = dft_loss.prior_func(mesh, c_center, sigma_s, baseline=baseline)
lhd_y = dft_loss.likelihood_func(mesh, 0, 1000, sigma_n)
plt.plot(mesh, prior_y)
plt.plot(mesh, lhd_y)
plt.show()
