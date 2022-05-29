import numpy as np
import core.bay_drift_dataset as dft_dataset # dataset
import core.bay_drift_drifter as drifter # model
import core.bay_drift_loss_torch as dft_loss # loss function
from torch.utils.data import DataLoader
import torch


def get_bay_drift(color_mesh, c_center, sigma_s, sigma_n, batch_size=16, dataset_size=1, device=torch.device('cpu'), fs_order=10, learning_rate=1e-6, n_epochs=500):
    '''
    given the prior distribution and the strength of noise, output the bayesian optimized drift force
    input:
      color_mesh (array): x mesh of bay_drift
      c_center (array): position of centers in prior distribution. For example, [40, 130, 220, 310]
      center_prob (float): probability of stimuli occur within the range of centers +- 2 * sigma_s.
      sigma_s (float): width of centers
      sigma_n (float): noise term in ddm
      fs_order (int): expand the bay_drift to the first ten order of Fourier series
      learning_rate (float): learning rate when optimizing the score function
    output:
      bay_drift (array): bayesian optimized force corresponding to the color_mesh
    '''
    input_set = dft_dataset.Noisy_s(c_center, sigma_s, sigma_n, batch_size=batch_size, dataset_size=dataset_size, device=device) # 1 batch with batchsize 2
    input_loader = DataLoader(input_set, batch_size=1, shuffle=False, collate_fn=dft_dataset.collate_fn)

    dfter = drifter.Drifter(fs_order=fs_order, device=device)

    post = dft_loss.Posterior(c_center, sigma_s, sigma_n)
    post.norm_table()
    optimizer = torch.optim.Adam(dfter.parameters(), lr=learning_rate)

    count = 0
    for i in range(n_epochs):
        for prior_s, delay_t, noisy_s in input_loader:

            t, xt = dfter(noisy_s, delay_t)
            loss = post.loss_tch(xt[dfter.t_mask], noisy_s, delay_t)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1
            if count % 10 == 0:
                current = batch_size * count
                print(f"bayesian drift loss: {loss:>7f} \t trials: {count * batch_size:>5d}")

    color_mesh_tch = torch.from_numpy(color_mesh)
    bay_drift = dfter.drift_force(color_mesh_tch).detach().cpu().numpy().astype(float)
    return bay_drift
