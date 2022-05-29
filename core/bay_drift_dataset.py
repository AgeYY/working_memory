# bayesian drift model. dataset for driftor
import numpy as np
from scipy.stats import vonmises
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from core.bay_drift_loss_torch import prior_s_table

import matplotlib.pyplot as plt

def collate_fn(batch):
    '''batch is generated from dataset, not data loader. This function is to flat the data'''
    return batch[0]

class Noisy_s(Dataset):
    def __init__(self, center_s, sigma_s, sigma_n=0.01, dataset_size=1, n_mesh=720, delay_t_min=0, delay_t_max=2000, batch_size=64, device=torch.device('cpu')):
        '''
        input:
          center_s (array): centers for the prior distribution
          sigma_s (array): width of each center in the prior distribution
          sigma_n (array): diffusion coefficient = sigma_n^2 / 2
          dataset_size (int): size of this dataset
          n_mesh_s (int): number of mesh points in generating distribution table for stimulus
          batch_size (int): this dataset can perform batching. One data point actually contains batch_size pairs of outputs
          t_min, t_max (float): boundaries of time interval. unit is ms
          dataset_size (int): number of batches contains in dataset
          delay_t_max (float): maximum delay time period. The ODESolver will calculate equation from t = 0 to t = delay_t_max
          delay_t_min (float): If you use the dataset with drifter, then the delay_t_min should fix to 0
        '''
        self.center_s = center_s
        self.sigma_s = sigma_s
        self.sigma_n = sigma_n
        self.dataset_size = dataset_size
        self.n_mesh = n_mesh
        self.batch_size = batch_size
        self.size=dataset_size
        self.device=device
        self.delay_t_max = delay_t_max + 1
        self.delay_t_min = delay_t_min

        self.mesh_s, self.pdf_s = prior_s_table(self.center_s, self.sigma_s, n_mesh) # generate pdf for prior distribution

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # get the stimulus and time
        # output shape (batch_size), (batch_size), (batch_size)
        # delay_t is about 1000, the angular range is -pi to pi. So sigma_n2 should in the order of 1e-2 -- 1e-1
        prior_s = np.random.choice(self.mesh_s, p=self.pdf_s, size=self.batch_size)
        # get the time
        delay_t = np.random.uniform(self.delay_t_min, self.delay_t_max, size=self.batch_size)
        # get the noisy input
        sigma = self.sigma_n * np.sqrt(delay_t)
        kappa = 1 / sigma**2
        noisy_s = np.random.vonmises(prior_s, kappa, size=self.batch_size)

        # move to device
        prior_s = torch.from_numpy(prior_s).to(self.device)
        delay_t = torch.from_numpy(delay_t).to(self.device)
        noisy_s = torch.from_numpy(noisy_s).to(self.device)

        return prior_s, delay_t, noisy_s
