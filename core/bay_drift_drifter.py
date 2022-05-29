# solve ODE dx/dt = f(x, t) using pytorch

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

def odeEuler(f,y0,t, device=torch.device('cpu')):
    '''Approximate the solution of y'=f(y,t) by Euler's method.

    Parameters
    ----------
    f : function
        Right-hand side of the differential equation y'=f(t,y), y(t_0)=y_0
    y0 (array [batch_size]):
        Initial value y(t0)=y0 wher t0 is the entry at index 0 in the array t
    t (array [batch_size, time_points]): t values where we approximate y values. Time step
        at each iteration is given by t[n+1] - t[n].

    Returns
    -------
    y (array [batch_size, time_points]): 1D NumPy array
        Approximation y[n] of the solution y(t_n) computed by Euler's method.

    '''
    y = torch.zeros(t.size(), device=device)
    y[:, 0] = y0
    for n in range(0,t.shape[1]-1):
        y[:, n+1] = y[:, n] + f(y[:, n],t[:, n])*(t[:, n+1] - t[:, n])
    y = (y + np.pi) % (2 * np.pi) - np.pi
    return y

class Drifter(nn.Module):
    def __init__(self, delay_t_max=2000, fs_order=5, dt=20, scale_force_unit=0.0001, device=torch.device('cpu')):
        '''
        delay_t_max (float): maximum delay time period. The ODESolver will calculate equation from t = 0 to t = delay_t_max
        dt ([float]): time steps in Euler method
        fs_order ([int]): the highest order in the fourier expansion on weight
        '''
        super(Drifter, self).__init__()
        self.delay_t_max = delay_t_max + 1
        self.fs_order = fs_order
        self.dt = dt
        self.device = device
        self.scale_force_unit = scale_force_unit / fs_order # scaling the unit of force
        sin_weight, cos_weight = self.scale_force_unit * torch.randn(2, fs_order, device=self.device)
        self.sin_weight, self.cos_weight = nn.Parameter(sin_weight), nn.Parameter(cos_weight)

        self.t = torch.arange(0, self.delay_t_max, self.dt, device=self.device, dtype=torch.float) # time mesh

    def gen_t_mask(self, t_sample):
        t_mask = torch.zeros(self.t_mesh.size(), device=self.device, dtype=torch.bool) # True means this time point should be used in loss function

        t_idx = torch.div(t_sample, self.dt, rounding_mode='floor').type(torch.long)
        for i in range(len(t_idx)):
            t_mask[i, t_idx[i]] = True
        return t_mask

    def drift_force(self, x, t=[]):
        '''
        function expanded by fourier seriers with period -pi to pi.
        input:
          x (array, [batch_size]).
          t (empty): just to match the input for odeEuler.
          cos_weight, sin_weight (array, ): these two weights do not necessarily the same. But they should all start with order 0 (although order 0 of sin is 0)
        '''
        y = torch.zeros(x.size(), device=self.device)
        for i in range(len(self.sin_weight)):
            y += self.sin_weight[i] * torch.sin(i * x)

        for i in range(len(self.cos_weight)):
            y += self.cos_weight[i] * torch.cos(i * x)

        return y

    def forward(self, x0_sample, t_sample):
        '''
        compute dx/dt = f(x, t)
        '''
        batch_size = x0_sample.shape[0]
        self.t_mesh = self.t.repeat(batch_size, 1)

        self.t_mask = self.gen_t_mask(t_sample)

        self.xt = odeEuler(self.drift_force, x0_sample, self.t_mesh, device=self.device)

        return self.t_mesh, self.xt
