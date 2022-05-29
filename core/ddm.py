# solve diffusion-drift equation
import matplotlib.pyplot as plt
import sys
import numpy as np
from core.diff_drift import Diff_Drift, plot_traj
from scipy.stats import vonmises
from core.tools import dif_matrix_circular, dif_matrix_circular_2nd, find_nearest
from scipy import linalg
from scipy.optimize import minimize_scalar
import torch
from torch import nn
from core.bay_drift_loss_torch import vonmises_pdf_tch
from core.RNN_dataset import RNN_Dataset
from torch.utils.data import DataLoader
from core.tools import collate_fn

class DDM(nn.Module):
    '''
    diffusion drift model
    '''
    def __init__(self, sigma_init=0.05, dt=20, t_max=1000, t_min=0):
        '''
        input:
          sigma_init (float): the width of initial probability distribution
          dt (float): time mesh space
          t_max (float): max delay period of time
        '''
        super(DDM, self).__init__()
        self.epsilon = torch.tensor(1e-5) # if the probability is negtive, we think its epsilon
        self.sigma_init = sigma_init
        self.dt = dt; self.t_max = t_max;
        sigma_diff = torch.rand(1) * 0.05 # 0.05 is only a scaling factor for initial value
        #self.sigma_diff = nn.Parameter(sigma_diff) # noise d theta = sigma dW + g(x) dt
        self.sigma_diff=sigma_diff
        self.t_mesh = np.arange(t_min, t_max + dt, dt)

    def set_drift(self, c_mesh, gtheta):
        '''
        input:
          gtheta (array [n_color]): drift field.
          c_mesh (array [n_color]): color value for the corresponding gtheta. Must be equally spaced.
        '''
        self.c_mesh = c_mesh
        self.gtheta = gtheta
        self.n_color = len(self.c_mesh)

        self.c_mesh_tch = torch.from_numpy(c_mesh).type(torch.float32)

    def prepare(self):
        '''
        prepare to solve fokker planck equation
        '''
        d_color = self.c_mesh[1] - self.c_mesh[0] # color step
        Ddff = dif_matrix_circular_2nd(self.c_mesh) # diffusion matrix
        Dder = dif_matrix_circular(self.c_mesh, n=1) #drift matrix

        self.Ddff_term = torch.from_numpy(Ddff).type(torch.float32)
        Dder_term = np.dot(Dder, np.diag(self.gtheta))
        self.Dder_term = torch.from_numpy(Dder_term).type(torch.float32)


    def loss(self, init_color, delay_t, report_color):
        '''
        input:
          init_color (torch tensor [batch_size]): angle in the begining of the delay period
          delay_t (torch tensor [batch_size]): The required delay time
          report_color (torch tensor [batch_size]): report color at the end of delay
        '''
        # initial probability distribution
        self.evo_mat = self.sigma_diff**2 / 2 * self.Ddff_term - self.Dder_term
        kappa = 1 / self.sigma_init**2
        batch_size = init_color.size()[0]
        loss = torch.tensor(0)

        for i in range(batch_size):
            p0 = vonmises.pdf(self.c_mesh, kappa, loc=init_color[i]) # 
            p0 = torch.from_numpy(p0).type(torch.float32)


            if torch.any(torch.isnan(p0)):
                print('p0 nan warning: ', p0)

            d_evo = torch.matrix_exp(self.evo_mat * delay_t[i])

            if torch.any(torch.isnan(d_evo)):
                print('devo nan warning: ', d_evo)

            p1 = torch.matmul(d_evo, p0)
            idx = find_nearest(self.c_mesh, report_color[i].numpy())
            loss = loss - torch.log(torch.nn.functional.relu(p1[idx]) + self.epsilon) # likelihood loss


            # use this to check if sigma_init and bin_width reasonable. The distribution should looks smooth
            #plt.figure()
            #plt.imshow(self.fokker_planck_p_mat(init_color[i], delay_t[i]))
            #plt.show()

            if torch.isnan(loss):
                print('loss nan warning with p1[idx] = : ', p1[idx])

        loss = loss / (i + 1)

        return loss

    def fokker_planck_p_mat(self, theta0, delay_t):
        '''
        input:
          theta0 (array [batch_size]): angle in the begining of the delay period. from -np.pi to np.pi
          delay_t (array [batch_size]): The required delay time
        '''
        # initial probability distribution
        self.evo_mat = self.sigma_diff**2 / 2 * self.Ddff_term - self.Dder_term
        kappa = 1 / self.sigma_init**2
        p0 = vonmises.pdf(self.c_mesh, kappa, loc=theta0)
        p0 = torch.tensor(p0)

        p_evo = torch.zeros((self.c_mesh.shape[0], self.t_mesh.shape[0]))
        p_evo[:, 0] = p0

        with torch.no_grad():
            d_evo = torch.matrix_exp(self.evo_mat * self.dt)

            for i in range(1, len(self.t_mesh), 1):
                p_evo[:, i] = torch.matmul(d_evo, p_evo[:, i-1])
        return p_evo

class Euler_Maruyama_solver():
    def read_terms(self, drift_x, drift_y, c_sigma):
        '''
        read two terms in ddm.
        input:
          drift_x (array): colors, value ranges from -np.pi to np.pi
          drift_y (array): drift of colors.
          c_sigma (float): second term sigma dW
        '''
        self.drift_x = drift_x
        self.drift_y = drift_y
        self.c_sigma = c_sigma

    def mu(self, x, t):
        drift = np.zeros(x.shape[0])
        for i, xi in enumerate(x):
            drift[i] = self.drift_y[find_nearest(self.drift_x, xi)]
        return drift
    def sigma(self, x, t):
        return self.c_sigma
    def dW(self, delta_t, batch_size):
        return np.random.normal(loc=0.0, scale=np.sqrt(delta_t), size=batch_size)

    def run(self, init_x, time_end, dt=20):
        '''
        solve the ddm
        input:
          init_x (np array [n_init]):
          time_end (float): unit ms. Typically from 100 to 2000
          dt (float): time spacing. The one used in RNN is 20ms
        output:
          sol (array [(n_init, time_end/dt)]): solution
        '''
        ts = np.arange(0, time_end + dt, dt)
        n_init = init_x.shape[0]
        sol = np.zeros((time_end // dt + 1, n_init))
        sol[0, :] = init_x
        for i in range(1, ts.shape[0]):
            t = (i - 1) * dt
            y = sol[i - 1, :]
            sol[i, :] = y + self.mu(y, t) * dt + self.sigma(y, t) * self.dW(dt, n_init) # you need multiple dw
        return ts, sol

def fit_ddm(sub, bin_width=5, batch_size=800, prod_interval=1000, sigma_init=0.2):
    '''
    fit the drift and noise coefficient. Unit of drift is rad/ms
    '''
    ddf = Diff_Drift()
    ddf.read_rnn_agent(sub)

    color_bin, v_bin = ddf.drift(bin_width=bin_width, padding=0)
    color_bin, v_bin = color_bin / 360.0 * 2 * np.pi - np.pi, v_bin / 360.0 * 2 * np.pi # convert unit to rad

    #plt.figure()
    #plt.plot(color_bin, v_bin)
    #plt.show()

    ddm = DDM(sigma_init=sigma_init, t_max=prod_interval)

    ddm.set_drift(color_bin, v_bin)
    ddm.prepare()

    rnn_ds = RNN_Dataset(batch_size)
    rnn_ds.set_sub(sub)
    ds_loader = DataLoader(rnn_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    init_color, report_color, delay_t = rnn_ds[0]

    def llk_noise_func_wrapper(sigma_diff):
        '''
        one variable function. input is sigma_diff, output is the likelihood
        '''

        with torch.no_grad():
            ddm.sigma_diff = sigma_diff
            loss = ddm.loss(init_color, delay_t, report_color)
        #print('diffusion strength fitting loss: ', loss.numpy())
        #sys.stdout.flush()
        return loss.numpy()

    
    res = minimize_scalar(llk_noise_func_wrapper, bracket=[0, 0.1], tol=1e-5)
    return color_bin, v_bin, res.x, res.fun
