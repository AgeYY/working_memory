# find the fixed points and calculate their related quantities
import numpy as np
import torch
import torch.optim as optim
import core.tools as tools
import core.network as network
import torch.nn as nn
from core.agent import Agent
from core.data_plot.manifold_ploter import Manifold_ploter as MPloter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Fix_point_finder():
    ''' Find the fix points of RNN given a input'''
    def __init__(self, model_dir, rule_name):
        ''' load the model'''
        self.model_dir = model_dir
        self.rule_name = rule_name
        self.hp = tools.load_hp(model_dir)
        self.net = network.RNN(self.hp, is_cuda=False)
        self.net.load(model_dir)
        self.net.sigma_rec = 0 # turn off noise when searching for fixedpoints

    def search(self, input_rnn, batch_size=64, sigma_init=10, n_epochs=50000, hidden0=None, lr=0.1, milestones=[1e10], speed_thresh=None, n_epoch_clect_slow_points=500):
        '''
        input:
          input_rnn (array [float] (input_size)): the input array, can be numpy.
          batch_size (int): number of initial searching points. This must be provided if the hidden0 is None.
          sigma_init (float): the standard deviation of the distribution of the searching points
          n_epochs (int): searching epochs
          hidden0 (array [float] (batch_size, hidden_size)): init condition. If None this function will generate one initial for you.
          speed_thresh (float): speed smaller than this threshold is considered as a fixpoint
        output:
          fixedpoints (numpy array [float] (batch_size, hidden_size)): the fix points
          hidden_init (numpy array): search initial condition
        '''
        ## Here hidden activity is the variable to be optimized
        ## Initialized randomly for search in parallel (activity all positive) if it is not provided in the input
        if hidden0 is None:
            hidden_size = self.net.hidden_size
            hhelper = Hidden0_helper(hidden_size)
            hidden0 = hhelper.random_zero(sigma_init=sigma_init, batch_size=batch_size)
            hidden = hidden0
        else:
            hidden = torch.tensor(hidden0, dtype=torch.float32)
            batch_size = hidden0.shape[0]

        inputs = np.tile(input_rnn, (batch_size, 1))
        inputs = torch.tensor(inputs, dtype=torch.float32)


        hidden_init = hidden.clone().cpu().detach().numpy()
        hidden.requires_grad = True

        # Use Adam optimizer
        optimizer = optim.Adam([hidden], lr=lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
        criterion = nn.MSELoss()

        running_loss = 0
        self.fixedpoints = np.zeros((1, hidden_init.shape[1]))

        print('Searching fixpoints...', flush=True)
        for i in range(n_epochs):
            optimizer.zero_grad()   # zero the gradient buffers
    
            # Take the one-step recurrent function from the trained network
            new_h = self.net.recurrence(inputs, hidden)
            loss = criterion(new_h, hidden)

            if not (speed_thresh is None):
                # store points less than certain speed
                if i % n_epoch_clect_slow_points == 1:
                    hidden_np = hidden.cpu().detach().numpy()
                    hidden_new_np = new_h.cpu().detach().numpy()
                    speed_bool = np.linalg.norm(hidden_new_np - hidden_np, axis=1) < speed_thresh
                    self.fixedpoints = np.vstack((self.fixedpoints, (hidden_np[speed_bool] + hidden_new_np[speed_bool]) / 2 ))

            loss.backward()
            optimizer.step()    # Does the update
            scheduler.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                running_loss /= 1000
                print('Step {}, Loss {:0.10f}'.format(i+1, running_loss), flush=True)
                running_loss = 0
        if speed_thresh is None:
            self.fixedpoints = hidden.cpu().detach().numpy()
        else:
            self.fixedpoints = self.fixedpoints[1:]
        return self.fixedpoints, hidden_init

class Hidden0_helper():
    def __init__(self, hidden_size=0):
        '''different ways to generate the initial hidden state for search fixed points'''
        self.hidden_size = hidden_size

    def center_state(self, sub, sigma_init, batch_size=1000):
        '''
        sub (Agent class): it has finished one expeiment by do_exp
        '''
        hidden_size = sub.state.shape[-1]
        one_color_state = np.mean(sub.state[sub.epochs['interval'][1]]) # choose the state of one color
        hidden0 = one_color_state + np.random.randn(batch_size, hidden_size)*sigma_init # initialize the search points near the color
        return hidden0

    def random_zero(self, sigma_init, batch_size=1000):
        hidden0 = np.random.randn(batch_size, self.hidden_size)*sigma_init
        return hidden0

    def noisy_ring(self, sub, sigma_init=0, batch_size=None):
        '''note the batch size must be the same as state colors'''
        hidden0 = sub.state[sub.epochs['interval'][0]] # use the neural states at the begining of the delay
        batch_size = hidden0.shape[0]
        noise = np.random.randn(batch_size, self.hidden_size)*sigma_init
        hidden0 = hidden0 + noise
        return hidden0

    def mesh_pca_plane(self, sub, xlim=[-10, 10], ylim=[-10, 10], edge_batch_size=10, period_name='interval'):
        '''
        mesh on the hyperplane constructed by the first two principle components
        inputs:
          sub (Agent): has finished at least one experiment, so that we know where the hyperplane should be
          xlim, ylim (array [float] (2)): xlimits and y limits in the pca plane
          edge_batch_size (integer): the number of points in each edge. So the total points would be edge_batch_size^2
          period_name (str): 'fix', 'stim1', 'interval' means delay, 'go_cue' or 'response'. This function will fit pc1-pc2 depends on the end of the period.
        outputs:
          cords_pca (array [float] (edge_batch_size^2, 2)): points in the pca plane
          cords_origin (array [float] (edge_batch_size^2, hidden_size)): each row is a state point
        '''
        n_components = 2
        pca = PCA(n_components=n_components)
        pca.fit(sub.state[sub.epochs[period_name][1] - 1])
        x_mesh = np.linspace(xlim[0], xlim[1], edge_batch_size)
        y_mesh = np.linspace(ylim[1], ylim[0], edge_batch_size)
        x_mesh, y_mesh = np.meshgrid(x_mesh, y_mesh)
        x_flat = x_mesh.flatten()
        y_flat = y_mesh.flatten()
        cords_pca = np.zeros((x_flat.shape[0], 2))
        cords_pca[:, 0] = x_flat
        cords_pca[:, 1] = y_flat

        cords_origin = pca.inverse_transform(cords_pca)

        return cords_pca, cords_origin

    def mesh_fir_rate_pca_plane(self, sub, xlim=[-1, 1], ylim=[-1, 1], edge_batch_size=10):
        '''
        mesh on the hyperplane constructed by the first two principle components, in firing rate sapce
        inputs:
          sub (Agent): has finished at least one experiment, so that we know where the hyperplane should be
          xlim, ylim (array [float] (2)): xlimits and y limits in the pca plane
          edge_batch_size (integer): the number of points in each edge. So the total points would be edge_batch_size^2
        outputs:
          cords_pca (array [float] (edge_batch_size^2, 2)): points in the pca plane, in firing rate space
          cords_state_origin (array [float] (edge_batch_size^2, hidden_size)): each row is a state point
        '''
        n_components = 2
        pca = PCA(n_components=n_components)
        pca.fit(sub.fir_rate[sub.epochs['interval'][1]])

        x_mesh = np.linspace(xlim[0], xlim[1], edge_batch_size)
        y_mesh = np.linspace(ylim[1], ylim[0], edge_batch_size)
        x_mesh, y_mesh = np.meshgrid(x_mesh, y_mesh)
        x_flat = x_mesh.flatten()
        y_flat = y_mesh.flatten()

        cords_pca = np.zeros((x_flat.shape[0], 2))
        cords_pca[:, 0] = x_flat
        cords_pca[:, 1] = y_flat

        cords_fir_rate_origin = pca.inverse_transform(cords_pca)
        cords_state_origin = np.arctanh(cords_fir_rate_origin - 1)

        return cords_pca, cords_state_origin

    def delay_ring(self, sub, sigma_init=0, batch_size=1000, period_name='interval', return_pca=False):
        '''
        At the end of delay, the manifold is a ring (although not perfect). Delay_ring is a ring on the pc1-pc2 plane. Its radius is the same as the points on the end of delay.  We sample points on delay ring, and map it back to the original high dimensional space.
        input:
          sigma_init (float): noise added to the ring
          period_name (str): 'fix', 'stim1', 'interval' means delay, 'go_cue' or 'response'. This function will fit pc1-pc2 depends on the end of the period.
          batch_size (int): number of points sampled from the ring
        '''
        n_components = 2
        pca = PCA(n_components=n_components)
        pca.fit(sub.state[sub.epochs[period_name][1] - 1])

        # find the radius
        hidden0_ring = self.noisy_ring(sub, sigma_init=0)
        hidden0_ring_pca = pca.transform(hidden0_ring)
    
        radius = np.mean(np.linalg.norm(hidden0_ring_pca, axis=1)) # find the radius of that ring

        # sample points on the plane
        angles = np.linspace(0, 2 * np.pi, batch_size, endpoint=False)
        cords_pca = np.zeros((batch_size, 2))

        cords_pca[:, 0] = radius * np.cos(angles)
        cords_pca[:, 1] = radius * np.sin(angles)

        cords_pca = cords_pca + np.random.randn(*cords_pca.shape)*sigma_init

        # map back to the original space

        cords_origin = pca.inverse_transform(cords_pca)

        if return_pca:
            return cords_pca, cords_origin, pca
        else:
            return cords_pca, cords_origin
