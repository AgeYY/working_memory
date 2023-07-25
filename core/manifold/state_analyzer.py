# analyze the state in state space. Input is state, outputs are their properties such as speed, eigenvector etc.
from core.manifold.fix_point import Fix_point_finder, Hidden0_helper
import numpy as np
from core.agent import Agent
from core.rnn_decoder import RNN_decoder
from sklearn.decomposition import PCA
import core.tools as tools
from core.color_error import Circular_operator
import torch
import matplotlib.pyplot as plt

class State_analyzer():
    ''' output properties of input states. The parameters of one object (prod_intervals, ...) should be fixed after one initialization. If you wanna change another parameters, please initialize a new object.'''
    def __init__(self, prod_intervals=800, pca_degree=None, sigma_rec=0, sigma_x=0, n_colors=1000):
        ''' parameters for generate delay ring
        n_colors: number of colors for pca_degree. This would be used only when pca_degree is None
        '''
        self.sigma_rec=sigma_rec; self.sigma_x = sigma_x
        self.prod_intervals=prod_intervals
        if pca_degree is None:
            self.pca_degree = np.linspace(0, 360, n_colors, endpoint=False) # Plot the trajectories of these colors
        else:
            self.pca_degree = pca_degree

        self.n_colors = self.pca_degree.shape[0]

    def read_rnn_file(self, model_dir, rule_name):
        # read in the RNN from file
        self.rule_name = rule_name
        self.sub = Agent(model_dir, rule_name)
        self.hidden_size = self.sub.hp['n_rnn']
        self.input_size = self.sub.hp['n_input']

        self.sub.do_exp(prod_intervals=self.prod_intervals, ring_centers=self.pca_degree, sigma_rec=self.sigma_rec, sigma_x=self.sigma_x)
        self._gen_table() # generate function angle(color) and its derivertive.

    def read_rnn_agent(self, agent):
        # read in the RNN from agent
        self.sub = agent
        self.rule_name = self.sub.rule_name
        self.hidden_size = self.sub.hp['n_rnn']
        self.input_size = self.sub.hp['n_input']

        self.sub.do_exp(prod_intervals=self.prod_intervals, ring_centers=self.pca_degree, sigma_rec=self.sigma_rec, sigma_x=self.sigma_x)
        self._gen_table()

    def velocity_state(self, states, input_rnn=None):
        '''
        state velocity ds/dt
        input:
          states (array [float] (batch_size, hidden_size)): neural states.
          input_rnn (array [float] (input_size)): the input array, can be numpy. for example, rnn with only 3 inputs in the delay, input_rnn = [0, 0, 0]
        input:
          vel (array (batch_size, hidden_size)): velocity = state_{t+1} - state_{t}
        '''
        states_tch = torch.from_numpy(states).type(torch.FloatTensor).to(self.sub.model.device)

        # input
        if input_rnn is None: # in the delay case
            input_rnn = np.zeros(self.input_size)

        batch_size = states.shape[0] # consider states one by one
        inputs = np.tile(input_rnn, (batch_size, 1))
        inputs = torch.tensor(inputs, dtype=torch.float32, device=self.sub.model.device)
        vel = self.sub.model.recurrence(inputs, states_tch) - states_tch #run one step
        return vel.detach().cpu().numpy()

    def velocity_state_central(self, states, input_rnn=None):
        '''
        state velocity v(s_middle) = s_new - s_old, where s_middel = s_new + s_old
        input:
          states (array [float] (batch_size, hidden_size)): neural states.
          input_rnn (array [float] (input_size)): the input array, can be numpy. for example, rnn with only 3 inputs in the delay, input_rnn = [0, 0, 0]
        input:
          vel (array (batch_size, hidden_size)): velocity = state_{t+1} - state_{t}
        '''
        states_tch = torch.from_numpy(states).type(torch.FloatTensor).to(self.sub.model.device)

        # input
        if input_rnn is None: # in the delay case
            input_rnn = np.zeros(self.input_size)

        batch_size = states.shape[0] # consider states one by one
        inputs = np.tile(input_rnn, (batch_size, 1))
        inputs = torch.tensor(inputs, dtype=torch.float32, device=self.sub.model.device)
        states_new = self.sub.model.recurrence(inputs, states_tch)
        vel = states_new - states_tch #run one step
        states_m = (states_new + states_tch) / 2
        return vel.detach().cpu().numpy(), states_m

    def velocity_fr(self, states, input_rnn=None, mode='state'):
        '''
        firing rate velocity. df(s)/dt = df(s)/ds * ds/dt. Where f(s) is the firing rate = tanh(s) + 1
        input_mode ['state' or 'fir']: if state, then input states is neural state, if fir, input states is actually the firing rate
        '''
        if (mode == 'fir'):
            states_temp = np.arctanh(states - 1)
        else:
            states_temp = states
        vs = self.velocity_state(states_temp, input_rnn)
        vfir = (1 - np.tanh(states_temp)**2) * vs
        return vfir

    def angle(self, states, fit_pca=False, state_type='data'):
        '''
        find the angle of states in state space
        states (array [float] (batch_size, hidden_size)): neural states.
        state_type: 'data' or 'vector'. Angle is computed by projecting state to the pc1-pc2 plane, then compute the angle within the pc1-pc2 plane. However, there are two types of projection. One is 'data' type, which is more traditional. A data wil be projected to the pc1-pc2 plane. Consider a fitted pca transformer with mean vector R and pc1, pc2 vector. the data vector will be firstly be substracted by R, then substracted vector will be projected to the pc1, pc2 vectors. Intuitively it's like projecting a data point to the pc1-pc2 plane. The second type is vector projection. In this case we want to directly compute the projection of that vector to two pcs, without substracting the mean.
        '''
        # fit the pca of delay ring
        if fit_pca:
            self.pca = PCA(n_components=2)
            self.pca.fit(self.sub.state[self.sub.epochs['interval'][1]])

        if state_type == 'data':
            states_pca = self.pca.transform(states)
        elif state_type == 'vector':
            states_pca = self.pca.components_ @ states.T
            states_pca = states_pca.T
        else:
            print('The state type can only be data or vector')
            quit()
        angle = np.arctan2(states_pca[:, 1], states_pca[:, 0]) # calculate the angle of states_pca
        angle = np.mod(angle, 2*np.pi) / 2.0 / np.pi * 360.0
        return angle

    def projection(self, states, fit_pca=False):
        '''
        project high dimensional states to the pca1-pc2 plane
        '''
        if fit_pca:
            self.pca = PCA(n_components=2)
            self.pca.fit(self.sub.state[self.sub.epochs['interval'][1]])

        states_pca = self.pca.transform(states)
        return states_pca

    def _gen_table(self):
        '''
        generate angle(color) function and its derivertive dydx(color)
        '''
        self.color, self.angle_pca, self.state = gen_type_RNN(self.sub, prod_intervals=self.prod_intervals, pca_degree=self.pca_degree, sigma_rec=self.sigma_rec, sigma_x=self.sigma_x, n_colors=self.n_colors)
        self.color_for_d, self.dydx = diff_xy(self.color, self.angle_pca) # color, d(angle_pca)/d(color)

    def encode_space_density(self, states):
        '''
        Calculate the angular occupytion of a color -- dphi/dtheta
        states (array [float] (batch_size, hidden_size)): neural states.
        gen_table (bool): generate a table for searching the density
        '''
        angles = self.angle(states) # calculate the angles of input state
        dydx = np.zeros(angles.shape[0])
        i = 0
        for ag in angles:
            # find the nearrest angle in phi_for_d
            idx = tools.find_nearest(self.angle_pca, ag)
            dydx[i] = self.dydx[idx]
            # output dydx
            i=i+1
        return dydx

    def jacobian(self, states, input_rnn=None):
        '''calculate jacobian
        input:
          states (array [float] (batch_size, hidden_size)): neural states.
          input_rnn (array [float] (input_size)): the input array, can be numpy. for example, rnn with only 3 inputs in the delay, input_rnn = [0, 0, 0]
        output:
          jacs (array (batch_size, hidden_size, hidden_size)): jacobians
        '''
        states_tch = torch.from_numpy(states).type(torch.FloatTensor).to(self.sub.model.device)

        # input
        if input_rnn is None: # in the delay case
            input_rnn = np.zeros(self.input_size)

        batch_size = 1 # consider states one by one
        inputs = np.tile(input_rnn, (batch_size, 1))
        inputs = torch.tensor(inputs, dtype=torch.float32, device=self.sub.model.device)

        jacs = np.zeros((states.shape[0], self.hidden_size, self.hidden_size))
        i_jac=0
        for state in states_tch:
            state.requires_grad=True
            deltah = self.sub.model.recurrence(inputs, state) - state #run one step
            deltah = deltah.view(-1)
            jacT = torch.zeros(self.hidden_size, self.hidden_size)
            for i in range(self.hidden_size):
                output = torch.zeros(self.hidden_size, device=self.sub.model.device)
                output[i] = 1.
                jacT[:,i] = torch.autograd.grad(deltah, state, grad_outputs=output, retain_graph=True)[0]

            jac = jacT.detach().cpu().numpy().T
            jacs[i_jac, :, :] = jac
            i_jac=i_jac+1

            state.requires_grad=False

        return jacs

    def eigen(self, jacs):
        '''
        input:
          jacs (array (batch_size, hidden_size, hidden_size)): jacobians
        output:
          eigvec (array (batch_size, hidden_size, hidden_size)): eigenvectors
          eigval (array (batch_size, hidden_size)): eigenvalues
        '''
        eigvec = np.zeros(jacs.shape, dtype=np.cdouble)
        eigval = np.zeros((jacs.shape[0], jacs.shape[1]), dtype=np.complex)

        i_jac=0
        for jac in jacs:
            eigval_temp, eigvec_temp = np.linalg.eig(jac)
            eigval[i_jac, :], eigvec[i_jac, :, :] = eigval_temp, eigvec_temp
            i_jac=i_jac+1
        return eigval, eigvec

    def attractor(self, fixpoints):
        '''
        input:
          fixpoints (array [float] (batch_size, hidden_size)): neural states.
        output:
          atr_status (array [bool] (batch_size)): True -- attractor, false -- saddle
          jacs (array (batch_size, hidden_size, hidden_size)): jacobians
          eigvec (array (batch_size, hidden_size, hidden_size)): eigenvectors
          eigval (array (batch_size, hidden_size)): eigenvalues

        '''
        jacs = self.jacobian(fixpoints)
        eigval, eigvec = self.eigen(jacs)
        atr_status = np.zeros(fixpoints.shape[0])
        for i, egv in enumerate(eigval):
            if np.all(np.real(egv) < 0):
                atr_status[i] = True
        return atr_status, jacs, eigvec, eigval

    def angle_color(self, x, input_var='color'):
        '''
        please self._gen_table before runing this function
        input:
          x (array [float] (batch_size)): if input_var = 'color', x represents colors, or angle then, x is angle. both range from 0 to 360
        input_var (string): color or angle
        output:
          y (array [float] (batch_size)): angle of color
        '''
        y = np.zeros(x.shape[0])
        if input_var == 'color':
            independent = self.color
            dependent = self.angle_pca
        elif input_var == 'angle':
            dependent = self.color
            independent = self.angle_pca

        for i, xi in enumerate(x):
            # find the nearrest angle in phi_for_d
            idx = tools.find_nearest(independent, xi)
            y[i] = dependent[idx]

        return y

    def color_state(self, x):
        '''
        please self._gen_table before runing this function
        input:
          x (array [float] (batch_size)): x represents colors which value from 0 to 360.
        output:
          y (array [float] (batch_size)): state of color
        '''
        y = np.zeros((x.shape[0], self.hidden_size))

        for i, xi in enumerate(x):
            # find the nearrest angle in phi_for_d
            idx = tools.find_nearest(self.color, xi)
            y[i] = self.state[idx, :]

        return y

def gen_type_RNN(sub, prod_intervals=800, pca_degree=None, sigma_rec=0, sigma_x=0, n_colors=400, batch_size=1000):
    '''
    generate data for one type of RNN
    output:
      color (array, [n_color]): colors
      angle_pca (array, [n_color]): angles correspond to the colors
      state (array, [n_color, hidden_sise]): state vector corresponds to the colors
    '''
    ########## Points on the ring
    sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x)

    ##### fit data to find the pca plane
    n_components = 2
    pca = PCA(n_components=n_components)
    pca.fit(sub.state[sub.epochs['interval'][1]])
    
    ##### state in the hidimensional space and pca plane
    hidden_size = sub.state.shape[-1]
    hhelper = Hidden0_helper(hidden_size=hidden_size)
    hidden0_ring_pca, hidden0_ring = hhelper.delay_ring(sub)

    ##### decode states from high dimesional space
    rnn_de = RNN_decoder()
    rnn_de.read_rnn_agent(sub)
    
    report_color_ring = rnn_de.decode(hidden0_ring)
    deg_pca = np.arctan2(hidden0_ring_pca[:, 1], hidden0_ring_pca[:, 0]) # calculate the angle of hidden0_ring_pca
    deg_pca = np.mod(deg_pca, 2*np.pi) / 2.0 / np.pi * 360.0

    angle_pca = deg_pca # angle in the pca space
    color = report_color_ring # the corresponding colors
    state = hidden0_ring

    return color, angle_pca, state

def diff_xy(x, y, d_abs=True, step=1):
    cptor = Circular_operator(0, 360)
    diff_y = cptor.diff(y[step:], y[:-step])
    diff_x = cptor.diff(x[step:], x[:-step])

    if d_abs:
        dydx = abs(diff_y / diff_x) # the derivertive might be all negtive due to the difference of defination of rotational direction in deg_pca and report_color
    else:
        dydx = diff_y / diff_x

    # reorder
    order = np.argsort(x[step:])
    x_order = x[step:][order]
    dydx_order = dydx[order]
    return x_order, dydx_order
