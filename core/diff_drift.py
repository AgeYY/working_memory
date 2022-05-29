# input a RNN and output a the diffusion drift.
import matplotlib.pyplot as plt
from core.agent import Agent, Agent_loader, Agent_group
import numpy as np
from core.rnn_decoder import RNN_decoder
from core.manifold.state_analyzer import diff_xy
from core.net_struct.struct_analyzer import bin_mat
from core.manifold.state_analyzer import State_analyzer
from core.delay_runner import Delay_runner
import pandas as pd
import sys

class Diff_Drift():
    def eat_para(self, prod_intervals=800, n_colors=300, input_color=None, sigma_x=0, sigma_rec=0, input0=0, input1=360):
        if input_color is None:
            self.input_color = np.linspace(input0, input1, n_colors, endpoint=False) # Plot the trajectories of these colors
            self.n_colors = n_colors
        else:
            self.n_colors = len(input_color)
            self.input_color = input_color
        self.prod_intervals = prod_intervals
        self.sigma_x = sigma_x; self.sigma_rec = sigma_rec

    def read_rnn_file(self, model_dir, rule_name):
        # read in the RNN from file
        self.rule_name = rule_name
        self.sub = Agent(model_dir, rule_name)
        self.hidden_size = self.sub.hp['n_rnn']
        self.input_size = self.sub.hp['n_input']
        self.dt = self.sub.hp['dt']

    def read_rnn_agent(self, agent):
        # read in the RNN from agent
        self.sub = agent
        self.rule_name = self.sub.rule_name
        self.hidden_size = self.sub.hp['n_rnn']
        self.input_size = self.sub.hp['n_input']
        self.dt = self.sub.hp['dt']

    def traj(self, padding=5, prod_intervals=800, n_colors=300, input_color=None, sigma_x=0, sigma_rec=0, input0=0, input1=360):
        '''
        input:
          padding (int): ignore the first and last padding points
        output:
          time (array, [n_time_delay - 2 * padding]): time in delay. delay start correspond to time = 0
          colors (array, [n_time, batch_size]): color of state points. n_colors = batch_size
        '''
        self.eat_para(prod_intervals=prod_intervals, n_colors=n_colors, input_color=input_color, sigma_x=sigma_x, sigma_rec=sigma_rec, input0=input0, input1=input1)

        self.sub.do_exp(prod_intervals=self.prod_intervals, ring_centers=self.input_color, sigma_rec=self.sigma_rec, sigma_x=self.sigma_x)
        # shape of state is same as fir_rate fir_rate ( array [float] (time_len, ring_centers, rnn_size) )
        delay_epoch = self.sub.epochs['interval']
        states_delay = self.sub.state.copy()[delay_epoch[0] + padding: delay_epoch[1] - padding]
        states = states_delay.reshape((-1, states_delay.shape[-1]))

        rnn_de = RNN_decoder()
        rnn_de.read_rnn_agent(self.sub)
        colors = rnn_de.decode(states)

        colors = colors.reshape(states_delay.shape[:-1]) # ignore the dimension of hidden size. shape is [time, batch_size]

        time = np.arange((padding)*self.dt, (delay_epoch[1] - delay_epoch[0] - padding) * self.dt, self.dt)

        return time, colors

    def traj_fix_start(self, init_color, prod_intervals=800, sigma_x=0, sigma_rec=0):
        '''
        the color status during delay period. The initial color of the delay is fixed to color_start
        input:
          color_start (array, [n_colors]): initial colors
          prod_interval (int): delay time length
          sigma_x, sgiam_rec (float): noise of RNN
        output:
          time (array, [n_time_delay]): time in delay. delay start correspond to time = 0
          colors (array, [n_time, batch_size]): color of state points. n_colors = batch_size
        '''

        # convert the required color to init_state
        sa = State_analyzer()
        sa.read_rnn_agent(self.sub)
        init_states = sa.color_state(init_color)

        # run delay epoch
        dr = Delay_runner()
        dr.read_rnn_agent(self.sub)
        states_delay = dr.delay_run(init_states, prod_intervals, sigma_rec=sigma_rec, sigma_x=sigma_x)

        states = states_delay.reshape((-1, states_delay.shape[-1]))

        sys.stdout.flush()

        rnn_de = RNN_decoder()
        rnn_de.read_rnn_agent(self.sub)
        colors = rnn_de.decode(states)

        colors = colors.reshape(states_delay.shape[:-1]) # ignore the dimension of hidden size. shape is [time, batch_size]

        time = np.arange(0, colors.shape[0] * self.dt, self.dt)

        return time, colors

    def drift(self, bin_width=5, padding=5, prod_intervals=800, n_colors=300, input_color=None, sigma_x=0, sigma_rec=0):

        #time, colors = self.traj(padding=padding, prod_intervals=prod_intervals, n_colors=n_colors, input_color=input_color, sigma_x=sigma_x, sigma_rec=sigma_rec)
        init_color = np.linspace(0, 360, n_colors)
        time, colors = self.traj_fix_start(init_color, prod_intervals=prod_intervals, sigma_x=sigma_x, sigma_rec=sigma_rec)

        color_v_batch = []; v_batch = [];
        for i in range(colors.shape[-1]): # loop over all batches
            color_v, dtdc = diff_xy(colors[:, i], time, d_abs=False, step=1) # diff_xy will output d time / d color in this case. velocity is the reverce of dtdc
            v = 1.0 / dtdc
            color_v_batch.append(color_v)
            v_batch.append(v)

        color_v = np.concatenate(color_v_batch).flatten()
        velocity = np.concatenate(v_batch).reshape((1, -1)) # convert to 2d so you can use bin_mat

        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #    print(pd.DataFrame(colors))
        #    print(pd.DataFrame(color_v))

        if bin_width is None:
            return color_v, velocity
        else:
            v_bin, color_bin, _ = bin_mat(velocity, color_v, bin_width=bin_width) ## empty
            v_bin = v_bin.flatten()
            return color_bin, v_bin


## plot trajectories
#for i in range(colors.shape[-1]): # loop over all batches
#    plt.plot(np.arange(len(colors)) * dt, colors[:, i])
#plt.show()

#time = np.arange(len(colors)) * dt # time points in delay

#color_v_batch = []; v_batch = [];
#for i in range(colors.shape[-1]): # loop over all batches
#    color_v, dtdc = diff_xy(colors[:, i], time, d_abs=False, step=1) # diff_xy will output d time / d color in this case. velocity is the reverce of dtdc
#    v = 1.0 / dtdc
#    color_v_batch.append(color_v)
#    v_batch.append(v)
#
#color_v = np.concatenate(color_v_batch).flatten()
#velocity = np.concatenate(v_batch).reshape((1, -1)) # convert to 2d so you can use bin_mat
#
#plt.scatter(color_v, velocity.flatten())
#plt.show()
#
#v_bin, color_bin, _ = bin_mat(velocity, color_v, bin_width=bin_width)
#
#v_bin = v_bin.flatten()
#
#plt.plot(color_bin, v_bin)
#plt.show()

def plot_traj(time, colors):
    '''
    receive output from Diff_Drift.traj()
    '''
    for i in range(colors.shape[-1]): # loop over all batches
        plt.plot(time, colors[:, i])
    plt.show()

#def group_drift(model_dir, sub_dir, rule_name, bin_width=5, padding=5, prod_intervals=800, n_colors=720, input_color=None, sigma_x=0, sigma_rec=0):
#    group = Agent_group(model_dir, rule_name, sub_dir)
#    ddf = Diff_Drift()
#    for sub in group.group:
#        ddf.read_rnn_agent(sub)
#        color_bin, v_bin = ddf.drift(bin_width=None, padding=padding, prod_intervals=prod_intervals, n_colors=n_colors, input_color=input_color, sigma_x=sigma_x, sigma_rec=sigma_rec)
