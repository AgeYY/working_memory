# generate bump activity
import context
import matplotlib.pyplot as plt
import numpy as np
import sys
from core.net_struct.struct_analyzer import Struct_analyzer, array_pipline
from core.rnn_decoder import RNN_decoder

class bump_finder(Struct_analyzer):
    '''
    calculate one bump activity for one rnn
    '''
    def __init__(self, input_color=180, prod_intervals=2000, sigma_rec=None, sigma_x=None, delta_color=20, max_iter=2000):
        '''
        some experimental parameters
        input:
          input_color (float):
          prod_intervals (float): delay_length
          sigma_rec, sigma_x (float): noise in experiment
          delta_color (float): the difference between the begining of experiment and the end above delta_color so the iteration would stop
          max_iter (int): maximun iteration
        '''
        self.input_color = input_color
        self.prod_intervals = prod_intervals
        self.sigma_rec = sigma_rec
        self.sigma_x = sigma_x
        self.delta_color = delta_color
        self.max_iter = max_iter

    def do_one_exp(self):
        '''
        do experiment one time
        output:
          state_list ([float], [n_delay_time, n_neurons]: state_list in the delay epoch. Neurons are arranged in the original order.
        '''
        self.sub.do_exp(prod_intervals=self.prod_intervals, ring_centers=np.array(self.input_color), sigma_rec=self.sigma_rec, sigma_x=self.sigma_x)
        state_list = self.sub.state.copy()

        state_list = state_list[self.sub.epochs['interval'][0]: self.sub.epochs['interval'][1]]
        state_list = state_list.reshape((state_list.shape[0], state_list.shape[2])) # only one batch here

        return state_list

    def search_exceed_delta_color(self, verbose=True):
        for i in range(self.max_iter):
            state_list = self.do_one_exp()

            delay_start_state = np.mean(state_list[0:5, :], axis=0).reshape((1, -1))
            delay_end_state = np.mean(state_list[-4:, :], axis=0).reshape((1, -1))

            # use rnn decoder to decode colors
            rnn_de = RNN_decoder()
            rnn_de.read_rnn_agent(self.sub)

            delay_start_color = rnn_de.decode(delay_start_state)[0]
            delay_end_color = rnn_de.decode(delay_end_state)[0]
            exp_delta_color = abs(delay_start_color - delay_end_color)

            if verbose:
                print('start_color: {}'.format( delay_start_color))
                print('end_color: {}'.format( delay_end_color ))
                print('delta_color: {}'.format( exp_delta_color ))
                print('\n')
            if  exp_delta_color > self.delta_color:
                print('Searching bump succeed! \n start: {} \t end: {}'.format(delay_start_color, delay_end_color))
                break
        if i == (self.max_iter - 1): print('Searching bump failed!')
        return state_list

    def out_bump(self, state_list, thresh=None, bin_width=None, sort=True, nan_method='remove'):
        '''
        reordering neurons and convert state to firing. To execute this function, one must run super().prepare_label before.
        input:
          state_list (n_time, n_neuron): state in one experiment
          label (n_neuron): prefer color of neurons
        '''
        fir_rate = np.tanh(state_list) + 1
        fir_rate_pped, label_pped = array_pipline(fir_rate, self.label, t_strength=self.t_strength, thresh=thresh, bin_width=bin_width, sort=sort, nan_method=nan_method)

        return fir_rate_pped, label_pped

def bump_activity(sub, sigma_rec=None, delta_color=20, max_iter=2000, bin_width=5):
    bumpfinder = bump_finder(sigma_rec=sigma_rec, delta_color=delta_color, max_iter=max_iter)
    bumpfinder.read_rnn_agent(sub)
    state_list = bumpfinder.search_exceed_delta_color()
    bumpfinder.prepare_label()
    fir_rate, label = bumpfinder.out_bump(state_list, bin_width=bin_width)
    return fir_rate, label


