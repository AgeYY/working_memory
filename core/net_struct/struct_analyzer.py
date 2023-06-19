# structure connectivity matrix ordered in degree, one step update
import os
import context
import numpy as np
from core.agent import Agent, Agent_loader
from core.rnn_decoder import RNN_decoder
from core.manifold.fix_point import Hidden0_helper

class Labeler(Agent_loader):
    '''
    The final output is a tuning curve self.tuning with shape (len(self.sense_color), len(self.label)), whose column are sense colors, row are neurons labeled be their prefer colors. A measurement for accessing if the neuron has accute tuning curve is given by t_strength (shape = self.label). Closer to 1 means the neuron would only fire in response to one color. Closer to 0 means the neuron might not use for encoding colors
    '''
    def do_exp_by_trial(self, n_colors=360, sigma_rec=None, sigma_x=None, batch_size=2, prod_intervals=200):
        pca_degree = np.linspace(0, 360, n_colors, endpoint=False)

        state_list = []
        input_colors = []
        for i in range(batch_size):
            self.sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x)
            t0, t1 = self.sub.epochs['interval'][0], self.sub.epochs['interval'][1] # we only record the delay activity

            input_color_temp = self.sub.behaviour['target_color']
            input_color_temp = np.tile(input_color_temp, (t1-t0, 1))
            input_colors.append(input_color_temp)
            state_list.append(self.sub.state[t0:t1].copy())

        state_list = np.concatenate(state_list) # the shape is (batch_size * time_points, n_input_colors, rnn_hidden_size)

        self.state_list = state_list.reshape(-1, self.hidden_size) # for method == rnn decoder
        self.fir_rate = np.tanh(self.state_list) + 1

        self.input_colors = np.concatenate(input_colors) # input colors in experiment. This would be used if you wanna calculate the response matrix in delay method. Expand input colors
        self.input_colors = self.input_colors.flatten()

        return self.state_list, self.fir_rate, self.input_colors

    def do_exp_by_delay_ring(self, n_colors=360, sigma_rec=None, sigma_x=None, batch_size=2, prod_intervals=200):
        pca_degree = np.linspace(0, 360, n_colors, endpoint=False)
        self.sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x)

        n_states = n_colors * batch_size # sample n_states from the PC1-PC2 delay ring

        delay_ring_helper = Hidden0_helper(self.sub.model.hidden_size) # hidden_size is the number of recurrent neurons
        _, self.state_list = delay_ring_helper.delay_ring(self.sub, batch_size=n_states)
        self.fir_rate = np.tanh(self.state_list) + 1
        self.input_colors = self.label_input_color_rnn_decoder(self.state_list)

        return self.state_list, self.fir_rate, self.input_colors


    def do_exp(self, n_colors=360, sigma_rec=None, sigma_x=None, batch_size=2, prod_intervals=200, method='trial'):
        '''
        do experiments. The result would be used for further analysis.
        input:
          sigma_rec (float): recurrent noise. Set to None to use default value in RNN
          sigma_x (float): input noise. Set to None to use default value in RNN
          batch_size (int): the total number of trials is batch_size * number of colors
          n_colors (int): the input colors are equally spaced colors.
        output:
          state_list (n, hidden_size): n states each has possible different color information.
          fir_rate_list (n, hidden_size): n states each has possible different color information.
          input_colors (n, hidden_size): n states each has possible different color information.
        '''
        if method == 'trial':
            self.state_list, self.fir_rate, self.input_colors = self.do_exp_by_trial(n_colors=n_colors, sigma_rec=sigma_rec, sigma_x=sigma_x, batch_size=batch_size, prod_intervals=prod_intervals)
        elif method == 'delay_ring':
            self.state_list, self.fir_rate, self.input_colors = self.do_exp_by_delay_ring(n_colors=n_colors, sigma_rec=sigma_rec, sigma_x=sigma_x, batch_size=batch_size, prod_intervals=prod_intervals)

        return self.state_list, self.fir_rate, self.input_colors

    def label_input_color_delay_avg(self):
        # do nothing because in this case the input_color in the experiment are already the input_color for delay_avg mehtod
        return None

    def label_input_color_rnn_decoder(self, state_list, step_size=1e4):
        '''
        while input_colors in used for labeling neuron using average delay method, we can also decode every states, and label them by state color by RNN decoder.
        input:
          state_list (array (n, hidden_size)): n states for decodin. if n > 1e4, we decode it seperaterally with each epoch decoding step_size batches
        output:
          colors (array, (n))
        '''
        if state_list is None:
            state_list = self.state_list

        rnn_de = RNN_decoder()
        rnn_de.read_rnn_agent(self.sub)
        n_parts = state_list.shape[0] // step_size + 1
        bins = np.linspace(0, state_list.shape[0], int(n_parts), endpoint=False)
        bins = np.append(bins, state_list.shape[0]) # add endpoint
        bins = bins.astype(int)
        self.input_colors = []
        for i in range(bins.shape[0] - 1):
            input_color_temp = rnn_de.decode(state_list[bins[i]: bins[i + 1], :]) # the shape is (batch_size * time_points * n_input_colors)
            self.input_colors.append(input_color_temp)

        self.input_colors = np.concatenate(self.input_colors)

        return self.input_colors.copy()

    def bin_input_color(self, bin_width=5, nan_method='remove'):
        '''
        later we will labeling neurons using population vector, so you need to make sure the chance of every color occur is equal.
        '''
        fir_rate_T = np.transpose(self.fir_rate)
        self.fir_rate, self.input_colors, _ = bin_mat(fir_rate_T, self.input_colors, bin_width=bin_width, nan_method=nan_method)
        self.fir_rate = np.transpose(self.fir_rate)
        return self.fir_rate.copy(), self.input_colors.copy()

    def label_neuron(self, method='rnn_decoder', label_by_mean=True):
        '''
        labeling neurons by their prefer colors
        output:
          label ([float] (rnn_size)): the prefer color for unit i, where i < rnn_size
          t_strength ([float] (larger t means the correspond neuron are more perferent to a particular color))
        '''
        fir_rate = self.fir_rate.copy()
        input_colors = self.input_colors.copy()

        #fir_rate_mean = np.mean(self.fir_rate, axis = 0) # mean over the delay period

        # calculate the preference color
        self.label = []
        self.t_strength = []
        for j in range(fir_rate.shape[1]): # loop over all neurons
            pref_angle, norm = circular_mean(fir_rate[:, j], input_colors)
            self.label.append(pref_angle)
            self.t_strength.append(norm)

        if label_by_mean:
            self.label = np.array(self.label)
        else:
            label_idx = np.argmax(fir_rate, axis=0)
            self.label = input_colors[label_idx]

        self.t_strength = np.array(self.t_strength)

        return self.label.copy(), self.t_strength.copy()

def circular_mean(weights, angles):
    weight_norm = weights / np.sum(weights)
    x = np.dot(np.cos(np.radians(angles)) , weight_norm)
    y = np.dot(np.sin(np.radians(angles)) , weight_norm)

    mean = np.degrees(np.arctan2(y, x)) % 360 # set to positive
    norm = np.linalg.norm([x, y])
    return mean, norm

def array_pipline(array, label, t_strength=None, thresh=None, bin_width=5, sort=True, nan_method='remove', avg_method='bin'):
    '''
    array (n, hidden_size): something related to neurons. This function will do the following procedures to this array
    
    1. cut the weak tuned neurons.
    2. sort neurons by the order of preferential color.
    3. bin similar neurons, the related quantities would be averged
    t_strength (array (n_neurons)): none then do not thresh
    '''
    if not (thresh is None):
        label_temp = thre_weak_tuning(label, t_strength, thre=thresh)
        array_temp = thre_weak_tuning(array, t_strength, thre=thresh)
    else:
        label_temp = label.copy()
        array_temp = array.copy()

    if sort:
        array_temp, label_temp = sort_mat(array_temp, label_temp)

    if bin_width is None:
        array_pped, label_pped = array_temp, label_temp
    else:
        array_pped, label_pped, _ = bin_mat(array_temp, label_temp, bin_width, nan_method=nan_method, avg_method=avg_method)

    return array_pped, label_pped

def bin_mat(mat, label, bin_width, nan_method='remove', avg_method='bin'):
    '''
    averge the columns of mat by their labels.
    input:
      mat ([float] (n, n_labels))
      label ([float] (n_labels))
      bin_width (int): if its 5 then the neuron label would be its example [0, 5, 10, ..., 355]
      avg_method (str 'bin' or 'gaussian'):
      nan_method (str or float):
        float -- fill in nan with the value
        'remove': remove columns contains nan
        If there are no labels with one label_bin, python will report runtimewarning: mean of empty slice, the result value for mat_bin would be nan. this nan_method is used for dealing such case.
    output:
      mat_bin ([float] (n, n_bin)): where n_bin = 360 / bin_width
      label_bin ([float] (n_bin)): example [2.5, 7.5, 12.5, ..., 357.5]
      nan_args (array [bool], [mat_bin.shape[1]]): true if the column contains nan. This can help you know which columns is removed.
    '''
    #avg_method='gaussian'

    label_bin = np.arange(0, 360, bin_width)
    label_bin_ex = np.append(label_bin, [360])
    n_bin = len(label_bin)
    mat_bin = np.zeros((mat.shape[0], n_bin))
    for i in range(len(label_bin)):
        if avg_method == 'bin':
            col_idx = (label < label_bin_ex[i + 1]) * (label > label_bin_ex[i])
            mat_bin[:, i] = np.mean( mat[:, col_idx], axis=1 )
        elif avg_method == 'gaussian':
            weight = np.exp( -(label - label_bin[i])**2 / 2.0 / (bin_width / 2.0)**2 )
            denominator = mat @ weight
            nomalization = np.sum(weight)
            mat_bin[:, i] = denominator / nomalization

    label_bin = label_bin + bin_width / 2 # use the center

    # dealing with nan
    nan_args = np.sum(np.isnan(mat_bin), axis=0)

    if nan_method=='remove':
        mat_bin = mat_bin[:, np.logical_not(nan_args)]
        label_bin = label_bin[np.logical_not(nan_args)]
    else:
        mat_bin = np.nan_to_num(mat_bin, nan_method)

    return mat_bin, label_bin, nan_args

def thre_weak_tuning(mat, t_strength, thre=-1):
    '''
    delete non preference neurons
    input:
        mat ([float] (n, num_neurons), or [float] (num_neurons) ): n can be any interger larger than 0. the num_neurons must be the same with length of label.
       t_strength (array (num_neuron)): tuning strength of each neuron in mat
    output:
        mat_cut ([float] (n, num_neurons_cut) or [float] (num_neurons_cut) ): num_neurons_cut is the number of neurons which is stronger than the threshold
    '''
    strong_t = t_strength > thre

    if len(mat.shape) == 1:
        mat_cut = mat[strong_t]
    elif len(mat.shape) == 2:
        mat_cut = mat[:, strong_t]
    return mat_cut

def sort_mat(mat, label):
    '''
    reordering mat accoring the the label

    input:
        mat ([float] (n, rnn_size)): reordering the tensor according to the find axis, so that mat[,,,0] indicate the features of neuron has preference in self.input_colors[0]
    output:
        label_sorted:
        mat_sorted: same shape as above
    '''
    order = np.argsort(label)
    mat_soted = mat[:, order]
    label_sorted = label[order]
    return mat_soted, label_sorted

class Struct_analyzer(Labeler):
    '''
    input a agent, output its weight and bias array with various forms
    '''
    def prepare_label(self, n_colors=720, sigma_rec=None, sigma_x=None, batch_size=1, prod_intervals=200, method='rnn_decoder', bin_width_color=8, nan_method='remove', generate_state_method='trial', label_neuron_by_mean=True):
        '''
        The final output is a tuning curve self.tuning with shape (len(self.sense_color), len(self.label)), whose column are sense colors, row are neurons labeled be their prefer colors. A measurement for accessing if the neuron has accute tuning curve is given by t_strength (shape = self.label). Closer to 1 means the neuron would only fire in response to one color. Closer to 0 means the neuron might not use for encoding colors
        input:
          sigma (float): noise in doing the experiment
          batch_size (int): the total number of trials is batch_size * number of colors
          bin_width_color : labeling neurons by population vector, so you need to make sure the chance of every color occur is equal. This is acheved by aveger input colors within small bin
          generate_state_method (str): methods of generating firing rate profile. delay_ring means draw a ring in pc1-pc2. trial means using different stimulus color
          label_neuron_by_mean: bool. label neuron by mean or max of the tuning curves.
        output:
          state_list (n, hidden_size): n states each has possible different color information.
        '''
        if generate_state_method == 'delay_ring':
            super().do_exp_by_delay_ring(n_colors=n_colors, sigma_rec=sigma_rec, sigma_x=sigma_x, batch_size=batch_size, prod_intervals=prod_intervals) # generated state will be automatically decoded by rnn decoder

        elif generate_state_method == 'trial':
            super().do_exp_by_trial(n_colors=n_colors, sigma_rec=sigma_rec, sigma_x=sigma_x, batch_size=batch_size, prod_intervals=prod_intervals)
            if method=='rnn_decoder':
                super().label_input_color_rnn_decoder(self.state_list)
            elif method=='delay_avg' :
                super().label_input_color_delay_avg() # the name is misleading
            else:
                os.abort('method for decoding state can only be rnn_decoder or delay_avg')
        else:
            os.abort('method for generating states can only be delay_ring or trial')

        super().bin_input_color(bin_width=bin_width_color, nan_method=nan_method)

        super().label_neuron(method=method, label_by_mean=label_neuron_by_mean)

        return self.fir_rate, self.input_colors, self.label, self.t_strength


    def output_bias(self, thresh=None, sort=None, bin_width=None, nan_method='remove', avg_method='gaussian'):
        '''
        1. Obtain bias weight from rnn
        2. thresh, sort and bin bias weight according to the label
        '''
        bias_hh = self.sub.model.bias_h.detach().cpu().numpy()
        bias_hh = bias_hh.reshape(1, -1)
        bias_pped, label_pped = array_pipline(bias_hh, self.label, t_strength=self.t_strength, thresh=thresh, bin_width=bin_width, sort=sort, nan_method=nan_method, avg_method=avg_method)

        return bias_pped, label_pped


    def output_weight(self, thresh=None, bin_width=None, sort=True, nan_method='remove', avg_method='gaussian'):
        '''
        1. Obtain rnn weight
        2. thresh, sort and bin bias weight according to the label
        '''
        weight_hh = self.sub.model.weight_hh.detach().cpu().numpy()
        weight_hh, label_pped = array_pipline(weight_hh, self.label, t_strength=self.t_strength, thresh=thresh, bin_width=bin_width, sort=sort, nan_method=nan_method, avg_method=avg_method)
        weight_hh_pped, label_pped = array_pipline(np.transpose(weight_hh), self.label, t_strength=self.t_strength, thresh=thresh, bin_width=bin_width, sort=sort, nan_method=nan_method, avg_method=avg_method)
        weight_hh_pped = np.transpose(weight_hh_pped)

        return weight_hh_pped, label_pped

    def output_tuning(self, thresh=None, bin_width_neuron=None, sort=True, bin_width_color=None, method='rnn_decoder', nan_method='remove'):

        fir_rate_list = self.fir_rate.copy()

        fir_rate_neuron_pped, label_pped = array_pipline(fir_rate_list, self.label, t_strength=self.t_strength, thresh=thresh, bin_width=bin_width_neuron, sort=sort, nan_method=nan_method)
        fir_rate_pped, color_pped = array_pipline(np.transpose(fir_rate_neuron_pped), self.input_colors, t_strength=None, thresh=None, bin_width=bin_width_color, sort=sort, nan_method=nan_method) # colors, we do not threshold
        fir_rate_pped = np.transpose(fir_rate_pped)
        return fir_rate_pped, label_pped, color_pped

    #def output_bump(self, thresh=None, bin_width=None, sort=True, nan_method='remove'):
    #    '''
    #    bump activity
    #    '''
    #    self.sub.do_exp()
