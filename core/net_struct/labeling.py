# labeling neurons in RNN with theirs prefer color
import numpy as np
from core.net_struct.main import Bump_activity, bump_pipline, bin_fir, sc_dist, circular_mean
from core.rnn_decoder import RNN_decoder
import matplotlib.pyplot as plt
from core.tools import mean_se

class Labeler():
    '''
    The final output is a tuning curve self.tuning with shape (len(self.sense_color), len(self.label)), whose column are sense colors, row are neurons labeled be their prefer colors. A measurement for accessing if the neuron has accute tuning curve is given by t_strength (shape = self.label). Closer to 1 means the neuron would only fire in response to one color. Closer to 0 means the neuron might not use for encoding colors
    '''
    def read_rnn_file(self, model_dir, rule_name):
        # read in the RNN from file
        self.rule_name = rule_name
        self.sub = Agent(model_dir, rule_name)
        self.hidden_size = self.sub.hp['n_rnn']
        self.input_size = self.sub.hp['n_input']

    def read_rnn_agent(self, agent):
        # read in the RNN from agent
        self.sub = agent
        self.rule_name = self.sub.rule_name
        self.hidden_size = self.sub.hp['n_rnn']
        self.input_size = self.sub.hp['n_input']

    def label_delay(self, prod_intervals=20, n_colors = 180):
        '''
        labeling neurons according to the input color and the neural activity in the delay
        input:
          n_colors (int): number of colors equally spaced from 0 to 360 degree. This is for generating trials for labeling.
          prod_intervals (float): delay length from 20 to 2000.
        output:
        '''
        # repeat trials
        pca_degree = np.linspace(0, 360, n_colors, endpoint=False)
        fir_rate_list = []
        fir_rate, _, _ = self.sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=0.0, sigma_x=0.0)
        fir_rate_list.append(fir_rate)
        # mean firing rate
        fir_rate_list = np.concatenate(fir_rate_list).reshape(-1, *fir_rate.shape)
        fir_rate_mean = np.mean(fir_rate_list, axis=0) # mean activity in the delay

        # get the tunning matrix
        self.bump = Bump_activity()
        self.bump.fit(self.sub.behaviour['target_color'], fir_rate_mean, self.sub.epochs['interval'])
        self.bump._prefer_color()
        self.label, self.t_strength = self.bump.label, self.bump.t_strength

        self.tuning = self.bump.tuning
        self.sense_color = self.bump.input_colors
        self.label = self.bump.label
        self.t_strength = self.bump.t_strength

        return self.bump.label, self.bump.t_strength

    def do_exp(self, sigma=0.05, n_colors=180, batch_size=5, prod_intervals=20):
        '''
        do experiment and collect neural states
        input:
          sigma (float): noise in doing the experiment
          batch_size (int): the total number of trials is batch_size * number of colors
        output:
          state_list (n, hidden_size): n states each has possible different color information.
        '''
        pca_degree = np.linspace(0, 360, n_colors, endpoint=False)
        state_list = []
        for i in range(batch_size):
            self.sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=0.0, sigma_x=0.0)
            t0, t1 = self.sub.epochs['interval'][0], self.sub.epochs['interval'][1]

            state_list.append(self.sub.state[t0:t1].copy())
        state_list = np.concatenate(state_list) # the shape is (batch_size * time_points, n_input_colors, rnn_hidden_size)
        self.state_list = state_list.reshape(-1, self.hidden_size)
        return self.state_list

    def decode_data(self, state_list, step_size=1e4):
        '''
        decode states to colors
        input:
          state_list (array (n, hidden_size)): n states for decodin. if n > 1e4, we decode it seperaterally with each epoch decoding step_size batches
        output:
          colors (array, (n))
        '''
        rnn_de = RNN_decoder()
        rnn_de.read_rnn_agent(self.sub)
        n_parts = state_list.shape[0] // step_size + 1
        bins = np.linspace(0, state_list.shape[0], n_parts, endpoint=False)
        bins = np.append(bins, state_list.shape[0]) # add endpoint
        bins = bins.astype(int)
        self.report_color = []
        for i in range(bins.shape[0] - 1):
            report_color_temp = rnn_de.decode(state_list[bins[i]: bins[i + 1], :]) # the shape is (batch_size * time_points * n_input_colors)
            self.report_color.append(report_color_temp)

        self.report_color = np.concatenate(self.report_color)

        return self.report_color

    def compute_tuning(self, state_list, report_color, bin_width=2):
        '''
        input:
          state_list (n, hidden_size): n states each has possible different color information.
          report_color (n): decoded color by RNN decoder
        output:
          tuning (n_color, hidden_size): response of neurons to colors, where colors are bined from 0 to 360 with binwidth=binwidth
          sense_color (n_colors): colors for rows of tuning
        '''
        fir = np.tanh(state_list) + 1
        report_color = report_color // bin_width * bin_width + bin_width / 2 # bin colors that are similar (within binwidth)
        fir_T = np.transpose(fir)
        fir_bin, self.sense_color = bin_fir(fir_T, report_color, bin_width) 
        self.tuning = np.transpose(fir_bin) # shape of fir_bin is (label_bin, hidden_size)
        return self.tuning, self.sense_color


    def label_rnn_decoder(self, sigma=0.05, n_colors=180, batch_size=5, prod_intervals=20, bin_width=5):
        '''
        decode the neural state with RNN decoder. The decoded color is considered as the color information at that specific time. It then used as weight when calculating the prefer color for a neuron
        '''
        state_list = self.do_exp(sigma=sigma, batch_size=batch_size, prod_intervals=prod_intervals, n_colors=n_colors) # generate data
        report_color = self.decode_data(state_list) # decode data to color
        tuning, sense_color = self.compute_tuning(state_list, report_color, bin_width=bin_width) # neural response to different colors, where colors are bined
        label, t_strength = self.prefer_color(tuning, sense_color) # labeling neurons with prefer colors
        return label, t_strength

    def prefer_color(self, tuning, sense_color):
        '''
        labeling neurons with prefer colors
        input:
          tuning (n_color, hidden_size): response of neurons to colors, where colors are bined from 0 to 360 with binwidth=binwidth
          sense_color (n_colors): colors for rows of tuning
        '''
        # calculate the preference color
        self.label = []
        self.t_strength = []
        for j in range(tuning.shape[1]): # loop over all neurons
            pref_angle, norm = circular_mean(self.tuning[:, j], sense_color)
            self.label.append(pref_angle)
            self.t_strength.append(norm)

        self.label = np.array(self.label)
        self.t_strength = np.array(self.t_strength)

        return self.label, self.t_strength

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

def reorder_prefer(mat, label):
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

def array_pipline(array, label, t_strength=None, thresh=None, bin_width=5, sort=True):
    '''
    array (n, hidden_size): something related to neurons. This function will do the following procedures to this array
    
    1. cut the weak tuned neurons.
    2. sort neurons by the order of preferential color.
    3. bin similar neurons, the related quantities would be averged
    t_strength (array (n_neurons)): none then do not thresh
    '''
    if not (t_strength is None):
        label_temp = thre_weak_tuning(label, t_strength, thre=thresh)
        array_temp = thre_weak_tuning(array, t_strength, thre=thresh)
    else:
        label_temp = label.copy()
        array_temp = array.copy()

    if sort:
        array_temp, label_temp = reorder_prefer(array_temp, label_temp)

    if bin_width is None:
        array_pped, label_pped = array_temp, array_temp
    else:
        array_pped, label_pped = bin_fir(array_temp, label_temp, bin_width)

    return array_pped, label_pped

