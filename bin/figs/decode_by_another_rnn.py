import context
import os
from core.agent import Agent
import pickle
from scipy.stats import mannwhitneyu, ttest_ind
from core.net_struct.struct_analyzer import Struct_analyzer
import numpy as np
from core.rnn_decoder import RNN_decoder
from core.color_error import Color_error
import random
import matplotlib.pyplot as plt
from core.ploter import plot_layer_boxplot_helper
import matplotlib as mpl
import math

def SET_MPL_FONT_SIZE(font_size):
    mpl.rcParams['axes.labelsize'] = font_size
    mpl.rcParams['ytick.labelsize'] = font_size
    mpl.rcParams['xtick.labelsize'] = font_size
    mpl.rcParams['xtick.labelsize'] = font_size
    mpl.rcParams['ytick.labelsize'] = font_size
    return
SET_MPL_FONT_SIZE(12)
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['legend.frameon'] = False

######## Parameters
uniform_sigma = 90.0  # Sigma value for uniform RNNs.
biased_sigma = 12.5  # Sigma value for biased RNNs.
input_color = 40  # Input color for decoding.
batch_size = 50  # Number of trials.
prod_int = 800  # Duration of delay phase.
n_sampling = 50  # Number of bootstrapping samples.
sigma_rec = None; sigma_x = None # Default noise values.
sub_dir = 'noise_delta/'
rule_name = 'color_reproduction_delay_unit'


# Function to remove outliers using the interquartile range (IQR) method
def removeOutliers(a, outlierConstant=1.5):
    """
        Remove outliers from an array based on the IQR method.

        Parameters:
            a (array-like): Input array.
            outlierConstant (float): Multiplier for the IQR to define outlier thresholds.

        Returns:
            array-like: Filtered array with outliers removed.
    """
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant # 1.5
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    return a[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))]

# Function to align two group of neurons according their preferred inputs
def align_arrays(labels_1, labels_2):
    """
    Align two groups of neurons by reordering the second one to match the first one.

    Parameters:
        labels_1 (array-like): Preferred input of first group of neurons.
        labels_2 (array-like): Preferred input of first group of neurons.

    Returns:
        array-like: Indices for reordering the second group
    """
    order_1 = list(np.argsort(labels_1))
    rank_1 = [order_1.index(i) for i in range(len(labels_1))]
    order_2 = list(np.argsort(labels_2))
    rank_2 = [order_2.index(i) for i in range(len(labels_2))]
    indices = [list(rank_2).index(idx) for idx in list(rank_1)]
    return indices

# Function to perform cross-decoding (concatenate the delay and decoding of two models) and calculate memory error
def cross_decoding(delay_sig_s, decode_sig_s, delay_model=0, decode_model=0, input_color=40, batch_size=500):
    delay_model_dir_parent = "../core/model/model_{}/color_reproduction_delay_unit/".format(delay_sig_s)
    delay_model_dir = 'model_{}/'.format(delay_model)  # example delay RNN
    decode_model_dir_parent = "../core/model/model_{}/color_reproduction_delay_unit/".format(decode_sig_s)
    decode_model_dir = 'model_{}/'.format(decode_model)  # example decode RNN

    delay_f = os.path.join(delay_model_dir_parent, delay_model_dir, sub_dir)
    decode_f = os.path.join(decode_model_dir_parent, decode_model_dir, sub_dir)

    #### load the two RNNs (for delay and decoding, respectively)
    delay_sub = Agent(delay_f, rule_name)  # Load RNN (for delay).
    delay_sa = Struct_analyzer()
    delay_sa.read_rnn_agent(delay_sub)
    # Perform experiment and get preferred inputs of each neuron
    _, _, delay_neuron_label, _ = delay_sa.prepare_label(sigma_rec=0, sigma_x=0, method='rnn_decoder', label_neuron_by_mean=False, generate_state_method='delay_ring') # use label_by_mean=False to use tuning peaks not mean; use generate_state_method='delay_ring' to obtain densor state

    decode_sub = Agent(decode_f, rule_name)  # Load decoding RNN.
    decode_sa = Struct_analyzer()
    decode_sa.read_rnn_agent(decode_sub)
    # Perform experiment and get preferred inputs of each neuron
    _, _, decode_neuron_label, _ = decode_sa.prepare_label(sigma_rec=0, sigma_x=0, method='rnn_decoder', label_neuron_by_mean=False, generate_state_method='delay_ring')

    # Generate input colors for decoding
    input_colors = np.ones(batch_size)*input_color

    ######### Method 1: reorder the neuron states of delay
    align_indices = align_arrays(decode_neuron_label, delay_neuron_label)
    # quickly check whether two arrays are aligned in same order, if so, the two arrays should follow same order
    # print(np.sort(decode_neuron_label))
    # print((delay_neuron_label[align_indices])[np.argsort(decode_neuron_label)])
    delay_sub.do_exp(prod_intervals=prod_int, ring_centers=input_colors, sigma_rec=sigma_rec, sigma_x=sigma_x)
    delay_states = delay_sub.state[delay_sub.epochs['interval'][-1]]
    decode_states = delay_states[:,align_indices]  # Reorder the delay states to align with the decode RNN.
    rd = RNN_decoder()
    rd.read_rnn_file(decode_f,rule_name)

    ######### Method 2: reorder the weights of decoder
    # align_indices = align_arrays(delay_neuron_label, decode_neuron_label)
    # rd = RNN_decoder()
    # rd.read_rnn_file(decode_f,rule_name)
    # print(rde.sub.model.weight_hh.shape)
    # rd.sub.model.weight_hh= rde.sub.model.weight_hh[align_indices] # align weights
    # rd.sub.model.bias_hh= rde.sub.model.bias_hh[align_indices] # align bias

    ######## Calculate MSE
    out_colors = rd.decode(decode_states)
    color_error = Color_error()
    color_error.add_data(out_colors, input_colors)
    error = color_error.calculate_error()  # circular substraction
    error = removeOutliers(error)  # remove outliers
    mse = np.linalg.norm(error)**2 / len(error)

    return math.sqrt(mse)


# mse_uni_uni = []  # uniform delay and decode
# mse_bias_bias = []  # biased delay and decode
# mse_uni_bias = []  # uniform delay and biased decode
# mse_bias_uni = []  # biased delay and uniform decode

# for k in range(n_sampling):
#     print(k)
#     model_1, model_2 = random.sample(list(np.arange(50)),2)
#     mse_uni_uni.append(cross_decoding(uniform_sigma, uniform_sigma, model_1, model_2, batch_size=batch_size))
#     mse_bias_bias.append(cross_decoding(biased_sigma, biased_sigma, model_1, model_2, batch_size=batch_size))
#     mse_uni_bias.append(cross_decoding(uniform_sigma, biased_sigma, model_1, model_2, batch_size=batch_size))
#     mse_bias_uni.append(cross_decoding(biased_sigma, uniform_sigma, model_1, model_2, batch_size=batch_size))


# score_exps = {'biased delay state\n&\nbiased decoding': mse_bias_bias,
#               'uniform delay state\n&\nbiased decoding': mse_uni_bias,
#               'biased delay state\n&\nuniform decoding': mse_bias_uni,
#               'uniform delay state\n&\nuniform decoding': mse_uni_uni}
# layer_order = {'biased delay state\n&\nbiased decoding': 0,
#               'uniform delay state\n&\nbiased decoding': 1,
#               'biased delay state\n&\nuniform decoding': 2,
#               'uniform delay state\n&\nuniform decoding': 3}
# data = {'score_exps': score_exps, 'layer_order': layer_order}
# with open('../bin/figs/fig_data/cross_decoding.pkl', 'wb') as f:
#     pickle.dump(data, f)

# Load previously calculated cross-decoding results
with open('../bin/figs/fig_data/cross_decoding.pkl', 'rb') as f:
    data = pickle.load(f)
score_exps = data['score_exps']; layer_order = data['layer_order']

# Plot cross-decoding results as a box plot
fig, ax = plt.subplots(figsize=(3.2, 3.2))
fig, ax = plot_layer_boxplot_helper(score_exps,layer_order, ax=ax, fig=fig, show_outlier=True, jitter_s=20, jitter_color='grey')
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['Bias\n&\nBias', 'Uniform\n&\nBias', 'Bias\n&\nUniform', 'Uniform\n&\nUniform'])
ax.set_ylabel('Memory Error (color degree) \n input = common color')
fig.tight_layout()
fig.savefig('../bin/figs/fig_collect/cross_decoding.svg',format='svg',bbox_inches='tight')

# Mann-Whitney U test. Or one can also run the Wilcoxon signed-rank test (paired).
u_statistic_01, p_value_01 = mannwhitneyu(score_exps['biased delay state\n&\nbiased decoding'], score_exps['uniform delay state\n&\nbiased decoding'])
u_statistic_23, p_value_23 = mannwhitneyu(score_exps['biased delay state\n&\nuniform decoding'], score_exps['uniform delay state\n&\nuniform decoding'])
u_statistic_02, p_value_02 = mannwhitneyu(score_exps['biased delay state\n&\nbiased decoding'], score_exps['biased delay state\n&\nuniform decoding'])
u_statistic_13, p_value_13 = mannwhitneyu(score_exps['uniform delay state\n&\nbiased decoding'], score_exps['uniform delay state\n&\nuniform decoding'])

t_statistic_01, p_value_t_01 = ttest_ind(score_exps['biased delay state\n&\nbiased decoding'], score_exps['uniform delay state\n&\nbiased decoding'])
t_statistic_23, p_value_t_23 = ttest_ind(score_exps['biased delay state\n&\nuniform decoding'], score_exps['uniform delay state\n&\nuniform decoding'])
t_statistic_02, p_value_t_02 = ttest_ind(score_exps['biased delay state\n&\nbiased decoding'], score_exps['biased delay state\n&\nuniform decoding'])
t_statistic_13, p_value_t_13 = ttest_ind(score_exps['uniform delay state\n&\nbiased decoding'], score_exps['uniform delay state\n&\nuniform decoding'])

print(p_value_01, p_value_23)
print(p_value_02, p_value_13)
print(p_value_t_01, p_value_t_23)
print(p_value_t_02, p_value_t_13)

plt.show()
