import context
from core.data_plot import color_reproduction_dly_lib as clib
from core.color_manager import Degree_color
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from core.agent import Agent, Agent_group
import pandas as pd
import sys

out_path_unbias = './figs/fig_data/unbias_common.csv'
out_path_bias = './figs/fig_data/bias_common.csv'

# Attempt to parse command-line arguments for model configuration, or use default values.
try:
    model_dir = sys.argv[1]
    rule_name = sys.argv[2]
    sub_dir = sys.argv[3]
except:
    rule_name = 'color_reproduction_delay_unit'
    model_dir = '../core/model/model_25.0/color_reproduction_delay_unit/'
    sub_dir = '/noise_delta'

# Determine whether to generate data; default is False.
try:
    if sys.argv[4] == 'Y': # set false so it will not generate data
        gen_data = True
    else:
        gen_data = False
except:
    gen_data = False

gen_data = True


# Parameters
batch_size = 1000  # Number of trials
prod_intervals = np.array([100, 1000])  # Delay intervals
common_color = [40, 130, 220, 310]  # Common colors with high probabilities in the training prior.
reg_up = 15; reg_low = -15  # Regression range for target colors around common colors.
sigma_rec=None; sigma_x=None  # Noise
bin_size = 2

# calculate the mean and se
def mean_se(x, y, epsilon = 1e-5):
    '''
    Calculate the mean and standard error for y values grouped by unique x values.

    x (array [float]): [0,0,0, 1,1,1, 2,2,2, ...] or [2.5,2.5,2.5, 2.6,2.6,2.6, ...]
    y (array [float]): the value y[i] is the y value of x[i]. Note that there are many repetitions in the x. This function will calculate the mean and se in every x value
    epsilon: the difference of x smaller than epsilon is considered as same value
    '''
    # unique works poorly in case x is float, we use epsilon to select the value. code comes from https://stackoverflow.com/questions/5426908/find-unique-elements-of-floating-point-array-in-numpy-with-comparison-using-a-d
    x = np.array(x); y = np.array(y)
    b = x.copy()
    b.sort()
    d = np.append(True, np.diff(b))
    target = b[d>epsilon]

    mean_y = []
    se_y  = []
    for uq in target:
        sub_list = y[np.abs(x - uq) < epsilon]
        mean_y.append(sub_list.mean())
        se_y.append(sub_list.std() / np.sqrt(sub_list.shape[0]))

    mean_y = np.array(mean_y)
    se_y = np.array(se_y)
    return target, mean_y, se_y

def output_data(sub_dir, out_path):
    """
       Generate and save bias data for target and reported colors around common colors.

       Parameters:
           sub_dir (str): Sub-directory for models.
           out_path (str): Path to save the output data.
       """
    group = Agent_group(model_dir, rule_name, sub_dir=sub_dir)
    print(model_dir, sub_dir)
    # Generate data for short delay intervals
    group.do_batch_exp(prod_intervals=prod_intervals[0], sigma_rec=sigma_rec, sigma_x=sigma_x, batch_size=batch_size, bin_size=bin_size)
    dire_s = group.group_behaviour.copy()

    # Generate data for long delay intervals
    group.do_batch_exp(prod_intervals=prod_intervals[1], sigma_rec=sigma_rec, batch_size=batch_size, bin_size=bin_size, sigma_x=sigma_x)
    dire_l = group.group_behaviour.copy()

    # Calculate difference around common colors
    target_common_s, bias_s = clib.bias_around_common(dire_s['report_color'], dire_s['target_color'], common_color)
    target_common_l, bias_l = clib.bias_around_common(dire_l['report_color'], dire_l['target_color'], common_color)

    # stack into dic
    dire_dic = {
        'target_common_s': target_common_s,
        'target_common_l': target_common_l,
        'bias_s': bias_s,
        'bias_l': bias_l
    }

    dire_df = pd.DataFrame(dire_dic)
    dire_df.to_csv(out_path)

if gen_data:
    output_data(sub_dir, out_path_bias)
    output_data('/noise', out_path_unbias)

#### Linear regression
import statsmodels.api as sm

# Perform linear regression for a given range
def gen_reg_line(target, bias):
    '''
    linear fitting for target within a range
    input:
      target could be simply dire_df['target_common_s'] which should be a pandas series. Also bias
      target (array): independent variable
      bias (array): dependent variable
      reg_up, reg_low: range of independent variable
    output:
      x_regs (array): independent variable
      y_regs (array): prediction
      p_value (array): pvalue
    '''
    X = target.to_numpy()
    X_idx = (X < reg_up) * (X > reg_low)  # Select data within the regression range.
    X = X[X_idx].reshape((-1, 1))
    X2 = sm.add_constant(X)
    y = bias.to_numpy()[X_idx]

    est = sm.OLS(y, X2)  # Ordinary Least Squares regression.
    est2 = est.fit()

    # data for regression line
    x_regs = np.linspace(reg_low, reg_up, 1000)
    y_regs = est2.params[0] + est2.params[1] * x_regs
    p_value = est2.summary2().tables[1]['P>|t|'][1]

    return x_regs, y_regs, p_value


sns.set()
sns.set_style("ticks")

def plot_bias(target_common_s, bias_s, target_common_l, bias_l):
    '''
    Plot bias data with regression lines and error bands.
    all inputs should be pd series
    '''
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0.2, 0.3, 0.63, 0.6])

    ### regression line
    x_s_regs, y_s_regs, p_value_s = gen_reg_line(target_common_s, bias_s)
    x_l_regs, y_l_regs, p_value_l = gen_reg_line(target_common_l, bias_l)

    ### Error band
    target, mean_error, se_error = mean_se(target_common_s, bias_s)
    ax.fill_between(target, mean_error - se_error, mean_error + se_error, alpha=0.4)

    target, mean_error, se_error = mean_se(target_common_l, bias_l)
    ax.fill_between(target, mean_error - se_error, mean_error + se_error, alpha=0.4)
    #### Plot figures
    ax.plot(x_s_regs, y_s_regs, label='0.2s')
    ax.plot(x_l_regs, y_l_regs, label='1.0s')

    ax.legend([ax.lines[0], ax.lines[1]], ['0.2s', '1.0s'], loc='upper right', frameon=False, handlelength=1.5)

    ax.set_xlim([-30, 30]) # there is end effect for calculating ci
    ax.set_ylim([-5, 5]) # there is end effect for calculating ci
    ax.set_xticks([-30, 0, 30])
    ax.set_yticks([-5, 0, 5])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, ax

#### Read the data, plot and save results for biased and unbiased RNNs

## bias_rnn, i.e. models trained with biased distribution
dire_df = pd.read_csv(out_path_bias)

fig, ax = plot_bias(dire_df['target_common_s'], dire_df['bias_s'], dire_df['target_common_l'], dire_df['bias_l'])
fig.savefig('./figs/fig_collect/bias_rnn.pdf', format='pdf')
#ax.set_xlabel('Target color (deg)')
#plt.show()

## unbias_rnn, i.e. models trained with uniform distribution
dire_df = pd.read_csv(out_path_unbias)

fig, ax = plot_bias(dire_df['target_common_s'], dire_df['bias_s'], dire_df['target_common_l'], dire_df['bias_l'])
#ax.set_xlabel('Target color (deg)')
#ax.set_ylabel('Report - Target (deg)')
fig.savefig('./figs/fig_collect/unbias_rnn.pdf', format='pdf')
#plt.show()
