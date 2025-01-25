import context
import statsmodels.api as sm
from scipy import stats
from core.data_plot import color_reproduction_dly_lib as clib
from core.color_manager import Degree_color
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from core.agent import Agent, Agent_group
import pandas as pd
import core.tools as tools
import sys

# calculate the mean and se
def mean_se(x, y, epsilon = 1e-5):
    '''
    calculate the mean and standard error of the mean
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

def gen_reg_line(target, bias, reg_up, reg_low):
    '''
    linear fitting for target within a range
    '''
    X = target
    X_idx = (X < reg_up) * (X > reg_low)
    X = X[X_idx].reshape((-1, 1))
    X2 = sm.add_constant(X)
    y = bias[X_idx]

    est = sm.OLS(y, X2)
    est2 = est.fit()

    # Get slope and confidence intervals
    slope = est2.params[1]
    ci = est2.conf_int(alpha=0.05)
    slope_ci_low, slope_ci_high = ci[1]

    x_regs = np.linspace(reg_low, reg_up, 1000)
    y_regs = est2.params[0] + est2.params[1] * x_regs
    p_value = est2.summary2().tables[1]['P>|t|'][1]

    return x_regs, y_regs, p_value, slope, (slope_ci_low, slope_ci_high)

def gen_reg_line_batch(target_common, bias, reg_up, reg_low):
    '''
    Similar to gen_reg_line but handles batches of data by calling gen_reg_line for each batch
    Input:
        target_common: array of shape (n_batch, n)
        bias: array of shape (n_batch, n) 
        reg_up: upper bound for regression
        reg_low: lower bound for regression
    Output:
        x_regs: array for plotting regression line
        y_regs_batch: regression lines for each batch
        p_values: p-values for each batch
        slopes: slopes for each batch
        cis: confidence intervals for each batch
    '''
    n_batch = target_common.shape[0]
    y_regs_batch = []
    p_values = []
    slopes = []
    cis = []
    x_regs = None

    for i in range(n_batch):
        x_regs_i, y_regs, p_value, slope, ci = gen_reg_line(
            target_common[i], bias[i], reg_up, reg_low
        )
        if x_regs is None:
            x_regs = x_regs_i
        y_regs_batch.append(y_regs)
        p_values.append(p_value)
        slopes.append(slope)
        cis.append(ci)

    return x_regs, np.array(y_regs_batch), np.array(p_values), np.array(slopes), np.array(cis)

def plot_bias(target_common, bias, reg_up, reg_low):
    '''
    Modified to plot only one condition
    '''
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0.2, 0.3, 0.63, 0.6])

    ### regression line
    x_regs, y_regs, p_value, slope, ci = gen_reg_line(target_common, bias, reg_up, reg_low)

    ### band
    target, mean_error, se_error = mean_se(target_common, bias)
    ax.fill_between(target, mean_error - se_error, mean_error + se_error, alpha=0.4)

    #### Plot figures
    ax.plot(x_regs, y_regs, label='1.0s')
    ax.legend([ax.lines[0]], ['1.0s'], loc='upper right', frameon=False, handlelength=1.5)

    ax.set_xlim([-45, 45])
    # ax.set_ylim([-5, 5])
    ax.set_xticks([-45, 0, 45])
    # ax.set_yticks([-5, 0, 5])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, ax, slope, ci

rule_name = 'color_reproduction_delay_unit'
model_dir = '../core/model/model_25.0/color_reproduction_delay_unit/'
sub_dir = '/noise_delta'
prod_intervals = np.arange(100, 1200, 200)
# prod_intervals = np.array([100, 600, 800, 1000])
sigma_rec = None; sigma_x = None
batch_size = 5000
bin_size = 2
common_color = [40, 130, 220, 310] # high prob values
reg_up = 15; reg_low = -15; # regression range from common_color - 15 to common_color + 15

# Add function to compute slopes for all intervals
def compute_slopes_for_interval(prod_interval):
    group = Agent_group(model_dir, rule_name, sub_dir=sub_dir)
    group.do_batch_exp(prod_intervals=prod_interval, sigma_rec=sigma_rec, sigma_x=sigma_x, batch_size=batch_size, bin_size=bin_size)
    dire = group.group_behaviour.copy()

    report_color = dire['report_color'].reshape(-1, batch_size)
    target_color = dire['target_color'].reshape(-1, batch_size)
    target_common, bias = clib.bias_around_common(report_color, target_color, common_color)

    _, _, _, slope, ci = gen_reg_line_batch(target_common, bias, reg_up, reg_low)

    ci_low = ci[:, 0]
    ci_high = ci[:, 1]
    
    return slope, ci_low, ci_high

# Compute slopes and CIs for all intervals
slopes = []
ci_lows = []
ci_highs = []

print("Computing slopes for all intervals...")
for interval in prod_intervals:
    print(f"Processing interval {interval}ms")
    slope, ci_low, ci_high = compute_slopes_for_interval(interval)
    slopes.append(slope)
    ci_lows.append(ci_low)
    ci_highs.append(ci_high)

# Convert to numpy arrays
slopes = np.array(slopes)

# Compute correlation between slopes and production intervals

# Repeat prod_intervals to match number of subjects
num_subjects = slopes.shape[1]
repeated_intervals = np.repeat(prod_intervals[:, np.newaxis], num_subjects, axis=1).flatten()

# Flatten slopes array to match repeated intervals
flattened_slopes = slopes.flatten()

# Calculate correlation and p-value
correlation, pvalue = stats.pearsonr(repeated_intervals, flattened_slopes)

print("\nCorrelation Analysis:")
print(f"Correlation coefficient: {correlation:.3f}")
print(f"P-value: {pvalue:.3f}")

# Plot the results
fig = plt.figure(figsize=(3, 3))
ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])

# Calculate mean and std across subjects (axis 1)
mean_slopes = np.mean(slopes, axis=1)
se_slopes = np.std(slopes, axis=1) / np.sqrt(slopes.shape[1])

# Plot slope with error bars
ax.errorbar(prod_intervals, mean_slopes, yerr=se_slopes,
            fmt='o-', capsize=2, markersize=4, color='black')

ax.set_xlabel('Delay (ms)')
ax.set_ylabel('Slope')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save the results
fig.savefig('./figs/fig_collect/slopes_vs_interval.svg', format='svg')

plt.show()