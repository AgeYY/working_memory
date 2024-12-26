import context
import statsmodels.api as sm
from core.data_plot import color_reproduction_dly_lib as clib
from core.color_manager import Degree_color
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from core.agent import Agent, Agent_group
import pandas as pd
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

    ax.set_xlim([-30, 30])
    ax.set_ylim([-5, 5])
    ax.set_xticks([-30, 0, 30])
    ax.set_yticks([-5, 0, 5])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, ax, slope, ci

rule_name = 'color_reproduction_delay_unit'
model_dir = '../core/model/model_25.0/color_reproduction_delay_unit/'
sub_dir = '/noise_delta'
prod_intervals = np.concatenate([np.arange(0, 1800, 200), [1800]])
sigma_rec = None; sigma_x = None
batch_size = 1000
bin_size = 2
common_color = [40, 130, 220, 310] # high prob values
reg_up = 15; reg_low = -15; # regression range from common_color - 15 to common_color + 15

# Add function to compute slopes for all intervals
def compute_slopes_for_interval(prod_interval):
    group = Agent_group(model_dir, rule_name, sub_dir=sub_dir)
    group.do_batch_exp(prod_intervals=prod_interval, sigma_rec=sigma_rec, sigma_x=sigma_x, batch_size=batch_size, bin_size=bin_size)
    dire = group.group_behaviour.copy()
    
    target_common, bias = clib.bias_around_common(dire['report_color'], dire['target_color'], common_color)
    _, _, _, slope, (ci_low, ci_high) = gen_reg_line(target_common, bias, reg_up, reg_low)
    
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
ci_lows = np.array(ci_lows)
ci_highs = np.array(ci_highs)

# Plot the results
fig = plt.figure(figsize=(4, 3))
ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])

# Plot slope with error bars
ax.errorbar(prod_intervals, slopes, yerr=[slopes-ci_lows, ci_highs-slopes], 
            fmt='o-', capsize=2, markersize=4, color='black')

ax.set_xlabel('Production interval (ms)')
ax.set_ylabel('Slope')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save the results
fig.savefig('./figs/fig_collect/slopes_vs_interval.pdf', format='pdf')

# Print the final slopes and CIs
print("\nFinal results:")
for interval, slope, ci_l, ci_h in zip(prod_intervals, slopes, ci_lows, ci_highs):
    print(f"Interval {interval}ms - Slope: {slope:.4f}, 95% CI: [{ci_l:.4f}, {ci_h:.4f}]")

plt.show()