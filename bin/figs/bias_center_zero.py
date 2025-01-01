import context
import numpy as np
import scipy.stats as sci_st
from core.ploter import plot_layer_boxplot_helper
from core.tools import removeOutliers
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from core.color_manager import Degree_color
from core.agent import Agent, Agent_group
import sys

rule_name = 'color_reproduction_delay_unit'
model_dir = '../core/model/model_25.0/color_reproduction_delay_unit/'
sub_dir = '/noise_delta'

prod_int_short = 100
prod_int_long = 1000
batch_size = 1000
sigma_rec = None; sigma_x = None # set the noise to be default (training value)

fs = 10 # front size

#### Output data
def output_data(prod_intervals):
    group = Agent_group(model_dir, rule_name, sub_dir=sub_dir)
    group.do_batch_exp(prod_intervals=prod_intervals, sigma_rec=sigma_rec, batch_size=batch_size, sigma_x=sigma_x, sample_method=40)

    return group.group_behaviour['error_color']

error_color_short = output_data(prod_int_short)
error_color_long = output_data(prod_int_long)
error_color_short = error_color_short.reshape(-1, batch_size) # reshape, each row is a batch of one RNN
error_color_long = error_color_long.reshape(-1, batch_size) # reshape, each row is a batch of one RNN
error_color_short = np.mean(error_color_short, axis=1)
error_color_long = np.mean(error_color_long, axis=1)
error_color = {'Short': error_color_short, 'Long': error_color_long}

# Perform Wilcoxon signed-rank tests to compare against 0
stat_short, p_val_short = sci_st.wilcoxon(error_color_short, alternative='two-sided')
stat_long, p_val_long = sci_st.wilcoxon(error_color_long, alternative='two-sided')

print("\nStatistical Tests (Wilcoxon signed-rank test against 0):")
print(f"Short interval (0.2s):")
print(f"statistic: {stat_short:.3f}")
print(f"p-value: {p_val_short:.3e}")
print(f"Mean error: {np.mean(error_color_short):.3f}°")

print(f"\nLong interval (1.0s):")
print(f"statistic: {stat_long:.3f}")
print(f"p-value: {p_val_long:.3e}")
print(f"Mean error: {np.mean(error_color_long):.3f}°")

layer_order = {'Short': 0, 'Long': 1}

fig, ax = plt.subplots(figsize=(3, 3))
plot_layer_boxplot_helper(error_color, layer_order, ax=ax)
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.set_ylabel('Output color - Input color (degree)', fontsize=fs)
fig.savefig('./figs/fig_collect/bias_center_zero.pdf', format='pdf')
plt.show()


# #### read simulated data

# sns.set_theme()
# sns.set_style("ticks")

# def smooth_histogram(data, bins, sigma=5):
#     # Compute histogram
#     hist, _ = np.histogram(data, bins=bins, density=True)
    
#     # Gaussian kernel
#     window = np.exp(-(np.arange(-3*sigma, 3*sigma+1)**2)/(2*sigma**2))
#     window = window / window.sum()
    
#     # Smooth histogram
#     smoothed = np.convolve(hist, window, mode='same')
#     return smoothed
    

# #### Plot Simulation
# def plot_error_dist(error_color, legend=['Short'], ylim=[0, 7e-3], with_label=False):
#     fig = plt.figure(figsize=(3, 3))
#     ax = fig.add_axes([0.2, 0.3, 0.63, 0.6])

#     # Define bins and compute smoothed histogram
#     bins = np.linspace(-180, 180, 100)  # More bins for smoother curve
#     bin_centers = (bins[:-1] + bins[1:]) / 2
    
#     # Plot smoothed histograms
#     smoothed_short = smooth_histogram(np.array(error_color), bins)
#     ax.plot(bin_centers, smoothed_short, label='0.2s')

#     # remove label
#     ax.set_xlabel('')
#     ax.set_ylabel('')

#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(True)
#     ax.grid(False)
#     ax.set_xticks([-180, 0, 180])
#     ax.set_xticklabels(['-180', '0', '180'])
#     ax.set_yticks([0, ylim[1]])
#     ax.tick_params(direction='in')
    
#     ax.set_xlim([-180, 180])
#     ax.set_ylim(ylim)

#     ax.legend(frameon=False, handlelength=1.5)
#     plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

#     return fig, ax

# fig, ax = plot_error_dist(error_color, ylim=[0, 20e-3], legend=['0.2s'])
# ax.set_xlabel('Output color - Input color', fontsize=fs)
# ax.set_ylabel('Density', fontsize=fs)
# fig.savefig('./figs/fig_collect/gaussian_rnn.pdf', format='pdf')
# plt.show()