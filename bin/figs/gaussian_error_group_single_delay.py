import context
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from core.color_manager import Degree_color
from core.agent import Agent, Agent_group
import core.tools as tools
import sys
from scipy import stats

rule_name = 'color_reproduction_delay_unit'
model_dir = '../core/model/model_25.0/color_reproduction_delay_unit/'
sub_dir = '/noise_delta'

prod_int_shorts = np.arange(100, 1200, 200)  # include 1000
batch_size = 500
sigma_rec = None; sigma_x = None

def removeOutliers(a, outlierConstant=1.5):
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    return a[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))]

# Lists to store results
error_means = []
error_stds = []
all_errors = []  # New list to store all error values
all_delays = []  # New list to store corresponding delays

#### Generate data for each production interval
group = Agent_group(model_dir, rule_name, sub_dir=sub_dir)

for prod_int in prod_int_shorts:
    group.do_batch_exp(prod_intervals=prod_int, sigma_rec=sigma_rec, batch_size=batch_size, sigma_x=sigma_x)
    dire_df = pd.DataFrame(group.group_behaviour)
    
    diff_color = np.array(dire_df['error_color'])
    diff_color = diff_color.reshape(len(group.group), -1)
    
    # Remove outliers row by row
    diff_color_clean = np.array([removeOutliers(row) for row in diff_color])
    
    # Calculate std for each network
    error_color = [np.sqrt(np.mean(row**2)) for row in diff_color_clean]
    
    # Store all individual errors and their corresponding delays
    all_errors.extend(error_color)
    all_delays.extend([prod_int] * len(error_color))
    
    error_means.append(np.mean(error_color))
    error_stds.append(np.std(error_color) / np.sqrt(len(error_color))) # this is the standard error of the mean

# Convert to numpy arrays for correlation analysis
all_errors = np.array(all_errors)
all_delays = np.array(all_delays)

# Calculate correlation and p-value
correlation, p_value = stats.pearsonr(all_delays, all_errors)
print('correlation, p_value:', correlation, p_value)

# Plot results
# Create figure and axis
fig = plt.figure(figsize=(3, 3))
ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])

# Plot data
ax.errorbar(prod_int_shorts, error_means, yerr=error_stds, 
            fmt='o-', capsize=2, markersize=4, color='black')

# Customize axis
ax.set_xlabel('Delay length (ms)')
ax.set_ylabel('Memory error (degree)')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set ticks
# ax.set_xticks([200, 600, 1000])

# Save figure
fig.savefig('./figs/fig_collect/error_vs_interval.svg', format='svg')

plt.show()