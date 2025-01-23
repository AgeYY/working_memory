import context
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from core.color_manager import Degree_color
from core.agent import Agent, Agent_group
import sys

rule_name = 'color_reproduction_delay_unit'
model_dir = '../core/model/model_25.0/color_reproduction_delay_unit/'
sub_dir = '/noise_delta'

# prod_int_shorts = np.arange(0, 1100, 100)  # include 1000
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
    
    error_means.append(np.mean(error_color))
    error_stds.append(np.std(error_color))

# Plot results
plt.figure(figsize=(6, 4))
plt.errorbar(prod_int_shorts, error_means, yerr=error_stds, fmt='o-', capsize=5, color='black')
plt.xlabel('Delay (ms)')
plt.ylabel('Memory error (degree)')

# Save the plot
plt.savefig('./figs/fig_collect/error_vs_interval.pdf', format='pdf', bbox_inches='tight')

# Print numerical results
for interval, mean, std in zip(prod_int_shorts, error_means, error_stds):
    print(f"Interval: {interval}ms, Mean Error: {mean:.2f}°, Std: {std:.2f}°")
plt.show()