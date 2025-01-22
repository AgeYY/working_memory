import context
import numpy as np
import pickle
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
from core.post_delay_metric_analysis import (
    compute_metric, setup_plotting_style, 
    create_broken_axis_plot, process_metric_data
)

# Set up plotting style
setup_plotting_style()

# Common parameters
sigmas = [3.0, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 90.0]
metric_name = 'cv'  # coefficient of variation

# Load go period drift data
with open('./figs/fig_data/drift_' + metric_name + '_go_cue_sigmas.txt', 'rb') as fp:
    _, metric_go_all = pickle.load(fp)

with open('./figs/fig_data/drift_' + metric_name + '_response_sigmas.txt', 'rb') as fp:
    _, metric_response_all = pickle.load(fp)

# Load angular occupancy data
with open('./figs/fig_data/AO_' + metric_name + '_response_sigmas.txt', 'rb') as fp:
    ao_metric_all = np.array(pickle.load(fp))

# Process the data
metric_go_mean, metric_go_std = process_metric_data(metric_go_all, error_type='std')
metric_go_mean = np.array(metric_go_mean) - metric_go_mean[-1]
metric_response_mean, metric_response_std = process_metric_data(metric_response_all, error_type='std')
metric_response_mean = np.array(metric_response_mean) - metric_response_mean[-1]
ao_mean, ao_std = process_metric_data(ao_metric_all, error_type='std')
ao_mean = np.array(ao_mean) - ao_mean[-1]

# Create figure with broken axis
fig = plt.figure(figsize=(3, 3))
bax = brokenaxes(xlims=((0, 35), (85, 95)), hspace=.05)

# Plot drift metrics
bax.errorbar(x=sigmas, y=metric_go_mean, yerr=metric_go_std, 
             color="tab:blue", fmt='.-', linewidth=1.5, markersize=8, label='Go', alpha=1)

# Plot angular occupancy metrics
bax.errorbar(x=sigmas, y=metric_response_mean, yerr=metric_response_std,
             color="tab:red", fmt='.-', linewidth=1.5, markersize=8, label='Response', alpha=1)
    
bax.errorbar(x=sigmas, y=ao_mean, yerr=ao_std,
             color="tab:green", fmt='.-', linewidth=1.5, markersize=8, label='Readout', alpha=1)

# Customize plot
bax.set_ylabel('CV$_\mathrm{biased}$ - CV$_\mathrm{uniform}$', fontsize=13)
bax.set_xlabel(r'$\sigma_s$', fontsize=15)
bax.axs[0].set_xticks([10,20,30])
bax.axs[0].set_xticklabels(['10','20','30'])
bax.axs[1].set_xticks([90])
bax.axs[1].set_xticklabels(['90'])
bax.legend(loc='upper right', frameon=False)

# Save figure
plt.savefig('./figs/fig_collect/combined_cv_effects.svg', format='svg', bbox_inches='tight')
plt.show()

# Print some statistics
print("\nSummary Statistics:")
print("\nGo Period Drift CV:")
print(f"Mean range: {min(metric_go_mean):.3f} to {max(metric_go_mean):.3f}")
print(f"Std range: {min(metric_go_std):.3f} to {max(metric_go_std):.3f}")
print("\nResponse CV:")
print(f"Mean range: {min(metric_response_mean):.3f} to {max(metric_response_mean):.3f}")
print(f"Std range: {min(metric_response_std):.3f} to {max(metric_response_std):.3f}")
print("\nReadout CV:")
print(f"Mean range: {min(ao_mean):.3f} to {max(ao_mean):.3f}")
print(f"Std range: {min(ao_std):.3f} to {max(ao_std):.3f}")
