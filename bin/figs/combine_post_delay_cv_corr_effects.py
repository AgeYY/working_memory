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
    metric_start_all, metric_end_all = pickle.load(fp)

# Load angular occupancy data
with open('./figs/fig_data/AO_' + metric_name + '_response_sigmas.txt', 'rb') as fp:
    ao_metric_all = np.array(pickle.load(fp))

# Process the data
drift_start_mean, drift_start_std = process_metric_data(metric_start_all, error_type='std')
drift_end_mean, drift_end_std = process_metric_data(metric_end_all, error_type='std')
ao_mean, ao_std = process_metric_data(ao_metric_all, error_type='std')

# Create figure with broken axis
fig = plt.figure(figsize=(8, 4))
bax = brokenaxes(xlims=((0, 35), (85, 95)), hspace=.05)

# Plot drift metrics
bax.errorbar(x=sigmas, y=drift_end_mean, yerr=drift_end_std, 
             fmt='b.-', linewidth=1.5, markersize=8, label='Go-End CV', alpha=1)

# Plot angular occupancy metrics
bax.errorbar(x=sigmas, y=ao_mean, yerr=ao_std,
             fmt='r.-', linewidth=1.5, markersize=8, label='Response AO', alpha=1)

# Add reference lines
bax.axhline(y=drift_end_mean[-1], color='b', linestyle='--', alpha=0.5)
bax.axhline(y=ao_mean[-1], color='r', linestyle='--', alpha=0.5)

# Customize plot
bax.set_ylabel('Coefficient of Variation (CV)', fontsize=13)
bax.set_xlabel(r'$\sigma_s$', fontsize=15)
bax.axs[0].set_xticks([10,20,30])
bax.axs[0].set_xticklabels(['10.0','20.0','30.0'])
bax.axs[1].set_xticks([90])
bax.axs[1].set_xticklabels(['90.0'])
bax.legend(loc='upper right', frameon=False)

# Save figure
plt.savefig('./figs/fig_collect/combined_cv_effects.svg', format='svg', bbox_inches='tight')
plt.show()

# Print some statistics
print("\nSummary Statistics:")
print("\nGo Period Drift CV:")
print(f"Mean range: {min(drift_end_mean):.3f} to {max(drift_end_mean):.3f}")
print(f"Std range: {min(drift_end_std):.3f} to {max(drift_end_std):.3f}")

print("\nAngular Occupancy CV:")
print(f"Mean range: {min(ao_mean):.3f} to {max(ao_mean):.3f}")
print(f"Std range: {min(ao_std):.3f} to {max(ao_std):.3f}")

# Calculate correlations between metrics
correlations = np.corrcoef([drift_end_mean, ao_mean])[0,1]
print(f"\nCorrelation between Go-End CV and Response AO: {correlations:.3f}")
