import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
from core.tools import removeOutliers

def compute_metric(data, metric_type='entropy', bins=None):
    """Compute either entropy or coefficient of variation for the data.
    
    Args:
        data: Input data array
        metric_type: 'entropy' or 'cv' (coefficient of variation)
        bins: Optional bins for histogram computation. If None, uses data directly.
    
    Returns:
        float: Computed metric value
    """
    if bins is not None:
        hist, _ = np.histogram(data, bins=bins, density=True)
        data = hist

    if metric_type == 'entropy':
        return entropy(data)
    else:  # coefficient of variation
        return np.std(data) / np.mean(data) if np.mean(data) != 0 else 0

def setup_plotting_style(fontsize=15, linewidth=2):
    """Set up common matplotlib plotting style."""
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('axes', linewidth=linewidth)

def create_broken_axis_plot(sigmas, metric_mean, metric_std, metric_name, 
                          figsize=(3,3), ylabel_fontsize=13, xlabel_fontsize=15):
    """Create a broken axis plot with error bars.
    
    Args:
        sigmas: x-axis values
        metric_mean: y-axis mean values
        metric_std: y-axis standard deviation values
        metric_name: Name of metric for ylabel ('entropy' or 'cv')
        figsize: Figure size tuple
        ylabel_fontsize: Font size for y-label
        xlabel_fontsize: Font size for x-label
    
    Returns:
        tuple: (fig, bax) matplotlib figure and broken axes objects
    """
    fig = plt.figure(figsize=figsize)
    bax = brokenaxes(xlims=((0, 35), (85, 95)), hspace=.05)
    
    bax.errorbar(x=sigmas, y=metric_mean, yerr=metric_std, 
                color='k', fmt='.-', linewidth=1.5, 
                markersize=15, alpha=1)
    
    ylabel = 'Entropy' if metric_name == 'entropy' else 'Coefficient of Variation'
    bax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    bax.set_xlabel(r'$\sigma_s$', fontsize=xlabel_fontsize)
    
    bax.axs[0].set_xticks([10,20,30])
    bax.axs[0].set_xticklabels(['10.0','20.0','30.0'])
    bax.axs[1].set_xticks([90])
    bax.axs[1].set_xticklabels(['90.0'])
    
    return fig, bax

def process_metric_data(metric_all, error_type='sem'):
    """Process metric data to compute mean and standard deviation.
    
    Args:
        metric_all: List of metric values for each sigma
    
    Returns:
        tuple: (metric_mean, metric_std) Lists of means and standard deviations
    """
    metric_all = [removeOutliers(x) for x in metric_all]
    metric_mean = [np.mean(x) for x in metric_all]
    if error_type == 'sem':
        metric_std = [np.std(x) / np.sqrt(len(x)) for x in metric_all]
    elif error_type == 'std':
        metric_std = [np.std(x) for x in metric_all]
    return metric_mean, metric_std 