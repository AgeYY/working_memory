import context
from brokenaxes import brokenaxes
from core.tools import removeOutliers
import os
import numpy as np
import matplotlib.pyplot as plt
from core.agent import Agent
from core.manifold.state_analyzer import State_analyzer
from core.tools import state_to_angle
from core.manifold.fix_point import Hidden0_helper
from core.rnn_decoder import RNN_decoder
from core.color_error import Circular_operator
from core.post_delay_metric_analysis import (
    compute_metric, setup_plotting_style, 
    create_broken_axis_plot, process_metric_data
)
import pickle

#################### Figure setting
setup_plotting_style()

# Parameters
prod_intervals = 800  # Duration of delay interval
n_colors = 500  # Number of colors used for RNN inputs
batch_size = 36  # batch size for exact ring initial
pca_degree = np.linspace(0, 360, n_colors, endpoint=False) # Plot the trajectories of these colors
sigma_rec=0; sigma_x = 0  # Noise
common_color = [40, 130, 220, 310]
density_bin_size = 8  # Bin size for density computation
sigma_s_list = [3.0, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 90.0]  # List of prior sigma_s
# sigma_s_list = [27.5, 90.0]  # List of prior sigma_s
period_name = 'response'
rule_name = 'color_reproduction_delay_unit'
metric_name = 'cv'  # can be 'entropy' or 'cv' (coefficient of variation)

def gen_type_RNN(sub,batch_size=300):
    ''' generate data for one type of RNN'''
    ########## Points on the ring
    sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x)

    ##### state in the hidimensional space and pca plane
    hidden_size = sub.state.shape[-1]
    hhelper = Hidden0_helper(hidden_size=hidden_size)
    hidden0_ring_pca, hidden0_ring, pca = hhelper.delay_ring(sub, batch_size=batch_size, period_name=period_name, return_pca=True)

    ##### decode states from high dimesional space
    rnn_de = RNN_decoder()
    rnn_de.read_rnn_agent(sub)

    report_color_ring = rnn_de.decode(hidden0_ring, decoding_plane='response')
    deg = state_to_angle(hidden0_ring, pca=pca, state_type='data',verbose=False)

    return report_color_ring, deg

def diff_xy(x, y):
    """
    Compute angular occupancy by calculating derivatives of the angle.
    Parameters:
        x: Color values.
        y: Neural axis angles.

    Returns:
        x_order: Sorted color values.
        dydx_order: Sorted angular occupancy values.
    """
    cptor = Circular_operator(0, 360)
    diff_y = cptor.diff(y[1:], y[:-1])
    diff_x = cptor.diff(x[1:], x[:-1])

    dydx = abs(diff_y / diff_x)

    # reorder
    order = np.argsort(x[1:])
    x_order = x[1:][order]
    dydx_order = dydx[order]
    return x_order, dydx_order

######### Calculation
metric_all = []
for sigma_s in sigma_s_list:
    metric_sig = []
    model_dir_parent = '../core/model/model_'+str(sigma_s)+'/color_reproduction_delay_unit/'

    sa = State_analyzer()
    for filename in os.listdir(model_dir_parent):
        print(sigma_s, filename)
        f = os.path.join(model_dir_parent, filename)
        sub = Agent(f, rule_name)
        sa.read_rnn_agent(sub)
        report_color_ring, deg = gen_type_RNN(sub,batch_size=batch_size)
        x_delta = report_color_ring
        y_delta = deg
        x_delta, dydx_delta = diff_xy(x_delta, y_delta)

        # Compute metric
        metric_value = compute_metric(dydx_delta, metric_type=metric_name)
        metric_sig.append(metric_value)

    metric_all.append(metric_sig)

# Save results
with open('./figs/fig_data/AO_' + metric_name + '_' + period_name + '_sigmas.txt', 'wb') as fp:
    pickle.dump(metric_all, fp)

######## Load data and plot the figure
with open('./figs/fig_data/AO_' + metric_name + '_' + period_name + '_sigmas.txt', 'rb') as fp:
    metric_all = np.array(pickle.load(fp))

metric_mean, metric_std = process_metric_data(metric_all, error_type='std')

# Create plot for start states
fig = plt.figure(figsize=(3,3))
bax = brokenaxes(xlims=((0, 35), (85, 95)), hspace=.05)

bax.errorbar(x=sigma_s_list , y=metric_mean, yerr=metric_std, 
             fmt='k.-', linewidth=1.5, markersize=8, label='End', alpha=1)
bax.axhline(y=metric_mean[-1], color='k', linestyle='--', alpha=0.5)

bax.axhline(y=metric_mean[-1], color='k', linestyle='--', alpha=0.5)
# Add shaded error band
bax.fill_between(sigma_s_list, metric_mean[-1] - metric_std[-1], metric_mean[-1] + metric_std[-1],
                 color='gray', alpha=0.2)

ylabel = 'CV of d$\\theta$/d$\\phi$ against $\\phi$'
bax.set_ylabel(ylabel, fontsize=13)
bax.set_xlabel(r'$\sigma_s$', fontsize=15)
bax.axs[0].set_xticks([10,20,30])
bax.axs[0].set_xticklabels(['10.0','20.0','30.0'])
bax.axs[1].set_xticks([90])
bax.axs[1].set_xticklabels(['90.0'])
bax.legend(loc='lower right', frameon=False)

plt.savefig('./figs/fig_collect/AO_' + metric_name + '_' + period_name + '_sigmas.svg',
            format='svg', bbox_inches='tight')
plt.show()
