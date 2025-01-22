# compute the distribution of neural axis angle and preferred color using geometric method
import os
import context
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from core.agent import Agent
from core.manifold.state_analyzer import State_analyzer
from core.tools import state_to_angle
from core.manifold.fix_point import Fix_point_finder, Hidden0_helper
from core.rnn_decoder import RNN_decoder
from core.color_error import Circular_operator
from core.tools import find_nearest, mean_se
from sklearn.decomposition import PCA
# os.environ["CUDA_VISIABLE_DEVICES"] = "1"


# Set parameters for the experiment
prod_intervals = 800  # Delay period length
n_colors = 500  # Number of colors sampled
batch_size = 1000  # Batch size for exact ring initial
pca_degree = np.linspace(0, 360, n_colors, endpoint=False) # Plot the trajectories of these colors
sigma_rec=0; sigma_x = 0  # Noise
common_color = [40, 130, 220, 310]
density_bin_size = 8  # Bin size for density calculations
sigma_s = 3.0
period_name = 'response'

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
    deg = state_to_angle(hidden0_ring, pca=pca, state_type='data',verbose=False)  # (n_ring). Compute the angle of the ring states. using report_color_ring and deg one can then compute the angular occupation of color, see encode_space for more detail. The result should be x, mean_y in line 139 of encode_space.py

    return report_color_ring, deg

def diff_xy(x, y):
    """
    Calculate the derivative of angular occupation with respect to color.
    Args:
        x: Color input values.
        y: Angular positions of neural states.
    Returns:
        x_order: Ordered color inputs.
        dydx_order: Corresponding derivatives.
    """
    cptor = Circular_operator(0, 360)
    diff_y = cptor.diff(y[1:], y[:-1])
    diff_x = cptor.diff(x[1:], x[:-1])

    dydx = abs(diff_y / diff_x) # the derivertive might be all negtive due to the difference of defination of rotational direction in deg_pca and report_color

    # reorder
    order = np.argsort(x[1:])
    x_order = x[1:][order]
    dydx_order = dydx[order]
    return x_order, dydx_order

model_dir_parent = '../core/model/model_'+str(sigma_s)+'/color_reproduction_delay_unit/' # one rnn model
rule_name = 'color_reproduction_delay_unit'
angle_list = []
x_delta_list, dydx_delta_list = [], []
sa = State_analyzer()

# Iterate through each model
for filename in os.listdir(model_dir_parent):
    # if filename == 'model_0': pass
    # else: continue
    f = os.path.join(model_dir_parent, filename)
    sub = Agent(f, rule_name)
    sa.read_rnn_agent(sub)

    ########## Compute the angular ocupation ##########
    report_color_ring, deg = gen_type_RNN(sub,batch_size=batch_size)
    x_delta = report_color_ring
    y_delta = deg
    x_delta, dydx_delta = diff_xy(x_delta, y_delta)
    x_delta_list.append(x_delta)
    dydx_delta_list.append(dydx_delta)
    
###### plot the distribution of angular occupation
# '''
x_delta = np.concatenate(x_delta_list, axis=None).flatten()
dydx_delta = np.concatenate(dydx_delta_list, axis=None).flatten()
x_delta = x_delta // density_bin_size * density_bin_size + density_bin_size / 2.0
x, mean_y, se_y = mean_se(x_delta, dydx_delta, remove_outlier=True, epsilon=0.1, sd=True)

fig = plt.figure(figsize=(3, 3))
ax = fig.add_axes([0.25, 0.2, 0.6, 0.6])
ax.fill_between(x, mean_y + se_y, mean_y - se_y, alpha=0.5)
ax.axhline(y=1, linestyle='--',color='lightgrey')
ax.plot(x, mean_y)
ax.set_xlabel('')
ax.set_ylabel('')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
ax.set_xticks([0, 180, 360])
ax.set_xlabel(r'Color $\phi$')
ax.set_ylabel(r'$d\theta/d\phi$')
ax.set_ylim([0, 4.0])

ax.axhline(y = 1, linestyle = '--', linewidth = 1, color = 'k')
for cc_i in common_color:
    ax.axvline(x = cc_i, linestyle = '--', linewidth = 1, color = 'k')
fig.savefig('./figs/fig_collect/AO_{p}_{s}.svg'.format(p=period_name,s=sigma_s), format='svg')
plt.show()
# '''

