import context
import sys
from core.rnn_decoder import RNN_decoder
from core.agent import Agent, Agent_group
import numpy as np
from core.manifold.fix_point import Fix_point_finder, Hidden0_helper
from core.data_plot.manifold_ploter import Manifold_ploter as MPloter
import matplotlib.pyplot as plt
from core.color_manager import Degree_color
from sklearn.decomposition import PCA
import pandas as pd
from core.color_error import Circular_operator
import seaborn as sns
from core.tools import find_nearest, mean_se

##################################################
# Handle command-line arguments or set defaults
try:
    model_dir = sys.argv[1]
    rule_name = sys.argv[2]
    sub_dir = sys.argv[3]
    out_dir = sys.argv[5]
except:
    rule_name = 'color_reproduction_delay_unit'
    model_dir = '../core/model/model_12.5/color_reproduction_delay_unit/'
    sub_dir = '/model_0/noise_delta'
    out_dir = './figs/fig_collect/angle_occupation'

try:
    if sys.argv[4] == 'Y': # set false so it will not generate data
        gen_data = True
    else:
        gen_data = False
except:
    gen_data = False

##################################################
# Check whether to generate data.
gen_data = True

# Parameters
hidden_size = 256  # Size of the RNN hidden layer.
prod_intervals = 800  # Delay duration for experiments.
n_colors = 500  # Number of colors for sampling the delay plane.
batch_size = n_colors  # Batch size for exact ring initial
pca_degree = np.linspace(0, 360, n_colors, endpoint=False) # Plot the trajectories of these colors
sigma_rec=0; sigma_x = 0  # Noise
common_color = [40, 130, 220, 310]  # Common colors
density_bin_size = 8  # Bin size for computing density.


def gen_type_RNN(sub):
    ''' Generate decoded colors and angular positions for one type of RNN '''
    ########## Points on the ring
    sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x)

    ##### fit data to find the pca plane
    n_components = 2
    pca = PCA(n_components=n_components)
    pca.fit(sub.state[sub.epochs['interval'][1]])
    
    ##### state in the hidimensional space and pca plane
    hidden_size = sub.state.shape[-1]
    hhelper = Hidden0_helper(hidden_size=hidden_size)
    hidden0_ring_pca, hidden0_ring = hhelper.delay_ring(sub, batch_size=300)
    #hidden0_ring_pca = pca.transform(hidden0_ring)
    
    ##### decode states from high dimesional space
    rnn_de = RNN_decoder()
    rnn_de.read_rnn_agent(sub)
    report_color_ring = rnn_de.decode(hidden0_ring)

    # Calculate angular positions on the PCA plane
    deg_pca = np.arctan2(hidden0_ring_pca[:, 1], hidden0_ring_pca[:, 0]) # calculate the angle of hidden0_ring_pca
    deg_pca = np.mod(deg_pca, 2*np.pi) / 2.0 / np.pi * 360.0
    return report_color_ring, deg_pca

if gen_data:
    sub = Agent(model_dir + sub_dir, rule_name)
    report_color_ring_delta, deg_pca_delta = gen_type_RNN(sub)

    #store the data
    data_df = pd.DataFrame({'report_color_ring_delta': report_color_ring_delta, 'deg_pca_delta': deg_pca_delta}).to_csv(out_dir + '.csv')

data_df = pd.read_csv(out_dir + '.csv')

# Function to compute derivatives (angular occupation)
def diff_xy(x, y):
    """
    Compute the derivative for angular occupation analysis.
    """
    cptor = Circular_operator(0, 360)
    diff_y = cptor.diff(y[1:], y[:-1])  # Differences in angular positions.
    diff_x = cptor.diff(x[1:], x[:-1])  # Differences in decoded colors.

    dydx = abs(diff_y / diff_x) # the derivertive might be all negtive due to the difference of defination of rotational direction in deg_pca and report_color

    # Reorder values by x for plotting
    order = np.argsort(x[1:])
    x_order = x[1:][order]
    dydx_order = dydx[order]
    return x_order, dydx_order

##### plot figure
sns.set_theme()
sns.set_style("ticks")

# Extract decoded colors and angular positions
x_delta = data_df['report_color_ring_delta'].to_numpy()
y_delta = data_df['deg_pca_delta'].to_numpy()

########## Original function
fig = plt.figure(figsize=(3, 3))
ax = fig.add_axes([0.25, 0.2, 0.6, 0.6])

# Color mapping for the points
deg_color = Degree_color()
colors = deg_color.out_color(x_delta, fmat='RGBA')

for i in range(len(x_delta) - 10):
    ax.scatter(x_delta[i], (y_delta[i] + 92) % 360, color=colors[i])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
    
ax.set_xticks([0, 180, 360])
ax.set_yticks([0, 180, 360])

ax.set_xlabel(r'Decoded color $\theta$')
ax.set_ylabel(r'Angle on PC plane $\phi$')

fig.savefig(out_dir + '_function.pdf', format='pdf')

########## Derivative
fig = plt.figure(figsize=(3, 3))
ax = fig.add_axes([0.25, 0.2, 0.6, 0.6])
group = Agent_group(model_dir, rule_name, sub_dir='/noise_delta')
x_delta_list, dydx_delta_list = [], []
for sub in group.group: # iterate all RNNs
    x_delta, y_delta = gen_type_RNN(sub)
    x_delta, dydx_delta = diff_xy(x_delta, y_delta)
    x_delta_list.append(x_delta)
    dydx_delta_list.append(dydx_delta)

x_delta = np.concatenate(x_delta_list, axis=None).flatten()
dydx_delta = np.concatenate(dydx_delta_list, axis=None).flatten()
x_delta = x_delta // density_bin_size * density_bin_size + density_bin_size / 2.0
x, mean_y, se_y = mean_se(x_delta, dydx_delta, remove_outlier=True, epsilon=0.1, sd=True)

#plot_figure(x_delta, dydx_delta, ax=ax)
ax.fill_between(x, mean_y + se_y, mean_y - se_y, alpha=0.5, color='tab:red')
ax.plot(x, mean_y, color='tab:red')

ax.set_xlabel('')
ax.set_ylabel('')
    
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
    
ax.set_xticks([0, 180, 360])
    
ax.set_xlabel(r'Color $\phi$')
ax.set_ylabel('Angular Occupation $d\\theta/d\\phi$ \n (angle degree / color degree)')
    
ax.set_ylim([0., 3.0])

ax.axhline(y = 1, linestyle = '--', linewidth = 1, color = 'k')
for cc_i in common_color:
    ax.axvline(x = cc_i, linestyle = '--', linewidth = 1, color = 'black')


fig.savefig(out_dir + '.svg', format='svg')

plt.show()
