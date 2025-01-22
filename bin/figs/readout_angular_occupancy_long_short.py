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
from scipy.stats import entropy
import pickle
import math
from brokenaxes import brokenaxes
import matplotlib as mpl
from core.ploter import plot_layer_boxplot_helper

def SET_MPL_FONT_SIZE(font_size):
    mpl.rcParams['axes.labelsize'] = font_size
    mpl.rcParams['ytick.labelsize'] = font_size
    mpl.rcParams['xtick.labelsize'] = font_size
    mpl.rcParams['xtick.labelsize'] = font_size
    mpl.rcParams['ytick.labelsize'] = font_size
    return
SET_MPL_FONT_SIZE(13)
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['legend.frameon'] = False


# os.environ["CUDA_VISIABLE_DEVICES"] = "1"

metric_name = 'cv'
prod_intervals = 800
n_colors = 500
batch_size = 36 # batch size for exact ring initial
pca_degree = np.linspace(0, 360, n_colors, endpoint=False) # Plot the trajectories of these colors
sigma_rec=0; sigma_x = 0
common_color = [40, 130, 220, 310]
density_bin_size = 8
sigma_s_list = [3.0, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 90.0]
period_name = 'response'
decoding_plane='response'
rule_name = 'color_reproduction_delay_unit' # rule name is fixed to color_reproduction_delay_unit. Actually this repo can also train another type of RNN with slightly different input format, but in this paper we only use color_reproduction_delay_unit


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

    report_color_ring = rnn_de.decode(hidden0_ring, decoding_plane=decoding_plane)
    deg = state_to_angle(hidden0_ring, pca=pca, state_type='data',verbose=False)  # (n_ring). Compute the angle of the ring states. using report_color_ring and deg one can then compute the angular occupation of color, see encode_space for more detail. The result should be x, mean_y in line 139 of encode_space.py

    return report_color_ring, deg

def diff_xy(x, y):
    cptor = Circular_operator(0, 360)
    diff_y = cptor.diff(y[1:], y[:-1])
    diff_x = cptor.diff(x[1:], x[:-1])

    dydx = abs(diff_y / diff_x) # the derivertive might be all negtive due to the difference of defination of rotational direction in deg_pca and report_color

    # reorder
    order = np.argsort(x[1:])
    x_order = x[1:][order]
    dydx_order = dydx[order]
    return x_order, dydx_order



def AO_metric(model_dir_root, sigma_s, metric_name='entropy'):
    model_dir_parent = model_dir_root+'/model_' + str(sigma_s) + '/color_reproduction_delay_unit/'
    metric_list = []

    sa = State_analyzer()
    for filename in os.listdir(model_dir_parent):
        print(sigma_s, filename)
        f = os.path.join(model_dir_parent, filename)
        sub = Agent(f, rule_name)
        sa.read_rnn_agent(sub)
        ########## Compute the angular ocupation
        report_color_ring, deg = gen_type_RNN(sub, batch_size=batch_size)
        x_delta = report_color_ring
        y_delta = deg
        x_delta, dydx_delta = diff_xy(x_delta, y_delta)
        if metric_name == 'entropy':
            metric_list.append(entropy(dydx_delta))
        elif metric_name == 'cv':
            metric_list.append(np.std(dydx_delta) / np.mean(dydx_delta))

    return metric_list


######### Calculte AO entropy for all sigma_s (original model)
'''
model_dir_root = '../core/model'
metric_all = []
for sigma_s in sigma_s_list:
    metric_sig = AO_metric(model_dir_root,sigma_s, metric_name=metric_name)
    metric_all.append(metric_sig)

with open('../bin/figs/fig_data/AO_'+metric_name+'_'+period_name+'_sigmas.txt', 'wb') as fp:
    pickle.dump(metric_all, fp)
'''

# Load data and plot the figure
'''
with open('../bin/figs/fig_data/AO_'+metric_name+'_'+period_name+'_sigmas.txt', 'rb') as fp:
    metric_all = np.array(pickle.load(fp))

metric_mean = list(np.mean(metric_all,axis=1))
metric_ste = list(np.std(metric_all, axis=1) / math.sqrt(metric_all.shape[1]))
metric_std = list(np.std(metric_all, axis=1))

fig = plt.figure(figsize=(4,3))
bax = brokenaxes(xlims=((0, 35), (85, 95)), hspace=.05)

bax.errorbar(x=sigma_s_list, y=metric_mean,yerr=metric_ste,color='orangered',fmt='.-',linewidth=1.5, markersize=10,alpha=1)
bax.set_ylabel(metric_name,fontsize=13)
bax.set_xlabel(r'$\sigma_s$',fontsize=15)
bax.axs[0].set_xticks([10,20,30])
bax.axs[0].set_xticklabels(['10.0','20.0','30.0'])
bax.axs[1].set_xticks([90])
bax.axs[1].set_xticklabels(['90.0'])
plt.savefig('../bin/figs/fig_collect/AO_'+metric_name+'_'+period_name+'_sigmas.svg',format='svg',bbox_inches='tight')
plt.show()
'''

######### Compare AO entropy of original model and short-response model (sigma_s = 3.0)
# '''
metric_ori = AO_metric('../core/model', sigma_s=3.0, metric_name=metric_name)
metric_short = AO_metric('../core/model_short_res_40', sigma_s=3.0, metric_name=metric_name)

score_exps = {'Long': metric_ori,'Short': metric_short}
layer_order = {'Long': 0,'Short': 1}
print(score_exps)

fig, ax = plt.subplots(figsize=(3, 3))
fig, ax = plot_layer_boxplot_helper(score_exps,layer_order, fig=fig, ax=ax, jitter_s=20, show_outlier=False)
ax.set_ylabel(metric_name + ' of \n the normalized angular occupancy')
fig.tight_layout()
fig.savefig('../bin/figs/fig_collect/long_short_AO_'+metric_name+'_'+period_name+'.svg',format='svg',bbox_inches='tight')
plt.show()

# '''
