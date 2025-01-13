# For one example RNN, this shows how the intially uniformly distributed neural states changed over time in go and response epoch
import context
import numpy as np
import os
from core.manifold.fix_point import Hidden0_helper
from sklearn.decomposition import PCA
from core.agent import Agent
from core.state_evolver import State_Evolver
import matplotlib.pyplot as plt


def obtain_x_y_lim(cords_pca, padding_factor=0.1):
    '''
    Obtain the x and y limit which is slightly larger than the range of the data
    input:
        cords_pca: np.array. shape: (n, 2)
        padding_factor: float. The padding factor of the x and y range
    output:
        xlim, ylim: tuple. (min, max)
    '''
    x_range = max(cords_pca[:, 0]) - min(cords_pca[:, 0])
    y_range = max(cords_pca[:, 1]) - min(cords_pca[:, 1])
    padding_x = x_range * padding_factor
    padding_y = y_range * padding_factor

    xlim = (min(cords_pca[:, 0]) - padding_x, max(cords_pca[:, 0]) + padding_x)
    ylim = (min(cords_pca[:, 1]) - padding_y, max(cords_pca[:, 1]) + padding_y)
    return xlim, ylim

def scatter_points_within_box(cords_pca, xlim, ylim, fig=None, ax=None, spine_thickness=2, spine_color='k', **scatter_kwarg):
    '''
    draw scattering points within a box (confined by xlim and ylim)
    input:
        cords_pca: np.array. shape: (n, 2)
        xlim, ylim: tuple. (min, max). It's better to be obtained from obtain_x_y_lim
        fig, ax: matplotlib figure and axis
        spine_thickness: float. The thickness of the spine
        scatter_kwarg: dict. The kwargs for scatter
    output:
        fig, ax: matplotlib figure and axis
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))

    ax.scatter(cords_pca[:, 0], cords_pca[:, 1], color='k', **scatter_kwarg)  # s is the size of the points
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # Hide the ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(spine_color)
        spine.set_linewidth(spine_thickness)  # Set spine thickness
    return fig, ax

def origin_to_pca(feamap, n_components=2):
    '''
    Apply PCA transformation to neural states.
    '''
    pca = PCA(n_components=n_components)
    feamap_pca = pca.fit_transform(feamap)
    return feamap_pca

def prepare_and_evolve(period_name, evolve_period, fig=None, ax=None):
    '''
    prepare the initial states and evolve them
    input:
        period_name: str. The plane which uniform states initialized on, can be delay plane ('interval') or response plane ('response').
        evolve_period: list of str. The period to evolve the states, can be ['go_cue', 'go_cue'] or ['response', 'response']
    '''
    ### set up uniform initial states in the period_name plane
    sub = Agent(model_file, rule_name)
    sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree,
               sigma_rec=sigma_rec, sigma_x=sigma_x)
    hhelper = Hidden0_helper(hidden_size=256)
    cords_pca, cords_origin = hhelper.delay_ring(sub, period_name=period_name,
                                                 batch_size=n_ring_point)
    ### envolve through evolve_period
    se = State_Evolver()
    se.read_rnn_file(model_file, rule_name)
    end_cords_origin = se.evolve(cords_origin, evolve_period=evolve_period)
    end_cords_pca = origin_to_pca(end_cords_origin[-1])

    if period_name == 'response': # only the averaged neural state will be converted to color
        end_cords_pca = origin_to_pca(np.mean(end_cords_origin, axis=0))
    
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(5, 3))
    
    spine_colors = ['tab:blue', 'tab:red']
    for i, cords in enumerate([cords_pca, end_cords_pca]):
        xlim, ylim = obtain_x_y_lim(cords, padding_factor=0.1)
        fig, ax[i] = scatter_points_within_box(cords, xlim, ylim, fig=fig, ax=ax[i], s=30, spine_color=spine_colors[i])
    return fig, ax

#################### Hyperparameters
# Model parameters
prior_sig = 3.0  # Width of the environmental prior distribution
rule_name = 'color_reproduction_delay_unit'
adapted_model_dir_parent = "../core/model_short_res_40/model_" + str(prior_sig) + "/color_reproduction_delay_unit/"
model_dir = 'model_5/'  # Example RNN
sub_dir = 'noise_delta/'
model_file = os.path.join(adapted_model_dir_parent, model_dir, sub_dir)

# paramters to get appropriate neural states
prod_intervals = 100  # Delay duration  for experiment
sigma_rec, sigma_x = 0, 0  # Noise
n_colors = 20
pca_degree = np.linspace(0, 360, n_colors, endpoint=False)  # Color distribution on ring
n_ring_point = 20  # Number of points on delay or response plane

#################### Main
period_name = 'interval'
evolve_period = ['go_cue', 'go_cue']
fig, ax = prepare_and_evolve(period_name, evolve_period)

ax[0].set_ylabel('PC2')
ax[0].set_xlabel('PC1 \n (Delay Plane)')
ax[0].set_title('Go Start \n mannually set uniform states')
ax[1].set_xlabel('PC1 \n (Response Plane)')
ax[1].set_title('Go End')
fig.savefig('./figs/fig_collect/go_start_end_plane.svg', format='svg')

period_name = 'response'
evolve_period = ['response', 'response']
fig, ax = prepare_and_evolve(period_name, evolve_period)

ax[0].set_ylabel('PC2')
ax[0].set_xlabel('PC1 \n (Response Plane)')
ax[0].set_title('Response Start \n mannually set uniform states')
ax[1].set_xlabel('PC1 \n (Response Plane)')
ax[1].set_title('Response Average')
fig.savefig('./figs/fig_collect/res_start_end_plane.svg', format='svg')

plt.show()
