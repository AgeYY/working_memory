import context
import numpy as np
import os
from core.manifold.fix_point import Hidden0_helper
from core.agent import Agent
from core.rnn_decoder import RNN_decoder
import matplotlib.pyplot as plt
import seaborn as sns
from core.color_manager import Degree_color

#################### Hyperparameters
# file names
prior_sig = 3.0  # Sigma value for the environmental prior
rule_name = 'color_reproduction_delay_unit'
adapted_model_dir_parent = "../core/model/model_" + str(prior_sig) + "/color_reproduction_delay_unit/"
sub_dir = 'noise_delta/'

# paramters to get appropriate neural states
prod_intervals = 100  # Delay duration
sigma_rec, sigma_x = 0, 0  # Noise
n_colors = 20  # Number of colors
pca_degree = np.linspace(0, 360, n_colors, endpoint=False)  # Color angles sampled in a circular space

# parameters about sampling grids on the PC1-PC2 plane
xlim = [-30, 30]
ylim = [-30, 30]
edge_batch_size = 50 # number of points in each direction
# period_name = 'go_cue' # can be response, of interval PC1-PC2 plane
period_name = 'response' # can be response, of interval PC1-PC2 plane

##### Get mesh points in the response PC1-PC2 plane
model_dir = 'model_0/'  # example RNN
model_file = os.path.join(adapted_model_dir_parent, model_dir, sub_dir)

# Generate neural states
sub = Agent(model_file, rule_name)
sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x) # generate initial states

# Sample points
hhelper = Hidden0_helper(hidden_size=256)
cords_pca, cords_origin = hhelper.mesh_pca_plane(sub, period_name=period_name, xlim=xlim, ylim=ylim, edge_batch_size=edge_batch_size) # get the

# Decode corresponding colors
rd = RNN_decoder()
rd.read_rnn_file(model_file, rule_name)
decode_color = rd.decode(cords_origin, decoding_plane='response')


##### quick check the distribution of colors
# plt.hist(decode_color, bins=20)
# plt.show()

########### plot figures

hidden0_grid_pca = cords_pca
report_color_grid = decode_color


deg_color = Degree_color()
colors_grid = deg_color.out_color(report_color_grid, fmat='RGBA')

fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(111)
ax.scatter(hidden0_grid_pca[:, 0], hidden0_grid_pca[:, 1], c=colors_grid, alpha=1, s=60)
def disentangle(position):
    edge = int(np.sqrt(position.shape[0]))
    x = position[:, 0].reshape(edge, edge)
    y = position[:, 1].reshape(edge, edge)
    return np.flip(x, 0), np.flip(y, 0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.tick_params(
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False
)

fig.savefig('./figs/fig_collect/decode_color_{p}_{s}.png'.format(p=period_name,s=prior_sig), format='png',dpi=900)
plt.show()
