import context
from core.rnn_decoder import RNN_decoder
from core.agent import Agent
import numpy as np
from core.manifold.fix_point import Fix_point_finder, Hidden0_helper
from core.data_plot.manifold_ploter import Manifold_ploter as MPloter
import matplotlib.pyplot as plt
from core.color_manager import Degree_color
from sklearn.decomposition import PCA

model_dir = '../core/model_local/color_reproduction_delay_unit/model_16/noise_delta_stronger'
rule_name = 'color_reproduction_delay_unit'

hidden_size = 256
prod_intervals = 800
n_colors = 1000
pca_degree = np.linspace(0, 360, n_colors, endpoint=False) # Plot the trajectories of these colors
sigma_rec=0; sigma_x = 0
xlim=[-30, 30]; ylim=[-30, 30]; edge_batch_size=80;

########## Points on the ring
sub = Agent(model_dir, rule_name)
sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x)

##### fit data to find the pca plane
n_components = 2
pca = PCA(n_components=n_components)
pca.fit(sub.state[sub.epochs['interval'][1]])

##### state in the hidimensional space and pca plane
hidden_size = sub.state.shape[-1]
hhelper = Hidden0_helper(hidden_size=hidden_size, batch_size=n_colors)
hidden0_ring = hhelper.exact_ring(sub)
hidden0_ring_pca = pca.transform(hidden0_ring)

##### decode states from high dimesional space
rnn_de = RNN_decoder()
rnn_de.read_rnn_agent(sub)

report_color_ring = rnn_de.decode(hidden0_ring)

########## Points on the pca plane
hidden0_grid_pca, hidden0_grid = hhelper.mesh_pca_plane(sub, xlim, ylim, edge_batch_size)

report_color_grid = rnn_de.decode(hidden0_grid)

########### plot figures

deg_color = Degree_color()
colors_ring = deg_color.out_color(report_color_ring, fmat='RGBA')
colors_grid = deg_color.out_color(report_color_grid, fmat='RGBA')

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)

ax.scatter(hidden0_grid_pca[:, 0], hidden0_grid_pca[:, 1], c=colors_grid, alpha=1, s=20)
#ax.scatter(hidden0_ring_pca[:, 0], hidden0_ring_pca[:, 1], c='grey', alpha=1, s=10)
ax.plot(hidden0_ring_pca[:, 0], hidden0_ring_pca[:, 1], c='grey', alpha=0.4, linewidth=8)

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

plt.show()
