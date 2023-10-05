import context
import sys
from core.rnn_decoder import RNN_decoder
from core.agent import Agent
import numpy as np
from core.data_plot.manifold_ploter import Manifold_ploter as MPloter
import matplotlib.pyplot as plt
from core.color_manager import Degree_color
from sklearn.decomposition import PCA
from core.data_plot.plot_tool import color_curve_plot
import core.tools as tools
from core.manifold.state_analyzer import State_analyzer
from core.manifold.fix_point import Hidden0_helper
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default="../core/model/model_90.0/color_reproduction_delay_unit/", type=str,
                    help='models')
parser.add_argument('--rule_name', default='color_reproduction_delay_unit', type=str,
                    help='RNN and architeture type, fix to the default throught out this paper')
parser.add_argument('--sub_dir', default="/model_0/noise_delta", type=str,
                    help='example model')
parser.add_argument('--file_label', default='', type=str,
                    help='the figure filename would be decode_plane_ + file_label.pdf')
parser.add_argument('--gen_data', default=True, type=bool,
                    help='generate figure data')

arg = parser.parse_args()

model_dir = arg.model_dir
rule_name = arg.rule_name
sub_dir = arg.sub_dir
file_label = arg.file_label
gen_data = arg.gen_data

out_dir = './figs/fig_collect/decode_plane_' + file_label + '.pdf'
out_path = './figs/fig_data/decode_vel_plane' + file_label + '.json'

hidden_size = 256
prod_intervals = 100
n_colors = 20
batch_size = n_colors # batch size for exact ring initial, which is only used for hidden0_ring
pca_degree = np.linspace(0, 360, n_colors, endpoint=False)
sigma_rec=0; sigma_x = 0
edge_len = 30
stream_density, stream_maxlength = 0.7, 5
arrowsize = 1.5
xlim=[-edge_len, edge_len]; ylim=xlim; edge_batch_size=50; # edge_batch_size = 70

def gen_data_func():
    sub = Agent(model_dir + sub_dir, rule_name)
    sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x)
    
    # fit data to find the pca plane
    n_components = 2
    pca = PCA(n_components=n_components)
    pca.fit(sub.state[sub.epochs['interval'][1]])

    # state in the hidimensional space and pca plane
    hidden_size = sub.state.shape[-1]
    hhelper = Hidden0_helper(hidden_size=hidden_size)

    # decode states from high dimesional space
    rnn_de = RNN_decoder()
    rnn_de.read_rnn_agent(sub)

    hidden0_grid_pca, hidden0_grid = hhelper.mesh_pca_plane(sub, xlim, ylim, edge_batch_size)

    # color of the grid
    report_color_grid = rnn_de.decode(hidden0_grid)

    # velocity of the grid
    hidden0_vel, hidden0_vel_pca = hidden0_grid, hidden0_grid_pca
    sa = State_analyzer(prod_intervals=1000)
    sa.read_rnn_agent(sub)
    vel = sa.velocity_state(hidden0_vel)
    vel_pca = pca.transform(vel)

    traj = sub.state[sub.epochs['interval'][1]]
    traj_pca = pca.transform(traj)
    data_dic = {
        'hidden0_grid_pca': hidden0_grid_pca,
        'report_color_grid': report_color_grid,
        'hidden0_vel_pca': hidden0_vel_pca,
        'vel_pca': vel_pca,
        'traj_pca': traj_pca
    }

    tools.save_dic(data_dic, out_path)

if gen_data:
    gen_data_func()
########### plot figures

data_df = tools.load_dic(out_path)
hidden0_grid_pca = np.array(data_df['hidden0_grid_pca'])
report_color_grid = np.array(data_df['report_color_grid'])
vel_pca = np.array(data_df['vel_pca'])
hidden0_vel_pca = np.array(data_df['hidden0_vel_pca'])
traj_pca = np.array(data_df['traj_pca'])

deg_color = Degree_color()
colors_grid = deg_color.out_color(report_color_grid, fmat='RGBA')

fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(111)
##### decode backgroud
ax.scatter(hidden0_grid_pca[:, 0], hidden0_grid_pca[:, 1], c=colors_grid, alpha=1, s=60)
##### velocity field
'''
def disentangle(position):
    edge = int(np.sqrt(position.shape[0]))
    x = position[:, 0].reshape(edge, edge)
    y = position[:, 1].reshape(edge, edge)
    return np.flip(x, 0), np.flip(y, 0)

x_pca, y_pca = disentangle(hidden0_vel_pca)
vel_x_pca, vel_y_pca = disentangle(vel_pca)

speed = np.sqrt(vel_x_pca**2 + vel_y_pca**2)
lw = 5 * speed / 20
ax.streamplot(x_pca[0, :], y_pca[:, 0], vel_x_pca, vel_y_pca, density=stream_density, maxlength=stream_maxlength, linewidth=lw, color='b', integration_direction='both', arrowsize=arrowsize,  arrowstyle='->')
'''

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

out_dir = out_dir[:-3] + 'png'
fig.savefig(out_dir, format='png', dpi=900)
#out_dir = out_dir[:-3] + 'eps'
#fig.savefig(out_dir, format='eps')

#plt.show()


def plot_custom_colorbar(values, rgba_colors, label='Colorbar Label', tick_fontsize=12):
    """
    Create a custom colorbar using provided values and RGBA colors.

    Args:
        values (array-like): List of values.
        rgba_colors (array-like): List of RGBA colors corresponding to the values.
        label (str): Label for the colorbar.
        tick_fontsize (int): Font size for tick labels in the colorbar.

    Returns:
        None
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create a ScalarMappable object for mapping values to colors
    norm = plt.Normalize(min(values), max(values))

    # Create a custom colormap based on the provided RGBA colors
    custom_cmap = ListedColormap(rgba_colors)

    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])

    # Create a colorbar
    cbar = plt.colorbar(sm, ax=ax, location='left')
    cbar.set_label(label)

    # Adjust the colorbar's appearance
    cbar.ax.tick_params(labelsize=tick_fontsize)

    return fig, ax

color_list = np.linspace(0, 360, 100)
rgba_color = deg_color.out_color(color_list, fmat='RGBA')
fig, ax = plot_custom_colorbar(color_list, rgba_color, label='Custom Colorbar', tick_fontsize=14)
fig.savefig('./figs/fig_collect/decode_plane_cbar_' + file_label + '.pdf')
plt.show()
