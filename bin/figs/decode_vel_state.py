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

# Add arguments with default values
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

print('model_dir: ', model_dir)
out_dir = './figs/fig_collect/decode_plane_' + file_label + '.pdf'
out_path = './figs/fig_data/decode_vel_plane' + file_label + '.json'

# Parameters for PCA and decoding visualization
hidden_size = 256  # Size of the RNN hidden layer.
prod_intervals = 800  # Delay duration for experiments.
n_colors = 20   # Number of input colors for decoding visualization.
batch_size = n_colors  # Batch size for exacting ring initial, which is only used for hidden0_ring
pca_degree = np.linspace(0, 360, n_colors, endpoint=False)  # Input color range (0 to 360 degrees).
sigma_rec=0; sigma_x = 0  # Noise
edge_len = 30
stream_density, stream_maxlength = 0.7, 5
arrowsize = 1.5
xlim=[-edge_len, edge_len]; ylim=xlim; edge_batch_size=50; # edge_batch_size = 70


def gen_data_func():
    """
    Generate decoding data for RNN states projected into PCA space.
    """
    # Load the RNN model
    sub = Agent(model_dir + sub_dir, rule_name)
    # Run experiments with specified input colors
    sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x)
    
    # Fit PCA plane by the neural activity during the delay epoch
    n_components = 2
    pca = PCA(n_components=n_components)
    pca.fit(sub.state[sub.epochs['interval'][1]])

    # States in the hidimensional space and pca space
    hidden_size = sub.state.shape[-1]
    hhelper = Hidden0_helper(hidden_size=hidden_size)

    # Decode states from high dimesional space
    rnn_de = RNN_decoder()
    rnn_de.read_rnn_agent(sub)
    hidden0_grid_pca, hidden0_grid = hhelper.mesh_pca_plane(sub, xlim, ylim, edge_batch_size)
    print(xlim, ylim, edge_batch_size, hidden0_grid)

    # Color of the grid
    report_color_grid = rnn_de.decode(hidden0_grid)

    # Velocity of the grid
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


########### Plot Figures ###########
# Load the decoding data
data_df = tools.load_dic(out_path)
hidden0_grid_pca = np.array(data_df['hidden0_grid_pca'])
report_color_grid = np.array(data_df['report_color_grid'])
vel_pca = np.array(data_df['vel_pca'])
hidden0_vel_pca = np.array(data_df['hidden0_vel_pca'])
traj_pca = np.array(data_df['traj_pca'])

# Map decoded colors to RGBA values
deg_color = Degree_color()
colors_grid = deg_color.out_color(report_color_grid, fmat='RGBA')

fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(111)
##### decode backgroud
ax.scatter(hidden0_grid_pca[:, 0], hidden0_grid_pca[:, 1], c=colors_grid, alpha=1, s=60)
'''
##### velocity field
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

plt.show()
