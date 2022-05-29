import context
import sys
from core.rnn_decoder import RNN_decoder
from core.agent import Agent
import numpy as np
from core.manifold.fix_point import Fix_point_finder, Hidden0_helper
from core.data_plot.manifold_ploter import Manifold_ploter as MPloter
import matplotlib.pyplot as plt
from core.color_manager import Degree_color
from sklearn.decomposition import PCA
import core.tools as tools
from core.data_plot.plot_tool import color_curve_plot

try:
    model_dir = sys.argv[1]
    rule_name = sys.argv[2]
    sub_dir = sys.argv[3]
except:
    rule_name = 'color_reproduction_delay_unit'
    model_dir = '../core/model_local/color_reproduction_delay_unit/'
    sub_dir = '/model_16/noise_delta'

try:
    if sys.argv[4] == 'Y': # set false so it will not generate data
        gen_data = True
    else:
        gen_data = False
except:
    gen_data = False

hidden_size = 256
prod_intervals = 800
n_colors = 100
batch_size = n_colors # batch size for exact ring initial
pca_degree = np.linspace(0, 360, n_colors, endpoint=False) # Plot the trajectories of these colors
sigma_rec=0; sigma_x = 0
n_epochs = 60000
lr=0.1
milestones = [10000, 20000, 30000, 40000, 60000]
out_path = './figs/fig_data/noise_delay.json'

def gen_data_func():
    ########## Points on the ring
    sub = Agent(model_dir + sub_dir, rule_name)
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

    ##### Plot noise trajectory
    sub.do_exp(prod_intervals=prod_intervals, ring_centers=[180], sigma_rec=None, sigma_x=None)

    traject_pca = pca.transform(sub.state[sub.epochs['interval'][0]:sub.epochs['interval'][1]].reshape(-1, hidden_size))
    report_color_traject = [180]

    data_dic = {
            'traject_pca': traject_pca,
            'report_color_traject': report_color_traject,
            'hidden0_ring_pca': hidden0_ring_pca,
            'report_color_ring': report_color_ring}

    tools.save_dic(data_dic, out_path)

if gen_data:
    gen_data_func()

########### plot figures

data_df = tools.load_dic(out_path)
traject_pca = np.array(data_df['traject_pca'])
report_color_traject = np.array(data_df['report_color_traject'])
hidden0_ring_pca = np.array(data_df['hidden0_ring_pca'])
report_color_ring = np.array(data_df['report_color_ring'])

deg_color = Degree_color()
colors_ring = deg_color.out_color(report_color_ring, fmat='RGBA')
colors_traject = deg_color.out_color(report_color_traject, fmat='RGBA')
fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(111)

color_curve_plot(hidden0_ring_pca[:, 0], hidden0_ring_pca[:, 1], colors=colors_ring, ax=ax, kwargs={'linewidth': 15, 'alpha': 0.5})
#ax.plot(traject_pca[:, 0], traject_pca[:, 1], alpha=0.5, linewidth=5, color=colors_traject.flatten(), zorder=9999)
ax.plot(traject_pca[:, 0], traject_pca[:, 1], alpha=0.5, linewidth=5, color='grey', zorder=9999)

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

fig.savefig('./figs/fig_collect/fix_point_ring.pdf', format='pdf')

plt.show()
