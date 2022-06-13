import context
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from core.color_manager import Degree_color
from core.agent import Agent, Agent_group
import sys

try:
    model_dir = sys.argv[1]
    rule_name = sys.argv[2]
    sub_dir = sys.argv[3]
except:
    rule_name = 'color_reproduction_delay_unit'
    model_dir = '../core/model_local/color_reproduction_delay_unit/'
    sub_dir = '/noise_delta'

try:
    if sys.argv[4] == 'Y': # set false so it will not generate data
        gen_data = True
    else:
        gen_data = False
except:
    gen_data = True

binwidth = 5 # binwidth for outdire and target dire
noise_on = True
prod_int_short = 200
prod_int_long = 1000
batch_size = 2000
sigma_rec = None; sigma_x = None # set the noise to be default (training value)
out_path_short = './figs/fig_data/fig1_short_tri_noise.csv'
out_path_long = './figs/fig_data/fig1_long_tri_noise.csv'

fs = 10 # front size

#### Output data
def output_data(prod_intervals, out_path):
    group = Agent_group(model_dir, rule_name, sub_dir=sub_dir)
    group.do_batch_exp(prod_intervals=prod_intervals, sigma_rec=sigma_rec, batch_size=batch_size, sigma_x=sigma_x)

    dire_df = pd.DataFrame(group.group_behaviour)
    dire_df.to_csv(out_path)

if gen_data:
    output_data(prod_int_short, out_path_short)
    output_data(prod_int_long, out_path_long)

#### read simulated data
dire_df_short = pd.read_csv(out_path_short)
dire_df_long = pd.read_csv(out_path_long)

error_df = pd.DataFrame({
    'Short': dire_df_short['error_color'],
    'Long': dire_df_long['error_color']
})

sns.set_theme()
sns.set_style("ticks")
#### Plot Simulation
def plot_error_dist(error_df, legend=['Short', 'Long'], ylim=[0, 7e-3], with_label=False):
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0.2, 0.3, 0.63, 0.6])

    sns.kdeplot(data=error_df, ax=ax)

    # remove label
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.grid(False)
    ax.set_xticks([-180, 0, 180])
    ax.set_xticklabels(['0', '180', '360'])
    ax.set_yticks([0, ylim[1]])
    ax.tick_params(direction='in')
    
    ax.set_xlim([-180, 180])
    ax.set_ylim(ylim)

    ax.legend([ax.lines[1], ax.lines[0]], legend, loc='upper right', frameon=False, handlelength=1.5)

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

    return fig, ax

fig, ax = plot_error_dist(error_df, ylim=[0, 20e-3], legend=['0.2s', '1.0s'])
ax.set_xlabel('Response value - Stimulus', fontsize=fs)
ax.set_ylabel('Density', fontsize=fs)
fig.savefig('./figs/fig_collect/gaussian_rnn.pdf', format='pdf')
#plt.show()
