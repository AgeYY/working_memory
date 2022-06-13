# plot the prior distribution of colors
import context
import errno
import numpy as np
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from core.color_manager import Degree_color
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
    gen_data = False

def ax_setting(ax, y_up=0.01):
    ax.tick_params(direction='in')
    ax.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        left=False,         # ticks along the top edge are off
        labelbottom=False,
        labelleft=False,
    ) # labels along the bottom edge are off

    ax.set_xlim([-180, 180])
    ax.set_xticks([-180, 0, 180])
    ax.set_ylim([0, y_up])
    ax.set_xlabel('Stimulus (Degree)')
    ax.set_ylabel('Probability Density')

c_center = np.array([40, 130, 220, 310]) # 4 peaks
c_center = c_center / 360 * 2 * np.pi - np.pi # convert to rad
mesh = np.linspace(-np.pi, np.pi, 200)
center_prob = 0.5
color = Degree_color(radius=60)
RGBA = color.out_color((mesh + np.pi) / 2 / np.pi * 360, fmat='RGBA')

def plot_prior(sigma_s, file_handle, figsize=(3, 3)):
    ##### Small window prior

    fig_s = plt.figure(figsize=figsize)
    ax_s = fig_s.add_axes([0.2, 0.2, 0.6, 0.6])

    sigma_s_rad = sigma_s / 360 * 2 * np.pi
    n_center = len(c_center)
    height = center_prob / (1 - center_prob) * (360 - n_center * sigma_s * 2) / (n_center * sigma_s * 2) # this makes sure there are center_prob chance of hitting the center colors
    y = np.ones(mesh.shape)
    for center in c_center:
        y[(mesh < (center + sigma_s_rad)) * (mesh >= (center - sigma_s_rad))] = height

    prior_y = y / np.sum(y) / (mesh[1] - mesh[0]) # normalization

    mesh_deg = (mesh) / 2 / np.pi * 360 # convert to degree
    prior_y = prior_y / 360 * 2 * np.pi
    for i in range(len(mesh_deg)-2):
        ax_s.fill_between(mesh_deg[i:i+2], prior_y[i:i+2], color=RGBA[i])

    ax_setting(ax_s)

    fig_s.savefig('./figs/fig_collect/small_prior_dist_' + file_handle + '.pdf', format='pdf')

    ##### Normal window prior
    sns.set_theme()
    sns.set_style("ticks")

    fig_n = plt.figure(figsize=figsize)
    ax = fig_n.add_axes([0.2, 0.2, 0.6, 0.6])

    for i in range(len(mesh)-2):
        ax.fill_between(mesh_deg[i:i+2], prior_y[i:i+2], color=RGBA[i])

    ax.tick_params(direction='in')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim([-180, 180])
    ax.set_xticks([-180, 0, 180])
    ax.set_xlabel('Stimulus (Degree)')
    ax.set_ylabel('Probability Density')

    fig_n.savefig('./figs/fig_collect/large_prior_dist_' + file_handle + '.pdf', format='pdf')

    return fig_s, ax_s, fig_n, ax

sigma_s = 10.0 # degree
file_handle = 'rec_10'
fig_s_10, ax_s_10, fig_n_10, ax_n_10 = plot_prior(sigma_s, file_handle)

#plt.show()
