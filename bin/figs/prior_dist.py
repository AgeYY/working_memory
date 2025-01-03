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
from core.color_input import vonmises_prior
from scipy.stats import vonmises

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

    n_center = len(c_center)
    kappa = 1 / sigma_s**2
    y = np.zeros(mesh.shape)
    for i in range(n_center):
        y = y +  vonmises.pdf(mesh, kappa, loc=c_center[i])

    prior_y = y / n_center # divide 4 is nomalization for vonmises, second term is for baseline

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
    ax.set_ylim([0, 0.01])
    ax.set_xlabel('Stimulus (Degree)')
    ax.set_ylabel('Probability Density')

    fig_n.savefig('./figs/fig_collect/large_prior_dist_' + file_handle + '.pdf', format='pdf')

    return fig_s, ax_s, fig_n, ax

sigma_s = 3.0 / 360 * 2 * np.pi
file_handle = '3'
fig_s_3, ax_s_3, fig_n_3, ax_n_3 = plot_prior(sigma_s, file_handle)

sigma_s = 12.5 / 360 * 2 * np.pi
file_handle = '12.5'
plot_prior(sigma_s, file_handle)

sigma_s = 90.0 / 360 * 2 * np.pi
file_handle = '90'
fig_s_90, ax_s_90, fig_n_90, ax_n_90 = plot_prior(sigma_s, file_handle)

########### Double prior distribution
#
#fig_double = plt.figure(figsize=(5, 5))
#ax_double = fig_double.add_axes([0.2, 0.2, 0.6, 0.6])
#
#prior_y = dft_loss.prior_func(mesh, c_center, sigma_s)
#
#ax_double.plot(mesh, prior_y, label='CP = 0.7')
##ax_setting(ax_double, y_up=0.4)
#ax_double.set_xlim([-np.pi, np.pi])
#ax_double.set_xticks([-np.pi, 0, np.pi])
#ax_double.set_xticklabels([0, 180, 360])
#ax_double.set_ylim([0, 0.5])
#ax_double.set_xlabel('Stimuli')
#ax_double.set_ylabel('Probability density')
#ax_double.spines['top'].set_visible(False)
#ax_double.spines['right'].set_visible(False)
#plt.legend()
#
#fig_double.savefig('./figs/fig_collect/prior_double.pdf', format='pdf')
#
########### Prior distribution with two centers
#
#fig_2c = plt.figure(figsize=(5, 5))
#ax_2c = fig_2c.add_axes([0.2, 0.2, 0.6, 0.6])
#
#c_center = np.array([90, 270]) # for 2 peaks
#c_center = c_center / 360 * 2 * np.pi - np.pi # convert to rad
#center_prob = 0.5
#prior_y = dft_loss.prior_func(mesh, c_center, sigma_s)
#
#ax_2c.plot(mesh, prior_y, label='CP = 0.5')
#ax_2c.set_xlim([-np.pi, np.pi])
#ax_2c.set_xticks([-np.pi, 0, np.pi])
#ax_2c.set_xticklabels([0, 180, 360])
#ax_2c.set_ylim([0, 0.5])
#ax_2c.set_xlabel('Stimuli')
#ax_2c.set_ylabel('')
#ax_2c.spines['top'].set_visible(False)
#ax_2c.spines['right'].set_visible(False)
#plt.legend()
#
#fig_2c.savefig('./figs/fig_collect/prior_2c.pdf', format='pdf')
#
plt.show()
