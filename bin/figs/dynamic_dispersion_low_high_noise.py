# compute the the color dispersion
import context
import os
from core.color_error import Color_error
from core.color_input import Color_input
from core.tools import find_indices, removeOutliers
from core.ploter import plot_layer_boxplot_helper
import numpy as np
import matplotlib.pyplot as plt
from core.agent import Agent, Agent_group
from core.manifold.state_analyzer import State_analyzer
from brokenaxes import brokenaxes
import pickle
import math
from matplotlib.lines import Line2D


model_names = ['Biased RNN', 'Uniform RNN']
sigmas_all = [3.0, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 90.0] # all sigmas stored in the file
sigmas = [12.5, 90.0] # only draw boxes of these sigmas
sigmas_id = find_indices(sigmas_all, sigmas)

######### Plot dispersion
# '''
with open('../bin/figs/fig_data/dynamic_dispersion_40.txt','rb') as fp:
    dispersion_all = pickle.load(fp)

dispersion_all = np.array(dispersion_all)
dispersion_dict = {model_names[i]: np.sqrt(dispersion_all[sigmas_id[i]]) for i, sigs in enumerate(sigmas)} # select target sigmas
for key in dispersion_dict: dispersion_dict[key] = removeOutliers(dispersion_dict[key]) # remove outlier

layer_order = {'Uniform RNN':0, 'Biased RNN': 1}
jitter_color_order = {'Biased RNN': '#d62728', 'Uniform RNN': '#1f77b4'}
fig, ax = plot_layer_boxplot_helper(dispersion_dict, layer_order, jitter_color=jitter_color_order)

fig.savefig('../bin/figs/fig_collect/dynamic_dispersion_uniform_bias.svg',format='svg',bbox_inches='tight')
plt.show()
# '''
