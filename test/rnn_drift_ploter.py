import context
import seaborn as sns
import matplotlib.pyplot as plt
from core.agent import Agent, Agent_group
import numpy as np
from core.tools import mean_se, save_dic, load_dic, find_nearest
from core.ddm import fit_ddm
from core.bay_drift import get_bay_drift
import core.tools as tools
from mpi4py import MPI
import sys

keys = ['90.0', '30.0', '27.5', '25.0', '22.5', '20.0', '17.5', '15.0', '12.5']

file_label = keys[0]
out_dir = './figs/fig_data/rnn_noise_bay_' + file_label + '.json'
fig_name = 'rnn_bay_drift_' + file_label + '.pdf'

def rad2deg(arr, shift=False):
    '''
    arr ranges from -pi to pi and be converted to 0 to 360
    '''
    if shift:
        return (arr + np.pi) / 2 / np.pi * 360
    else:
        return arr / 2 / np.pi * 360

data = tools.load_dic(out_dir)

rnn_color, rnn_drift, bay_drift = np.array(data['rnn_color']), np.array(data['rnn_drift']), np.array(data['bay_drift'])

rnn_color, rnn_drift, bay_drift = rad2deg(rnn_color, shift=True), rad2deg(rnn_drift), rad2deg(bay_drift)
c_center = rad2deg(c_center, shift=True)
