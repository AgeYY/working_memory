import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from figs.report_dist import plot_report_dist
from core.ddm import Euler_Maruyama_solver
import core.tools as tools

rule_name = "color_reproduction_delay_unit" # rule name (RNN architeture and task type) through out this paper
model_name = '25.0'
model_dir = "../core/model/model_25.0/color_reproduction_delay_unit/" # source model
gen_data = 'Y' # generate figure data
sub_dir = "/noise_delta"
rnn_bay_drift_out_dir = './figs/fig_data/rnn_noise_bay_' + model_name + '.json'
rnn_report_dist_path = './figs/fig_data/report_dist.csv'
report_batch_size = 3000 # single agent do 3000 trials. Check the default value in report_dist.py
report_time_end = 100 # check the value of prod_intervals in report_dist

os.system('python ./figs/report_dist.py ' + model_dir + ' ' + rule_name + ' ' + sub_dir + ' ' + gen_data) # generate report distribution data

# read report data
dire_df = pd.read_csv(rnn_report_dist_path)
fig, ax = plot_report_dist(dire_df['report_color'], add_common_color=True)

##figure 6c, find figure as rnn_bay_drift_xxx.pdf
#os.system('mpiexec -n 2 python ./figs/rnn_noise_bay_drift.py' + ' --model_dir ' + model_dir + ' --file_label ' + model_name + ' --sigma_s ' + model_name) # generate ddm and bayesian optimal ddm data

data = tools.load_dic(rnn_bay_drift_out_dir)

noise = np.array(data['noise']) # unit is rad

rnn_color, rnn_drift, bay_drift = np.array(data['rnn_color']), np.array(data['rnn_drift']), np.array(data['bay_drift']) # unit is rad
n_sub = noise.shape[0]
rnn_color = rnn_color.reshape((n_sub, -1))
rnn_drift = rnn_drift.reshape((n_sub, -1)) # each row is one rnn
bay_drift = bay_drift.reshape((n_sub, -1)) # each row is one rnn

# use current drift setting to run ddm
stimuli = np.linspace(-np.pi, np.pi, report_batch_size)
ems = Euler_Maruyama_solver()
report_group = []
for i in range(n_sub):
    ems.read_terms(rnn_color[i], bay_drift[i], noise[i])
    _, report = ems.run(stimuli, time_end=report_time_end)
    report_group.append(report)

report_group = np.array(report_group).flatten()
report_group = (report_group + np.pi) / 2.0 / np.pi * 360.0

fig, ax = plot_report_dist(report_group, add_common_color=True)

plt.figure()
rnn_drift_mean = np.mean(rnn_drift, axis=0)
bay_drift_mean = np.mean(bay_drift, axis=0)

plt.plot(rnn_color[0], rnn_drift_mean)
plt.plot(rnn_color[0], bay_drift_mean)

plt.show()
