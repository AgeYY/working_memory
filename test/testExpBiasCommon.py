# plot the bias common figure in the paper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = '../bin/figs/fig_data/exp_bias_common.csv'
data_dic = pd.read_csv(data_path, dtype=float).to_dict(orient='list')

for key in data_dic:
    if (key[7:11] == 'line'):
        data_dic[key] = np.array(data_dic[key])[~np.isnan(data_dic[key])]

target = np.linspace(-30, 30, 100)
target_line = np.linspace(-15, 15, 50)

key_his = ''
for key in data_dic:
    if (key[:6] == 'report') and (key[7:11] != 'line'):
        data_dic[key] = np.interp(target, data_dic[key_his], data_dic[key])
        data_dic[key_his] = target
    elif (key[:6] == 'report') and (key[7:11] == 'line'):
        p = np.polyfit(data_dic[key_his], data_dic[key], deg=1)
        data_dic[key_his] = target_line
        data_dic[key] = p[0] * target_line + p[1]

    key_his = key

def plot_exp(target, report_up_short, report_low_short, report_up_long, report_low_long, report_line_short, report_line_long):
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0.23, 0.2, 0.6, 0.7])
    ax.fill_between(target, report_up_short, report_low_short, alpha=0.4)
    ax.plot(target_line, report_line_short)
    
    ax.fill_between(target, report_up_long, report_low_long, alpha=0.4)
    ax.plot(target_line, report_line_long)
    ax.set_xlim([-30, 30]) # there is end effect for calculating ci
    ax.set_ylim([-6, 6]) # there is end effect for calculating ci

    ax.set_xticks([-30, 0, 30])
    ax.set_yticks([-6, 0, 6])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Target color (deg) \n (relative to common color) ')
    ax.set_ylabel('Report - Target (deg)')
    return fig, ax

fig, ax = plot_exp(target, data_dic['report_up_short_unbias'], data_dic['report_low_short_unbias'], data_dic['report_up_long_unbias'], data_dic['report_low_long_unbias'], data_dic['report_line_short_unbias'], data_dic['report_line_long_unbias'])
plt.show()


fig, ax = plot_exp(target, data_dic['report_up_short_bias'], data_dic['report_low_short_bias'], data_dic['report_up_long_bias'], data_dic['report_low_long_bias'], data_dic['report_line_short_bias'], data_dic['report_line_long_bias'])
plt.show()
