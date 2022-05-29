# calculate the r-squared of the mean bayesian drift force and the RNN's drift force
import context
import seaborn as sns
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from core.tools import mean_se, save_dic, load_dic, find_nearest
import core.tools as tools

def rad2deg(arr, shift=False):
    '''
    arr ranges from -pi to pi and be converted to 0 to 360
    '''
    if shift:
        return (arr + np.pi) / 2 / np.pi * 360
    else:
        return arr / 2 / np.pi * 360

#################### READ CONFIGURATION
keys = ['90', '25', '20', '15']
keys_num = [90, 25, 20, 15]
keys_label = [r'$90^{\circ}$', r'$25^{\circ}$', r'$20^{\circ}$', r'$15^{\circ}$']
c_center = np.array([40, 130, 220, 310]) # 4 peaks
m=3.5 # remove the outlier

c_center = c_center / 360 * 2 * np.pi - np.pi # convert to rad
n_center = len(c_center)
c_center = rad2deg(c_center, shift=True)

########## read and calculate the r sqaure
r2score_list = []
for ky in keys:
    data_path = '../bin/figs/fig_data/rnn_noise_bay_drift_' + ky + '.json'
    fit_out_path = '../bin/figs/fig_collect/bay_drift_' + ky + '.pdf'
    r2_out_path = '../bin/figs/fig_collect/bay_drift_r2' + ky + '.pdf'

    data = tools.load_dic(data_path)
    rnn_color, rnn_drift, bay_drift = np.array(data['rnn_color']), np.array(data['rnn_drift']), np.array(data['bay_drift'])
    rnn_color, rnn_drift, bay_drift = rad2deg(rnn_color, shift=True), rad2deg(rnn_drift), rad2deg(bay_drift)

    ########## calculate the mean
    color_rnn, mean_drift_rnn, se_drift_rnn = mean_se(rnn_color, rnn_drift, remove_outlier=True, m=m)
    color_rnn, mean_drift_bay, se_drift_bay = mean_se(rnn_color, bay_drift, remove_outlier=True, m=m)

    sns.set_theme()
    sns.set_style("ticks")
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_axes([0.2, 0.3, 0.63, 0.6])

    ax.plot(color_rnn, mean_drift_rnn, label='RNN')
    ax.fill_between(color_rnn, mean_drift_rnn - se_drift_rnn, mean_drift_rnn + se_drift_rnn, alpha=0.4)

    ax.plot(color_rnn, mean_drift_bay, label='Bayesian')
    ax.fill_between(color_rnn, mean_drift_bay - se_drift_bay, mean_drift_bay + se_drift_bay, alpha=0.4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Current color (degree)')
    ax.set_ylabel('Drift force (degree/ms)')

    ax.axhline(y = 0, linestyle = '--', linewidth = 3, color = 'grey')

    for cc_i in c_center:
        ax.scatter(cc_i, 0, color = 'red', s=100)
    
    plt.legend()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

    fig.savefig(fit_out_path, format='pdf')

    ##### calculate r2 score
    r2score = r2_score(mean_drift_rnn, mean_drift_bay)
    r2score_list.append(r2score)


sns.set_style("ticks")
fig = plt.figure(figsize=(5, 5))
ax1 = fig.add_axes([0.2, 0.3, 0.63, 0.6])

ax1.plot(keys_num[1:], r2score_list[1:])
ax1.scatter(keys_num[1:], r2score_list[1:])

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(direction='in')

#ax1.set_xlim(0, 30)

ax1.set_xlabel(r'$\sigma_s$ of the prior distribution (degree)')
ax1.set_ylabel('R squared')

fig.savefig(r2_out_path, format='pdf')

plt.show()
