# calculate the r-squared of the mean bayesian drift force and the RNN's drift force
import context
import seaborn as sns
from scipy.stats import sem as sci_sem
from scipy.stats import tstd as sci_std
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from core.tools import mean_se, save_dic, load_dic, find_nearest
import core.tools as tools
from scipy.stats import bootstrap
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser()

parser.add_argument('--file_label', default='', type=str,
                    help='the figure filename would be name + file_label.pdf')
parser.add_argument('--gen_data', default=True, type=bool,
                    help='generate data or not')
parser.add_argument('--data_path_head', default='./figs/fig_data/rnn_noise_bay_', type=str,
                    help='output data path of rnn_noise_bay_drift.py')
parser.add_argument('--keys', default=['90.0', '25.0', '3.0'], type=str, nargs='+',
                    help='sigma_s of the prior distribution')

arg = parser.parse_args()

file_label = arg.file_label
gen_data = arg.gen_data
keys = arg.keys
data_path_head = arg.data_path_head
print(keys)

mse_out_path = './figs/fig_collect/bay_drift_mse.pdf'
fig_name = 'rnn_bay_drift_' + file_label + '.pdf'
fit_out_path_head = './figs/fig_collect/bay_drift_'

def rad2deg(arr, shift=False):
    '''
    arr ranges from -pi to pi and be converted to 0 to 360
    '''
    if shift:
        return (arr + np.pi) / 2 / np.pi * 360
    else:
        return arr / 2 / np.pi * 360

class Scorer():
    '''
    input is the bootstrap sequence from [i0, i1, i2, ..., im] where ij is integer from 0 to number of subjects. This input time seriese is used to resample
    '''
    def __init__(self, rnn_color, rnn_drift, bay_drift):
        '''
        rnn_color, rnn_drift, bay_drift [1d array]: with values of rnn_color as [2.5, 5, 7.5, ..., 2.5, 5, 7.5, ...]
        '''
        self.n_color = int(360 / (rnn_color[1] - rnn_color[0]))
        self.n_sub = len(rnn_color) // self.n_color
        self.m = 3.5

        self.rnn_color_mat = rnn_color.reshape((self.n_sub, -1))
        self.rnn_drift_mat = rnn_drift.reshape((self.n_sub, -1))
        self.bay_drift_mat = bay_drift.reshape((self.n_sub, -1))

    def r2_score(self, idx):
        '''
        idx (int array [m]): each element is the sub's id
        '''

        mean_rnn_drift = np.mean(self.rnn_drift_mat[idx], axis=0)
        mean_bay_drift = np.mean(self.bay_drift_mat[idx], axis=0)
        r2score = r2_score(mean_rnn_drift, mean_bay_drift)

        return r2score

    def mse_score(self, idx):
        '''
        idx (int array [m]): each element is the sub's id
        '''

        mean_rnn_drift = np.mean(self.rnn_drift_mat[idx], axis=0)
        mean_bay_drift = np.mean(self.bay_drift_mat[idx], axis=0)
        msescore = mean_squared_error(mean_rnn_drift, mean_bay_drift)
        return msescore

    def mse_score_se(self):
        '''
        give the mean and standard error of the mse
        '''
        msescore = mean_squared_error(self.rnn_drift_mat.T, self.bay_drift_mat.T, multioutput='raw_values')
        se = sci_sem(msescore)
        mean = np.mean(msescore)
        return mean, se

#################### READ CONFIGURATION
keys_num = [float(ky) for ky in keys]
c_center = np.array([40, 130, 220, 310]) # 4 peaks
m=3.5 # remove the outlier
n_resamples=9999 # bootstrap samples
ci_bound = [2.5, 97.5]

c_center = c_center / 360 * 2 * np.pi - np.pi # convert to rad
n_center = len(c_center)
c_center = rad2deg(c_center, shift=True)


########## read and calculate the MSE
r2score_list = []
msescore_list = []
ci_r2,ci_mse = [], []
for ky in keys:
    data_path = data_path_head + ky + '.json'
    fit_out_path = fit_out_path_head + ky + '.pdf'
    print('Computing ', ky, '...')

    data = tools.load_dic(data_path)
    rnn_color, rnn_drift, bay_drift = np.array(data['rnn_color']), np.array(data['rnn_drift']), np.array(data['bay_drift'])
    rnn_color, rnn_drift, bay_drift = rad2deg(rnn_color, shift=True), rad2deg(rnn_drift), rad2deg(bay_drift)

    ########## calculate the mean
    color_rnn, mean_drift_rnn, se_drift_rnn = mean_se(rnn_color, rnn_drift, remove_outlier=True, m=m)
    color_rnn, mean_drift_bay, se_drift_bay = mean_se(rnn_color, bay_drift, remove_outlier=True, m=m)

    ##### calculate mse score
    scorer = Scorer(rnn_color, rnn_drift, bay_drift)

    ###### calculate the squared error, use standard error for mse
    msescore, sem = scorer.mse_score_se()
    ci_mse.append([msescore - sem, msescore + sem])
    msescore_list.append(msescore)

ci_mse = np.array(ci_mse).T
ci_diff_mse = np.abs(ci_mse - msescore_list)

fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, sharey=True, gridspec_kw={'width_ratios': [3, 1]}, figsize=(3, 3))

ax1.errorbar(keys_num, msescore_list, yerr=ci_diff_mse)
ax1.scatter(keys_num, msescore_list)

ax2.errorbar(keys_num, msescore_list, yerr=ci_diff_mse)
ax2.scatter(keys_num, msescore_list)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(direction='in')
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.tick_params(direction='in', left=False)

ax1.set_xlim(9, 33)
ax2.set_xlim(88, 92)

ax1.set_xlabel(r'$\sigma_s$ of the prior distribution (deg)')
ax1.set_ylabel('Mean squared error')

plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)

fig.savefig(mse_out_path, format='pdf')

#plt.show()
