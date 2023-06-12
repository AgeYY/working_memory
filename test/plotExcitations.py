import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import pickle
import cv2 as cv


root_dir = 'weight_pped_L1_0.02'
model_names = ['90.0','10.0','3.0']
color_bin_size =8
color_bin_floor = np.arange(0,360,color_bin_size) # Left boundary of color bin
num_bins = int(360/color_bin_size)
common_color = [40, 130, 220, 310] # common color
common_color_tmp = [round(a*num_bins/360) for a in common_color] # common color in color bins
axis = 1 # Sum the weight matrix along row / column, 0 for column and 1 for row


# '''
fig,axes = plt.subplots(2,3,sharey='row',figsize=(12,7))

for model_name in model_names:
    with open(root_dir+'/weight_pped_' + model_name + '.txt', 'rb') as fp: # Load the list of RNN weight matrixes, each item is a 256*256 matrix
        weight_pped_all = pickle.load(fp)
    with open(root_dir+'/label_pped_' + model_name + '.txt', 'rb') as fp: # The list of labels for above weight matrixes, each with length 256
        label_pped_all = pickle.load(fp)

    ex_color_bin = np.zeros(num_bins) # Sum of excitation of neurons in each color bin (all rnn models)
    cnt = np.zeros(num_bins) # Count of non-NaN values (all rnn models)

    for i in range(len(weight_pped_all)): # Loop over all RNN models
        weight_pped_i = weight_pped_all[i] # Weight matrix (256 * 256)
        label_pped_i = label_pped_all[i] # Label (256,)

        weight_pped_i_mask = weight_pped_i.copy()
        weight_pped_i_mask[np.where(weight_pped_i<0)] = 0 # Only consider excitations
        sum_ex = np.sum(weight_pped_i_mask,axis=axis) # Sum over row / column

        label_color_bin = np.digitize(label_pped_i,bins=color_bin_floor)-1 # Assign neurons to color bins

        ex_color_bin_sum = np.zeros(num_bins)
        ex_color_bin_cnt = np.zeros(num_bins)
        for j in range(weight_pped_i.shape[1]): # Sum the excitation of neurons within same color bin
            ex_color_bin_sum[label_color_bin[j]]+=sum_ex[j]
            ex_color_bin_cnt[label_color_bin[j]]+=1

        ex_color_bin_avg = ex_color_bin_sum/ex_color_bin_cnt # Average the excitation of single rnn first, NaN may appear
        ex_color_bin += np.nan_to_num(ex_color_bin_avg) # Add up the average
        cnt[~np.isnan(ex_color_bin_avg)]+=1 # Count non-NaN values

    ex_color_bin /= cnt # Average over rnn models
    ax0 = axes[0, model_names.index(model_name)]  # First row of figure
    ax1 = axes[1, model_names.index(model_name)]  # Second row of figure

    sns.heatmap(np.mean(weight_pped_all,axis=0), cmap='seismic', ax=ax0) # Visualize the average weight matrix
    ax0.set_title('$\sigma_{s} = ' + model_name + '^\circ$')

    for c in common_color_tmp:
        ax1.axvline(x=c, color='r', ls='--')  # Common colors

    ax1.bar(np.arange(num_bins), ex_color_bin)
    ax1.set_xticks(common_color_tmp)
    ax1.set_xticklabels(common_color)
    ax1.set_ylim(bottom=2.6)
    ax1.set_ylabel('Excitation')
    ax1.set_xlabel("Color")
    # ax1.set_yscale('log')
plt.show()
# '''


