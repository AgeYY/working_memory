from matplotlib import pyplot as plt
import matplotlib as mpl

import os

fs = 10 # font size

_color_list = ['blue', 'red', 'black', 'yellow', 'pink']
plt.rcParams["font.family"] = "Helvetica"

plt.style.use('default')

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=fs)          # controls default text sizes
plt.rc('axes', titlesize=fs)     # fontsize of the axes title
plt.rc('axes', labelsize=fs)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=fs)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fs)    # fontsize of the tick labels
plt.rc('legend', fontsize=fs)    # legend fontsize
plt.rc('figure', titlesize=fs)

def traj3d_plot(concate_transform_split, trial_name):
    '''
    plot 3d trajectory, please reminded that color might be wrong
    '''
    fig = plt.figure(figsize=(3, 3))

    ax = fig.gca(projection='3d')
    for i in range(0, len(concate_transform_split)):

        ax.plot(concate_transform_split[i][:, 0], concate_transform_split[i][:, 1], concate_transform_split[i][:, 2], color=_color_list[i], label = trial_name[i])
        ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], concate_transform_split[i][0, 2],  marker='*', color=_color_list[i])
        ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1], concate_transform_split[i][-1, 2],  marker='o', color=_color_list[i])

    ax.legend()
    ax.set_xlabel('PC1', fontsize=fs,labelpad=-5)
    ax.set_ylabel('PC2', fontsize=fs,labelpad=-5)
    ax.set_zlabel('PC3', fontsize=fs,labelpad=-5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.grid(False)
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.show()

    return fig

def color_curve_plot(x, y, colors, ax, linewidth=2, kwargs=None):
    '''
    x (array (n)): x coordinates
    y (array (n)): y coordinates
    colors (array (n, 4)): RGBA colors for every point
    '''
    if isinstance(colors, str):
        for i in range(x.shape[0] - 1):
            ax.plot(x[i:i+2], y[i:i+2], color=colors, **kwargs)
    else:
        for i in range(x.shape[0] - 1):
            ax.plot(x[i:i+2], y[i:i+2], color=colors[i], **kwargs)
    return ax

