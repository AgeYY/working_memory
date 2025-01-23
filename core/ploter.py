import matplotlib.pyplot as plt
from core.tools import removeOutliers
import scipy.stats as sci_st
import numpy as np

def plot_layer_boxplot_helper(score_exps, layer_order, color="#747473", ax=None, fig=None, patch_artist=False, jitter=0.04, jitter_s=50, jitter_alpha=0.4, jitter_color='#1f77b4', show_outlier=True, show_box=True, **box_kwarg):
    '''
    score_exps (dict): {layer_name1: [list_of_score], layer_name2: [list_of_score], ...}
    layer_order (dict): {layer_name1: 0, layer_name2: 0, ...}. A dict uses layer name as key and the order as value
    jitter (float or None): if None, no jitter, if float
    '''
    boxprops = dict(
        linewidth=2,
        color=color,
    )
    if patch_artist:
        boxprops['facecolor'] = color

    whisprops = dict(
        linewidth=2,
        color=color,
    )

    medianprops = dict(
        linewidth=4,
        color=color,
        solid_capstyle="butt"
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    lo_order_id = []
    score_list = []
    jitter_color_list = []
    for reg, value in score_exps.items():
        lo_order_id.append(layer_order[reg])
        score_list.append(value)

        if type(jitter_color) == dict:
            jitter_color_list.append(jitter_color[reg])
        else:
            jitter_color_list.append(jitter_color)

    box_kwarg_default = {'positions': lo_order_id, 'showfliers': False, 'showcaps': False, 'medianprops': medianprops, 'whiskerprops': whisprops, 'boxprops': boxprops, 'patch_artist': patch_artist}
    box_kwarg.update(box_kwarg_default)

    if show_box:
        ax.boxplot(score_list, **box_kwarg)

    if not show_outlier:
        score_list = [removeOutliers(np.array(v)) for v in score_list]

    if jitter is not None:
        x_data = [np.array([lo_order_id[i]] * len(d)) for i, d in enumerate(score_list)]
        x_jit = [x + sci_st.t(df=6, scale=jitter).rvs(len(x)) for x in x_data]

        # Add jittered dots ----------------------------------------------
        #for x, y, color in zip(x_jit, score_list, COLOR_SCALE):
        for x, y, c in zip(x_jit, score_list, jitter_color_list):
            ax.scatter(x, y, s=jitter_s, alpha=jitter_alpha, c=c)

    pos, label = [], []
    for key in layer_order:
        pos.append(layer_order[key])
        label.append(str(key))
    ax.set_xticks(pos)
    ax.set_xticklabels(label)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return fig, ax

def error_bar_plot(x, y, fig=None, ax=None, color='tab:blue', label='', error_mode='se', mean_mode='mean'):
    if fig is None: fig, ax = plt.subplots()

    if mean_mode == 'mean':
        mean_y = [np.mean(v) for v in y]
    else:
        mean_y = [np.median(v) for v in y]

    if error_mode == 'se':
        se_y = [np.std(v) / np.sqrt(len(v)) for v in y]
        ax.errorbar(x, mean_y, yerr=se_y, fmt='o', color=color)
    elif error_mode == 'std':
        se_y = [np.std(v) for v in y]
        ax.errorbar(x, mean_y, yerr=se_y, fmt='o', color=color)
    elif error_mode == 'quantile':
        y_25 = np.array([np.percentile(v, 25) for v in y])
        y_75 = np.array([np.percentile(v, 75) for v in y])
        ax.errorbar(x, mean_y, yerr=[mean_y - y_25, y_75 - mean_y], fmt='o', color=color)

    ax.plot(x, mean_y, color=color, label=label)
    return fig, ax
