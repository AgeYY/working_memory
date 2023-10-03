import matplotlib.pyplot as plt
from global_setting import *
import scipy.stats as sci_st

def plot_layer_boxplot_helper(score_exps, layer_order, color=GREY_DARK, ax=None, fig=None, patch_artist=False, jitter=0.04, jitter_s=50, jitter_alpha=0.4, jitter_color='#1f77b4', **box_kwarg):
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

    ax.boxplot(score_list, **box_kwarg)

    if jitter is not None:
        x_data = [np.array([lo_order_id[i]] * len(d)) for i, d in enumerate(score_list)]
        x_jit = [x + sci_st.t(df=6, scale=jitter).rvs(len(x)) for x in x_data]

        # Add jittered dots ----------------------------------------------
        #for x, y, color in zip(x_jit, score_list, COLOR_SCALE):
        for x, y, c in zip(x_jit, score_list, jitter_color_list):
            ax.scatter(x, y, s=jitter_s, alpha=jitter_alpha, c=c)

    ax.set_xticklabels(layer_order.keys())
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return fig, ax
