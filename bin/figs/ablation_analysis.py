import context
import numpy as np
from brokenaxes import brokenaxes
import os
from core.post_delay_metric_analysis import PostDelayMemoryError, PostDelayEvolver, process_metric_data
from core.ploter import plot_layer_boxplot_helper
import matplotlib.pyplot as plt
import pickle

def main():
    rmse_no_post_delay_sigma = []
    rmse_go_dynamics_sigma = []
    rmse_response_dynamics_sigma = []
    rmse_readout_sigma = []
    rmse_full_sigma = []

    for sigma_s in sigma_s_list:
        rmse_no_post_delay_sigma.append([])
        rmse_go_dynamics_sigma.append([])
        rmse_response_dynamics_sigma.append([])
        rmse_readout_sigma.append([])
        rmse_full_sigma.append([])

        for i in range(n_models):
            print(f"sigma_s: {sigma_s}, i: {i}")
            model_file = model_file_func(sigma_s, i)

            pdme = PostDelayMemoryError(common_color=common_color, delta_angle=delta_angle, n_states=500)
            pdme.read_rnn_file(model_file, rule_name)
            rmse_no_post_delay_sigma[-1].append(pdme.memory_error_theoretical_uniform())
            rmse_full_sigma[-1].append(pdme.memory_error_full())
            rmse_go_dynamics_sigma[-1].append(pdme.memory_error_go_dynamics())
            rmse_response_dynamics_sigma[-1].append(pdme.memory_error_response_dynamics())
            rmse_readout_sigma[-1].append(pdme.memory_error_readout())
        
    with open('./figs/fig_data/ablation_analysis.txt', 'wb') as fp:
        pickle.dump((rmse_no_post_delay_sigma, rmse_go_dynamics_sigma, rmse_response_dynamics_sigma, rmse_readout_sigma, rmse_full_sigma), fp)

def plot_one_sigma(data, sigma_s_idx, jitter_color=None, fig=None, ax=None):
    rmse_no_post_delay_sigma = data[0]
    rmse_go_dynamics_sigma = data[1]
    rmse_response_dynamics_sigma = data[2]
    rmse_readout_sigma = data[3]
    rmse_full_sigma = data[4]

    score_dict = {
        # 'full': rmse_full_sigma[sigma_s_idx],
        'go dynamics': rmse_go_dynamics_sigma[sigma_s_idx],
        'response dynamics': rmse_response_dynamics_sigma[sigma_s_idx],
        'readout': rmse_readout_sigma[sigma_s_idx]
    }

    layer_order = {
        'go dynamics': 0,
        'response dynamics': 1,
        'readout': 2,
        # 'full': 3
    }

    if jitter_color is None:
        jitter_color = {'go dynamics': 'tab:blue', 'response dynamics': 'tab:red', 'readout': 'tab:green', 'full': 'tab:orange'}
    else:
        jitter_color = 'grey'

    plot_layer_boxplot_helper(score_dict, layer_order, ax=ax, jitter_color=jitter_color, fig=fig, show_outlier=False)
    return fig, ax

if __name__ == "__main__":
    common_color = 130
    delta_angle = 10
    n_models = 50
    sigma_s_list = [3.0, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 90.0]
    model_file_func = lambda sigma_s, i: "../core/model/model_" + str(sigma_s) + "/color_reproduction_delay_unit/" + f"model_{i}/"
    rule_name = 'color_reproduction_delay_unit'

    # main()

    with open('./figs/fig_data/ablation_analysis.txt', 'rb') as fp:
        data = pickle.load(fp)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    plot_one_sigma(data, 0, fig=fig, ax=ax)
    plot_one_sigma(data, -1, jitter_color='grey', fig=fig, ax=ax)

    uniform_concatenate = np.concatenate([data[1][-1], data[2][-1], data[3][-1]], axis=0).flatten()
    avg_error_uniform = np.mean(uniform_concatenate)
    ax.axhline(avg_error_uniform, color='black', linestyle='--', label='Theoretical memory error (all uniform)')
    ax.set_ylabel('Post-delay memory error \n when only one factor is non-uniform')
    ax.set_ylim(bottom=0)
    plt.xticks(rotation=45)
    plt.savefig('./figs/fig_collect/ablation_analysis.svg', dpi=300)


    metric_go_mean, metric_go_std = process_metric_data(data[1], error_type='std')
    # metric_go_mean = np.array(metric_go_mean) - metric_go_mean[-1]  # subtract the last element
    metric_response_mean, metric_response_std = process_metric_data(data[2], error_type='std')
    # metric_response_mean = np.array(metric_response_mean) - metric_response_mean[-1]  # subtract the last element
    metric_readout_mean, metric_readout_std = process_metric_data(data[3], error_type='std')
    # metric_readout_mean = np.array(metric_readout_mean) - metric_readout_mean[-1]  # subtract the last element

    # Create figure with broken axis
    fig = plt.figure(figsize=(3, 3))
    bax = brokenaxes(xlims=((0, 35), (85, 95)), hspace=.05)

    # Plot drift metrics
    bax.errorbar(x=sigma_s_list, y=metric_go_mean, yerr=metric_go_std, 
                color="tab:blue", fmt='.-', linewidth=1.5, markersize=8, label='Go', alpha=1)

    # Plot angular occupancy metrics
    bax.errorbar(x=sigma_s_list, y=metric_response_mean, yerr=metric_response_std,
                color="tab:red", fmt='.-', linewidth=1.5, markersize=8, label='Response', alpha=1)
        
    bax.errorbar(x=sigma_s_list, y=metric_readout_mean, yerr=metric_readout_std,
                color="tab:green", fmt='.-', linewidth=1.5, markersize=8, label='Readout', alpha=1)

    # Customize plot
    bax.set_ylabel('CV$_\mathrm{biased}$ - CV$_\mathrm{uniform}$', fontsize=13)
    bax.set_xlabel(r'$\sigma_s$', fontsize=15)
    bax.axs[0].set_xticks([10,20,30])
    bax.axs[0].set_xticklabels(['10','20','30'])
    bax.axs[1].set_xticks([90])
    bax.axs[1].set_xticklabels(['90'])
    bax.legend(loc='upper right', frameon=False)
    plt.savefig('./figs/fig_collect/ablation_analysis_cv.svg', dpi=300)

    plt.show()
