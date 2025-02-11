import context
import os
import numpy as np
import matplotlib.pyplot as plt
from core.color_error import Color_error
from core.color_input import Color_input
from core.tools import find_indices, removeOutliers
from core.ploter import plot_layer_boxplot_helper
from core.agent import Agent, Agent_group
from core.manifold.state_analyzer import State_analyzer
from brokenaxes import brokenaxes
import pickle
import math
from matplotlib.lines import Line2D
from scipy import stats

# Make x and y tick labels larger
plt.rc('ytick', labelsize=20)
plt.rc('xtick', labelsize=20)

def compute_dispersion(rnn_id, prior_sig=12.5, prod_int=800, input_color=130, batch_size=500, 
                       sigma_rec=None, sigma_x=None, rule_name='color_reproduction_delay_unit', sub_dir='noise_delta/'):
    """
    Computes the dispersion for a given rnn_id.
    
    Parameters:
        rnn_id (int): The RNN identifier.
        prior_sig (float): The width of the prior.
        prod_int (int): Duration of the delay.
        input_color (float): Fixed input color.
        batch_size (int): Number of trials.
        sigma_rec (float): Recurrent noise (default: None).
        sigma_x (float): External noise (default: None).
        rule_name (str): The task rule name.
        sub_dir (str): Sub-directory for model data.
        
    Returns:
        dispersion (float): The computed dispersion.
    """
    # Path to the model directory
    model_dir_parent = "../core/model/model_" + str(prior_sig) + "/color_reproduction_delay_unit/"
    model_dir = 'model_{}/'.format(rnn_id)
    f = os.path.join(model_dir_parent, model_dir, sub_dir)
    
    # Create an agent and run experiment for the given configuration
    sub = Agent(f, rule_name)
    input_color_list = np.ones(batch_size) * input_color  # repeat common color over trials
    sub.do_exp(prod_intervals=prod_int, sigma_rec=sigma_rec, sigma_x=sigma_x, ring_centers=input_color_list)
    
    # Get the state at the end of delay period
    end_of_delay_state = sub.state[sub.epochs['interval'][1]]  # shape is [batch_size, hidden_size]
    
    # Analyze the state
    sa = State_analyzer()
    sa.read_rnn_file(f, rule_name)
    phii = sa.angle(end_of_delay_state, fit_pca=True)
    
    # Adjust phii to avoid circular boundary issues and remove outliers
    median_phii = np.median(phii)
    phii = (phii - median_phii + 180) % 360
    phii = removeOutliers(phii)
    
    # Compute the dispersion (standard deviation) of the angles
    sqe_phi = (phii - np.mean(phii)) ** 2
    dispersion = np.sqrt(np.mean(sqe_phi))
    
    return dispersion

if __name__ == "__main__":
    # # List of RNN IDs for which to compute dispersion.
    # prior_sig_list = [12.5, 90.0]
    # rnn_ids = np.arange(0, 50)
    
    # results = {}
    # for prior_sig in prior_sig_list:
    #     results[prior_sig] = []
    #     for rid in rnn_ids:
    #         dispersion = compute_dispersion(rid, prior_sig)
    #         print("RNN ID {}: Dispersion = {}".format(rid, dispersion))
    #         results[prior_sig].append(dispersion)

    # # Save results to pickle file
    # with open('../bin/figs/fig_data/dynamic_dispersion_uniform_bias.pkl', 'wb') as f:
    #     pickle.dump(results, f)

    # Load results from pickle file
    with open('../bin/figs/fig_data/dynamic_dispersion_uniform_bias.pkl', 'rb') as f:
        results = pickle.load(f)

    dispersion_dict = {"Uniform RNN": results[90.0], "Biased RNN": results[12.5]}
    layer_order = {'Uniform RNN':0, 'Biased RNN': 1}
    jitter_color_order = {'Biased RNN': '#d62728', 'Uniform RNN': '#1f77b4'}
    fig, ax = plot_layer_boxplot_helper(dispersion_dict, layer_order, jitter_color=jitter_color_order,show_outlier=True)
    # Perform Wilcoxon rank-sum test between uniform and biased RNN dispersions
    stat, pval = stats.ranksums(dispersion_dict['Uniform RNN'], dispersion_dict['Biased RNN'])
    print(f"\nWilcoxon rank-sum test results:")
    print(f"Statistic: {stat:.4f}")
    print(f"p-value: {pval:.4e}")

    fig.savefig('../bin/figs/fig_collect/dynamic_dispersion_uniform_bias.svg',format='svg',bbox_inches='tight')
    plt.show()