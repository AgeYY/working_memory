import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
from core.tools import removeOutliers, state_to_angle
from core.color_error import circular_difference, memory_rmse
from sklearn.decomposition import PCA
from core.agent import Agent
from core.state_evolver import State_Evolver
from core.rnn_decoder import RNN_decoder

def compute_metric(data, metric_type='entropy', bins=None):
    """Compute either entropy or coefficient of variation for the data.
    
    Args:
        data: Input data array
        metric_type: 'entropy' or 'cv' (coefficient of variation)
        bins: Optional bins for histogram computation. If None, uses data directly.
    
    Returns:
        float: Computed metric value
    """
    if bins is not None:
        hist, _ = np.histogram(data, bins=bins, density=True)
        data = hist

    if metric_type == 'entropy':
        return entropy(data)
    else:  # coefficient of variation
        return np.std(data) / np.mean(data) if np.mean(data) != 0 else 0

def setup_plotting_style(fontsize=15, linewidth=2):
    """Set up common matplotlib plotting style."""
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('axes', linewidth=linewidth)

def create_broken_axis_plot(sigmas, metric_mean, metric_std, metric_name, 
                          figsize=(3,3), ylabel_fontsize=13, xlabel_fontsize=15):
    """Create a broken axis plot with error bars.
    
    Args:
        sigmas: x-axis values
        metric_mean: y-axis mean values
        metric_std: y-axis standard deviation values
        metric_name: Name of metric for ylabel ('entropy' or 'cv')
        figsize: Figure size tuple
        ylabel_fontsize: Font size for y-label
        xlabel_fontsize: Font size for x-label
    
    Returns:
        tuple: (fig, bax) matplotlib figure and broken axes objects
    """
    fig = plt.figure(figsize=figsize)
    bax = brokenaxes(xlims=((0, 35), (85, 95)), hspace=.05)
    
    bax.errorbar(x=sigmas, y=metric_mean, yerr=metric_std, 
                color='k', fmt='.-', linewidth=1.5, 
                markersize=15, alpha=1)
    
    ylabel = 'Entropy' if metric_name == 'entropy' else 'Coefficient of Variation'
    bax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    bax.set_xlabel(r'$\sigma_s$', fontsize=xlabel_fontsize)
    
    bax.axs[0].set_xticks([10,20,30])
    bax.axs[0].set_xticklabels(['10.0','20.0','30.0'])
    bax.axs[1].set_xticks([90])
    bax.axs[1].set_xticklabels(['90.0'])
    
    return fig, bax

def process_metric_data(metric_all, error_type='sem'):
    """Process metric data to compute mean and standard deviation.
    
    Args:
        metric_all: List of metric values for each sigma
    
    Returns:
        tuple: (metric_mean, metric_std) Lists of means and standard deviations
    """
    metric_all = [removeOutliers(np.array(x)) for x in metric_all]
    metric_mean = [np.mean(x) for x in metric_all]
    if error_type == 'sem':
        metric_std = [np.std(x) / np.sqrt(len(x)) for x in metric_all]
    elif error_type == 'std':
        metric_std = [np.std(x) for x in metric_all]
    return metric_mean, metric_std 


class PostDelayEvolver():
    def __init__(self):
        pass

    def read_rnn_file(self, model_file, rule_name):
        self.model_file = model_file
        self.rule_name = rule_name

    def fit_planes(self, prod_intervals=1000, ring_centers=None, sigma_rec=0, sigma_x=0):
        if ring_centers is None:
            ring_centers = np.linspace(0, 360, 100, endpoint=False)
        sub = Agent(self.model_file, self.rule_name)
        sub.do_exp(prod_intervals=prod_intervals, ring_centers=ring_centers, sigma_rec=sigma_rec, sigma_x=sigma_x)

        self.pca_delay_go_start = PCA(n_components=2)
        go_start_states = sub.state[sub.epochs['interval'][1]-1]
        self.pca_delay_go_start.fit(go_start_states) # use the end of state 
        self.delay_radius_go_start = np.mean(np.linalg.norm(self.pca_delay_go_start.transform(go_start_states), axis=1))

        self.pca_delay_go_end = PCA(n_components=2)
        go_end_states = sub.state[sub.epochs['go_cue'][1] - 1]
        self.pca_delay_go_end.fit(go_end_states)
        self.delay_radius_go_end = np.mean(np.linalg.norm(self.pca_delay_go_end.transform(go_end_states), axis=1))

        self.pca_response = PCA(n_components=2)
        mean_response = np.mean(sub.state[sub.epochs['response'][0]:sub.epochs['response'][1]], axis=0)
        self.pca_response.fit(mean_response)
        self.response_radius = np.mean(np.linalg.norm(self.pca_response.transform(mean_response), axis=1))

        return self.pca_delay_go_start, self.pca_delay_go_end, self.pca_response

    def get_common_angles(self, common_color=130, dense_search=1000): # common_color = [40, 130, 220, 310]
        if common_color not in [40, 130, 220, 310]:
            raise ValueError("common_color must be one of [40, 130, 220, 310]")
        
        # sample a ring in the delay plane
        sub = Agent(self.model_file, self.rule_name)
        search_input_colors = np.linspace(0, 360, dense_search, endpoint=False)
        sub.do_exp(prod_intervals=800, ring_centers=search_input_colors, sigma_rec=None, sigma_x=None)
        output_colors = sub.behaviour['report_color']

        # Find index of color nearest to common_color
        color_diffs = np.abs(circular_difference(output_colors, common_color, max_angle=360))
        common_color_idx = np.argmin(color_diffs)

        # common color's angle in delay plane at go_start
        go_start_states = sub.state[sub.epochs['interval'][1]-1]
        self.angle_delay_common_go_start = state_to_angle(go_start_states, pca=self.pca_delay_go_start, state_type='data', verbose=False)[common_color_idx]

        # common color's angle in delay plane at go_end
        go_end_states = sub.state[sub.epochs['go_cue'][1] - 1]
        self.angle_delay_common_go_end = state_to_angle(go_end_states, pca=self.pca_delay_go_end, state_type='data', verbose=False)[common_color_idx]

        # common color's angle in response plane
        mean_response = np.mean(sub.state[sub.epochs['response'][0]:sub.epochs['response'][1]], axis=0)
        self.angle_response_common = state_to_angle(mean_response, pca=self.pca_response, state_type='data', verbose=False)[common_color_idx]

        return self.angle_delay_common_go_start, self.angle_delay_common_go_end, self.angle_response_common

    def _angles_to_states(self, init_angles, radius, pca):
        """Helper method to convert angles to neural states using PCA"""
        init_cords = np.array([
            [radius * np.cos(angle * np.pi / 180),
             radius * np.sin(angle * np.pi / 180)]
            for angle in init_angles])
        return pca.inverse_transform(init_cords)

    def evolve_go_dynamics(self, init_angles):
        init_states = self._angles_to_states(init_angles, 
                                           self.delay_radius_go_start,
                                           self.pca_delay_go_start)
        se = State_Evolver()
        se.read_rnn_file(self.model_file, self.rule_name)
        evolved_states = se.evolve(init_states, evolve_period=["go_cue", "go_cue"])[-1]
        evolved_angles = state_to_angle(evolved_states, pca=self.pca_delay_go_end, 
                            state_type='data', verbose=False)
        return evolved_angles

    def evolve_response_dynamics(self, init_angles):
        init_states = self._angles_to_states(init_angles,
                                           self.delay_radius_go_end, 
                                           self.pca_delay_go_end)
        se = State_Evolver()
        se.read_rnn_file(self.model_file, self.rule_name)
        evolved_states = se.evolve(init_states, evolve_period=["response", "response"])
        evolved_states = np.mean(evolved_states, axis=0)
        evolved_angles = state_to_angle(evolved_states, pca=self.pca_response,
                            state_type='data', verbose=False)
        return evolved_angles

    def evolve_readout(self, init_angles):
        init_states = self._angles_to_states(init_angles,
                                           self.response_radius,
                                           self.pca_response)
        sub = Agent(self.model_file, self.rule_name)
        decoder = RNN_decoder()
        decoder.read_rnn_agent(sub)
        evolved_colors = decoder.decode(init_states, decoding_plane='response')
        return evolved_colors

    def evolve_full_post_delay(self, init_angles):
        evolved_go_angles = self.evolve_go_dynamics(init_angles)
        evolved_response_angles = self.evolve_response_dynamics(evolved_go_angles)
        evolved_readout_colors = self.evolve_readout(evolved_response_angles)
        return evolved_readout_colors

class PostDelayMemoryError():
    def __init__(self, common_color=130, delta_angle=10, n_states=500):
        self.common_color = common_color
        self.delta_angle = delta_angle
        self.n_states = n_states

    def read_rnn_file(self, model_file, rule_name):
        self.pdf = PostDelayEvolver()
        self.pdf.read_rnn_file(model_file, rule_name)
        self.pdf.fit_planes() # fit the delay and response planes
        self.angle_delay_common_go_start, self.angle_delay_common_go_end, self.angle_response_common = self.pdf.get_common_angles(common_color=self.common_color)

    def memory_error_theoretical_uniform(self):
        init_angles = np.random.uniform(self.common_color - self.delta_angle, self.common_color + self.delta_angle, self.n_states)
        memory_rmse_no_post_delay = memory_rmse(init_angles, self.common_color)
        return memory_rmse_no_post_delay
    
    def memory_error_full(self):
        init_angles = np.random.uniform(self.angle_delay_common_go_start - self.delta_angle, self.angle_delay_common_go_start + self.delta_angle, self.n_states)
        evolved_full_angles = self.pdf.evolve_full_post_delay(init_angles)
        memory_rmse_full = memory_rmse(evolved_full_angles, self.common_color)
        return memory_rmse_full
    
    def memory_error_go_dynamics(self):
        init_angles = np.random.uniform(self.angle_delay_common_go_start - self.delta_angle, self.angle_delay_common_go_start + self.delta_angle, self.n_states)
        evolved_go_angles = self.pdf.evolve_go_dynamics(init_angles)
        memory_rmse_go_dynamics = memory_rmse(evolved_go_angles, self.angle_delay_common_go_end)
        return memory_rmse_go_dynamics

    def memory_error_response_dynamics(self):
        init_angles = np.random.uniform(self.angle_delay_common_go_end - self.delta_angle, self.angle_delay_common_go_end + self.delta_angle, self.n_states)
        evolved_response_angles = self.pdf.evolve_response_dynamics(init_angles)
        memory_rmse_response_dynamics = memory_rmse(evolved_response_angles, self.angle_response_common)
        return memory_rmse_response_dynamics

    def memory_error_readout(self):
        init_angles = np.random.uniform(self.angle_response_common - self.delta_angle, self.angle_response_common + self.delta_angle, self.n_states)
        evolved_readout_colors = self.pdf.evolve_readout(init_angles)
        memory_rmse_readout = memory_rmse(evolved_readout_colors, self.common_color)
        return memory_rmse_readout