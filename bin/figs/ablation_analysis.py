# For one example RNN, this shows how the intially uniformly distributed neural states changed over time in go and response epoch
import context
import numpy as np
import os
from core.manifold.fix_point import Hidden0_helper
from sklearn.decomposition import PCA
from core.agent import Agent
from core.state_evolver import State_Evolver
import matplotlib.pyplot as plt
from core.tools import state_to_angle, removeOutliers
from sklearn.metrics import mean_squared_error
from scipy import stats
from core.color_error import Circular_operator
from core.rnn_decoder import RNN_decoder

def circular_difference(angle1, angle2, max_angle=360):
    """
    Calculate the circular difference between two angles.
    """
    diff = (angle1 - angle2) % max_angle  # Difference in circular space
    diff = np.where(diff > max_angle / 2, diff - max_angle, diff)  # Ensure result is in [-max_angle/2, max_angle/2]
    return diff


def prepare_and_evolve(period_name, evolve_period, fig=None, ax=None):
    '''
    prepare the initial states and evolve them
    input:
        period_name: str. The plane which uniform states initialized on, can be delay plane ('interval') or response plane ('response').
        evolve_period: list of str. The period to evolve the states, can be ['go_cue', 'go_cue'] or ['response', 'response']
    '''
    ### set up uniform initial states in the period_name plane
    sub = Agent(model_file, rule_name)
    sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree,
               sigma_rec=sigma_rec, sigma_x=sigma_x)
    hhelper = Hidden0_helper(hidden_size=256)
    cords_pca, cords_origin = hhelper.delay_ring(sub, period_name=period_name,
                                                 batch_size=n_ring_point)
    ### envolve through evolve_period
    se = State_Evolver()
    se.read_rnn_file(model_file, rule_name)
    end_cords_origin = se.evolve(cords_origin, evolve_period=evolve_period)
    end_cords_pca = origin_to_pca(end_cords_origin[-1])

    if period_name == 'response':  # only the averaged neural state will be converted to color
        end_cords_pca = origin_to_pca(np.mean(end_cords_origin, axis=0))

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(5, 3))

    spine_colors = ['tab:blue', 'tab:red']
    for i, cords in enumerate([cords_pca, end_cords_pca]):
        xlim, ylim = obtain_x_y_lim(cords, padding_factor=0.1)
        fig, ax[i] = scatter_points_within_box(cords, xlim, ylim, fig=fig, ax=ax[i], s=30, spine_color=spine_colors[i])
    return fig, ax



def sample_and_evolve(model_file, input_color, period_name, evolve_period):
    '''
    Find out corresponding pca angle of specific input color, sample around the angle, transfer to neuron states and evolve them
    input:
        model_file: the directory of model to perform the task
        input_color: the color focused on
        period_name: str. The plane which uniform states initialized on, can be delay plane ('interval') or response plane ('response').
        evolve_period: list of str. The period to evolve the states, can be ['go_cue', 'go_cue'] or ['response', 'response']
    '''
    ### Perform the experiment
    sub = Agent(model_file, rule_name)
    sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x)

    #### Fit the delay pca plane and find out the angle of common color on the pca plane
    pca = PCA(n_components=2)
    states = sub.state[sub.epochs[period_name][1]-1]
    pca.fit(states)
    cords = pca.transform(states)
    angles = state_to_angle(states, pca=pca, state_type='data', verbose=False)
    input_color_angle = angles[list(pca_degree).index(input_color)] # I think a better way to compute input_color_angle is to average the neural states when fixing the input color as the common color??

    #### Sampling the points in the delay plane around common color (+- angle range)
    sampled_angles = np.linspace(input_color_angle - angle_range, input_color_angle + angle_range, num_samples)

    #### Map angles back to states
    radius = np.mean(np.linalg.norm(cords, axis=1))
    sampled_cords_pca = np.zeros((num_samples, 2))
    sampled_cords_pca[:, 0] = radius * np.cos(np.deg2rad(sampled_angles))
    sampled_cords_pca[:, 1] = radius * np.sin(np.deg2rad(sampled_angles))
    sampled_states = pca.inverse_transform(sampled_cords_pca) # states of the sampled pioints

    #### Evolve
    se = State_Evolver()
    se.read_rnn_file(model_file, rule_name)
    if period_name == 'response':
        evolved_states = np.mean(se.evolve(sampled_states, evolve_period=evolve_period), axis=0)
    else:
        evolved_states = se.evolve(sampled_states, evolve_period=evolve_period)[-1]
    evolved_angles = state_to_angle(evolved_states, pca=pca, state_type="data")

    #### MSE
    circular_op = Circular_operator(0, 360)
    mse = np.mean((circular_op.diff(np.ones(sampled_angles.shape) * input_color_angle, evolved_angles)) ** 2)  # difference between angle of input color and evolved angles
    memory_error = np.sqrt(mse)
    # mse = np.mean((circular_op.diff(sampled_angles, evolved_angles)) ** 2)  # difference between sampled angles and evolved angles

    return memory_error

######## Hyperparameters
# Model parameters
prior_sig = 3.0  # Width of the environmental prior distribution
rule_name = 'color_reproduction_delay_unit'
adapted_model_dir_parent = "../core/model_short_res_40/model_" + str(prior_sig) + "/color_reproduction_delay_unit/"
sub_dir = 'noise_delta/'

# paramters to get appropriate neural states
prod_intervals = 1000  # Delay duration  for experiment
sigma_rec, sigma_x = 0, 0  # Noise
input_color = 40  # The input will be fixed to 40 degree (common color) or 85 degree (uncommon color)
n_colors = 360
pca_degree = np.linspace(0, 360, n_colors, endpoint=False) # Colors to fit the PCA
angle_range = 10
num_samples = 50
n_models = 30

######## Go dynamic only
Errors_go = []

for i in range(n_models):
    model_dir = f'model_{i}/'  # Example RNN
    model_file = os.path.join(adapted_model_dir_parent, model_dir, sub_dir)
    print(model_file)

    error = sample_and_evolve(model_file, input_color, period_name='interval', evolve_period=["go_cue", "go_cue"])
    Errors_go.append(error)
    print('go dynamic error', error)

Errors_go = removeOutliers(np.array(Errors_go))


######## Response dynamic only
Errors_response = []

for i in range(n_models):
    model_dir = f'model_{i}/'  # Example RNN
    model_file = os.path.join(adapted_model_dir_parent, model_dir, sub_dir)
    print(model_file)

    error = sample_and_evolve(model_file, input_color, period_name='response', evolve_period=['response', 'response'])
    Errors_response.append(error)
    print('response dynamic error', error)

Errors_response = removeOutliers(np.array(Errors_response))


######## Readout only
Errors_readout = []


for i in range(n_models):
    print('model', i)
    #### Perform the experiment
    model_dir = f'model_{i}/'  # Example RNN
    model_file = os.path.join(adapted_model_dir_parent, model_dir, sub_dir)
    print(model_file)

    sub = Agent(model_file, rule_name)
    sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=sigma_rec, sigma_x=sigma_x)

    #### Fit the response pca plane and find out the angle of common color
    pca_res = PCA(n_components=2)
    states_res =sub.state[sub.epochs['response'][1]-1]
    pca_res.fit(states_res)
    states_res_pca = pca_res.transform(states_res)
    angles_res = state_to_angle(states_res, pca=pca_res, state_type='data',verbose=False)
    input_color_angle_res = angles_res[list(pca_degree).index(input_color)]

    #### Sampling the points in the delay plane around common color (+-10)
    sampled_angles = np.linspace(input_color_angle_res - angle_range, input_color_angle_res + angle_range, num_samples)

    radius = np.mean(np.linalg.norm(states_res_pca, axis=1))
    sampled_cords_pca = np.zeros((num_samples, 2))
    sampled_cords_pca[:, 0] = radius * np.cos(np.deg2rad(sampled_angles))
    sampled_cords_pca[:, 1] = radius * np.sin(np.deg2rad(sampled_angles))

    # states of the sampled pioints
    sampled_states = pca_res.inverse_transform(sampled_cords_pca)

    # Readout
    decoder = RNN_decoder()
    decoder.read_rnn_agent(sub)
    decoded_colors = decoder.decode(sampled_states, decoding_plane='response')

    # Compute MSE between sampled_angles and decoded_colors
    circular_op = Circular_operator(0, 360)
    mse = np.mean((circular_op.diff(np.ones(decoded_colors.shape)*input_color, decoded_colors)) ** 2)
    memory_error = np.sqrt(mse)

    Errors_readout.append(memory_error)
    print('readout error', memory_error)

Errors_readout = removeOutliers(np.array(Errors_readout))


fig, ax = plt.subplots()
ax.boxplot([Errors_go, Errors_response, Errors_readout], showfliers=False)
ax.set_xticklabels(['Go','Response','Readout'])
plt.show()




