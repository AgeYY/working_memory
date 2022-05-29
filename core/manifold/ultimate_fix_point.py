import numpy as np
from core.agent import Agent
from core.manifold.fix_point import Fix_point_finder, Hidden0_helper
import core.tools as tools
from core.manifold.state_analyzer import State_analyzer

def ultimate_find_fixpoints(model_dir, rule_name, prod_intervals=0, batch_size=700, initial_type='ring', n_epochs=20000, lr=0.001, speed_thre=None, plot_speed=False, milestones=[999999], sigma_init=7, min_angle=5, n_epoch_clect_slow_points=500, witheigen=False, sub=None):
    '''
    input:
        parameters for sub:
            prod_intervals: delay time length. This is for generating ring initial points for searching. 0 is recommended.
            batch_size (int): number of initial points
        initial_type (string): delay ring; random_zero; center_state. See fix_point.py/helper for more deltail.
        n_epochs (int): 20000 is recommended
        lr (float): learning rate in searching fix points
        milestones (array [int]): milestones lr = lr / 10 if steps larger than milestones. e.g. [10000, 20000]
        min_angle (float): if two fixpoints angles are smaller than min_angle, we remove one fixpoint. Unit is degree
        n_epoch_clect_slow_points (int): every n_epoch_clect_slow_points we collect slow_points. Hence the n_epoch must be larger than this.
    output:
        fixedpoints (numpy array [float] (batch_size, hidden_size)): the fix points
        atr_status (array [bool] (batch_size)): True -- attractor, false -- saddle
        jacs (array (batch_size, hidden_size, hidden_size)): jacobians
    '''
    pca_degree = np.linspace(0, 360, batch_size, endpoint=False) # Plot the trajectories of these colors
    if sub is None: # read the subject if no input subject
        sub = Agent(model_dir, rule_name)
    sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=0, sigma_x=0)

    ## find the fixed points
    ffinder = Fix_point_finder(model_dir, rule_name)

    input_size = ffinder.net.input_size
    input_rnn = np.zeros(input_size)

    # use different stratage to find the initial hidden state
    hidden_size = sub.state.shape[-1]
    hhelper = Hidden0_helper(hidden_size)
    if initial_type == 'noisy_ring':
        hidden0 = hhelper.noisy_ring(sub, batch_size=batch_size, sigma_init=sigma_init)
    elif initial_type == 'random_zero':
        hidden0 = hhelper.random_zero(sigma_init=sigma_init, batch_size=batch_size)
    elif initial_type == 'center_state':
        hidden0 = hhelper.center_state(sub, sigma_init=sigma_init, batch_size=batch_size)
    elif initial_type == 'delay_ring':
        _, hidden0 = hhelper.delay_ring(sub, sigma_init=sigma_init, batch_size=batch_size)
    else:
        raise NameError('Initial_type can only be: center_state, random_zero, or noisy_ring')

    result_points, hidden_init = ffinder.search(input_rnn, n_epochs=n_epochs, batch_size=batch_size, hidden0=hidden0, milestones=milestones, lr=lr, speed_thresh=speed_thre, n_epoch_clect_slow_points=n_epoch_clect_slow_points) # this result points contains saddle points, attractors and slow points.

    ### delete slow points, distinguish saddle and attractors by using state_analyzer
    sa = State_analyzer(prod_intervals=prod_intervals, pca_degree=pca_degree, sigma_rec=0, sigma_x=0)
    sa.read_rnn_agent(sub)

    angle = sa.angle(result_points, fit_pca=True) #angle of fixedpoints

    unique_angle_arg = tools.arg_select_unique(angle, min_angle) # several result_points overlap together (angular distance within 5 degree). We keep only one points in overlap groups.
    angle_unique = angle[unique_angle_arg]
    result_points_unique = result_points[unique_angle_arg]

    #vel = sa.velocity_state(result_points_unique) # remove slow points by speed.
    #speed = np.linalg.norm(vel, axis=1)
    #print(speed)
    #if plot_speed: # Plot speed of fixedpoints. So that one can determine the speed manually.
    #    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
    #    ax.scatter(angle_unique / 360 * 2 * np.pi, -np.log10(speed))

    #low_speed_arg = speed < speed_thre
    #fixedpoints = result_points_unique[low_speed_arg]
    fixedpoints = result_points_unique

    att_status, jacs, eigvec, eigval = sa.attractor(fixedpoints)
    if witheigen:
        fixpoint_output = {'fixpoints': fixedpoints, 'att_status': att_status, 'jacs': jacs, 'angle': angle_unique, 'eigvec': eigvec, 'eigval': eigval}
    else:
        fixpoint_output = {'fixpoints': fixedpoints, 'att_status': att_status, 'jacs': jacs, 'angle': angle_unique}

    return fixpoint_output
