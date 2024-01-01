import numpy as np
from . import task


def input_output_n(rule_name, hp):
    # The task is delayed-match task. The to-be-memorized color is periodic variable in [0, 2pi] drawn from certain distribution. Rule_name specify the architecure of the RNN. 
    if rule_name == 'color_reproduction_delay_unit': # RNN has hp['num_unit'] sensory input units. Each unit's tuning curve is a von Mise funciton. The center of hp['num_unit'] input units' tuning curve are equally spaced in [0, 2pi]
        return hp['num_unit']+1, hp['num_unit']
    elif rule_name == 'color_reproduction_delay_tri': # two sensory inputs: sin(color), cos(color)
        return 2+1, 2


def get_default_hp(rule_name, random_seed=None):
    '''Get a default hp.

    Useful for debugging.

    Returns:
        hp : a dictionary containing training hpuration
    '''


    # default seed of random number generator
    if random_seed is None:
        seed = np.random.randint(300)
    else:
        seed = random_seed
    #seed = 321985
    hp = {
        #################### Task related paprameters
        # batch size for training
        'batch_size_train': 64, #128,#64,
        # batch_size for testing
        'batch_size_test': 64,
        # recurrent noise in RNN
        'sigma_rec': 0.15,
        # input noise
        'sigma_x': 0.15,
        # time duration of go period
        'pulse_duration_go': 60,
        # the length of delay can vary from prod_interval to prod_interval uniformly in the training stage
        'prod_interval': [0, 2000],
        'bias_centers': [40., 130., 220., 310.], # 4 peaks of the prior distribution in exp2 in the paper. The unit is degree
        'n_degree': 360, # resolution of color in the natural world is 1 degree
        'sig': 10., # The prior distribution is the sum of several von Mises functions or rectangle function. Each has center 'bis_center' and width sig. In the von mises case sig would be the standard deviation, In the rectangle case sig is the width of the rect. The unit is degree
        'bias_method': 'uniform', # The prior distribution: uniform, delta (means rectangle prior distribution), vonmises
        'center_prob': 0.5, # The probability of common values, only works if the bias_method is delta
        'pulse_duration': 200, # Time length of perception period. I need to change this variable name.
        'response_duration': 200, # Time length of response period.
        'prod_step': 1000, # in the case of progressive training (see cluster_training_unit.py), increase prod_step for each step
        'sigma_rec_step': 0.05, # similar with above, but with sigma_rec.
        #################### RNN achitecure and training related parameters
        'rule_name': rule_name,
        # Type of RNNs: RNN
        'rnn_type': 'RNN',
        # Optimizer adam or sgd
        'optimizer': 'adam',
        # Type of activation functions: relu, softplus, rec_tanh, sigmoid, ReLU6, tanh
        'activation': 'tanh',
        # activation function for output, can be 'relu', 'softplus', linear, sigmoid
        'out_activation': 'linear',
        # Time constant (ms)
        'tau': 20,
        # discretization time step (ms)
        'dt': 20,
        # discretization time step/time constant
        'alpha': 1,
        # initial standard deviation of non-diagonal recurrent weights
        'initial_std': 0.3, #0.25,#0.27,#0.3
        # a default weak regularization prevents instability
        'l1_firing_rate': 0.0,
        # l2 regularization on activity
        'l2_firing_rate': 0.001,
        # l1 regularization on weight
        'l1_weight': 0.0,
        # l2 regularization on weight
        'l2_weight': 0.001,
        # number of recurrent units
        'n_rnn': 256,
        # learning rate
        'learning_rate': 0.00005, # If you use progressive training, learning rate is preset to 0.0005 in delay = [0, 0], the rest steps in progressive training, learning rate = 0.00005
        # random number generator
        'seed': seed,
        'rng': np.random.RandomState(seed),
        # Stop training if all three criteriers are met: 1. cost function smaller than stop_cost (or stop_noise_cost if sigma_rec > 0.0001 as usually is); 2. RNN's memorized color error smaller than stop_color_error (or stop_noise_color_error); 3. number of trials must larger than min_trials. Alternatively, training force to be stopped if number of trials exceed max_trials. In practice we set stop_cost and stop_color_error to be large so always satisfied, and only use min_trials and max_trials to control training.
        'stop_cost': 9999999,
        'stop_color_error': 999999,
        'stop_noise_cost': 999999,
        'stop_noise_color_error': 99999, # stop color error is for the training with no noise, while this one is the stop condition for training with noise
        'stop_delta_color_error': 99999, # stopping ceriteria for bias_method = delta training
        # if the number of trials is larger than max_trials, stop training
        'max_trials': 1e6,
        'min_trials': 2e5,
        # jacobian regulator strength. Negtive means we don't add this term
        'l2_jac': 1e-5,
        # a parameter used for computing the jacobian. Smaller faster, but also more inaccurate. Usually 1 would be good enough.
        'num_proj': 1,
        'num_unit': 12 # the number of inputs, without counting the go cue
    }
    if rule_name == 'color_reproduction_delay_unit':
        pass
        #hp['learning_rate'] = 0.0005
        #hp['stop_color_error'] = 3
        #hp['stop_cost'] = 0.002
    elif rule_name == 'color_reproduction_delay_tri':
        hp['learning_rate'] = 0.00005
    n_input, n_output = input_output_n(rule_name, hp)
    hp['n_input'] = n_input; hp['n_output'] = n_output

    return hp
