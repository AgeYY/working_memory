# train the model with specific hp. Although this file is called cluster, but is also used in training on you local computer
import os
import sys
import copy

sys.path.append(os.getcwd())

from core import task
from core import network
from core import train
from core import default

def train_model(rule_name, index, hp_replace={}, n_init_trial=500, is_cuda=True, device_id=0, final_name='finalResult', model_base= '../core/model', **kwargs):
    '''
    training the model
    input:
      rule_name (str):
      hp_replace (dict): the default hp is read from default.get_default_hp, but some parameters can be replaced by hp_replace. It is a dictionary as {'para0': val0, 'para1': val1, ...}
      index (int): the index added to the filename. The output filename would be 'model_str(index)'
      kwargs: kwargs for training
      n_init_trial (int): the max number of trials that initialize the model.
      final_name (str): beside saving the final result in rule_name/model_i. this function will also back up the final result in rule_name/model_i/final_name
    return:
      model.pth, hp.jason, log.jason: files in os.path.join('./core/model', rule_name, 'model_' + sre(index))
      label.csv: a description of parameters and perfomence of different models. It has the following structure:
        index, hp_replace, cost
        0, {'para0': val0, 'para1': val1, ...}, 0.0002
        3, {'para0': val0, 'para1': val1, ...}, 0.003
    '''

    parent_folder = os.path.join(model_base, rule_name)
    local_folder_name = os.path.join(model_base, rule_name, 'model_'+str(index))

    for i in range(n_init_trial):
        hp = default.get_default_hp(rule_name)
        replace_hp(hp, hp_replace, inplace=True)

        trainerObj = train.Trainer(model_dir=local_folder_name, rule_name=rule_name, hp=hp, **kwargs, is_cuda=is_cuda, device_id=device_id)
        stat, cost = trainerObj.train(max_samples=hp['max_trials'], min_samples=hp['min_trials'], display_step=200)

        trainerObj.save_final_result(final_name)

        if stat == 'OK':
            break
        else:
            run_cmd = 'rm -r ' + local_folder_name
            os.system(run_cmd)
    return cost

def two_step_training(rule_name, index, hp_replace={}, n_init_trial=500, is_cuda=True, device_id=0, **kwargs):
    '''
    Train the model with two steps. In the first step, prod_interval = [hp[prod_interval][0], hp[prod_interval][0]]. And no regularization terms. This step enable the model to find an approximate solution.

    Secondly, the model would be trained prod_interval = hp[prod_interval]
    '''

    hp_replace_step1 = hp_replace
    try:
        hp_replace_step1['prod_interval'] = [hp_replace['prod_interval'][0], hp_replace['prod_interval'][0]]
    except KeyError:
        hp = default.get_default_hp(rule_name)
        hp_replace_step1['prod_interval'] = [hp['prod_interval'][0], hp['prod_interval'][0]]

    hp_replace_step1['l1_firing_rate'] = -1e-5;
    hp_replace_step1['l2_firing_rate'] = -1e-5;
    hp_replace_step1['l1_weight'] = -1e-5;
    hp_replace_step1['l2_weight'] = -1e-5;
    hp_replace_step1['l2_jac'] = -1e-5;
    hp_replace_step1['sigma_rec'] = 0.05;

    # step 1,2
    cost = train_model(rule_name, index, hp_replace=hp_replace_step1, n_init_trial=n_init_trial, is_cuda=is_cuda, device_id=device_id, **kwargs)
    cost = train_model(rule_name, index, hp_replace=hp_replace, n_init_trial=n_init_trial, is_cuda=is_cuda, device_id=device_id, **kwargs)

    return cost

def four_step_training(rule_name, index, hp_replace={}, n_init_trial=500, is_cuda=True, device_id=0, **kwargs):
    '''
    Train the model with two steps. In the first step, prod_interval = [hp[prod_interval][0], hp[prod_interval][0]]. And no regularization terms. This step enable the model to find an approximate solution.

    Secondly, the model would be trained prod_interval = hp[prod_interval]
    '''

    hp_replace_step2 = copy.deepcopy(hp_replace)

    hp_replace_step2['sigma_rec'] = 0.05;

    hp_replace_step1 = copy.deepcopy(hp_replace_step2)

    hp_replace_step1['l1_firing_rate'] = -1e-5;
    hp_replace_step1['l2_firing_rate'] = -1e-5;
    hp_replace_step1['l1_weight'] = -1e-5;
    hp_replace_step1['l2_weight'] = -1e-5;
    hp_replace_step1['l2_jac'] = -1e-5;

    hp_replace_step1['prod_interval'] = [0, 0]

    # four steps
    cost = train_model(rule_name, index, hp_replace=hp_replace_step1, n_init_trial=n_init_trial, is_cuda=is_cuda, device_id=device_id, **kwargs)
    cost = train_model(rule_name, index, hp_replace=hp_replace_step2, n_init_trial=n_init_trial, is_cuda=is_cuda, device_id=device_id, **kwargs)
    cost = train_model(rule_name, index, hp_replace=hp_replace, n_init_trial=n_init_trial, is_cuda=is_cuda, device_id=device_id, **kwargs)

    return cost

def progressive_train(rule_name, index, hp_replace={}, n_init_trial=500, is_cuda=True, device_id=0, noise_delta_model=True, bias_method='delta', **kwargs):
    '''
    Train the model progressively.
    In the first stage, the noise and l2 regularization would be turned off. The model would be trained with uniform color, with the upperbound of prod_interval increase 1000 every step.
    If noise_delta_model is True, this program will further retrain the model to 2nd and 3rd stage.
    In 2nd stage, noise and l2 regularization will be turned on, with the value either be specified in hp_replace or in the default.py
    In the final stage, training with delta distributed color
    '''

    # read the default prod_step, sigma_rec_step, sigma_rec and prod_interval
    prod_step, sigma_rec_step, sigma_rec, prod_interval = set_steps(rule_name, hp_replace)

    hp_replace_prog = hp_replace.copy()
    hp_replace_prog['prod_interval'] = [0, 0]
    hp_replace_prog['sigma_rec'] = 0.
    hp_replace_prog['sigma_x'] = 0
    hp_replace_prog['l2_jac'] = -0.1
    hp_replace_prog['l2_weight'] = -0.1
    hp_replace_prog['l2_firing_rate'] = -0.1
    hp_replace_prog['bias_method'] = 'uniform'
    hp_replace_prog['learning_rate'] = 0.0005 # In the first step we need larger learning rate, to find the solution.

    for i in range(prod_interval[1] // prod_step + 1):
        cost = train_model(rule_name, index, hp_replace=hp_replace_prog, n_init_trial=n_init_trial, is_cuda=is_cuda, device_id=device_id, final_name='progressive_' + str(i), **kwargs)

        hp_replace_prog['prod_interval'][1] = hp_replace_prog['prod_interval'][1] + prod_step
        hp_replace_prog['learning_rate'] = 0.00005

    if noise_delta_model:
        hp_replace_noise = hp_replace.copy()
        hp_replace_noise['bias_method'] = 'uniform'
        cost = train_model(rule_name, index, hp_replace=hp_replace_noise, n_init_trial=n_init_trial, is_cuda=is_cuda, device_id=device_id, final_name='noise', **kwargs)

        hp_replace_noise['bias_method'] = bias_method
        cost = train_model(rule_name, index, hp_replace=hp_replace_noise, n_init_trial=n_init_trial, is_cuda=is_cuda, device_id=device_id, final_name='noise_delta', **kwargs)

    return cost

def replace_hp(hp, hp_replace, inplace=False):
    '''replace part of hp with hp_replace.'''
    if inplace == True:
        for key, value in hp_replace.items():
            hp[key] = value
        return None
    else:
        hp_temp = hp.copy()
        for key, value in hp_replace.items():
            hp_temp[key] = value
        return hp_temp

def set_steps(rule_name, hp_replace):
    '''
    set up the parameters for progressive training. If the parameter exists in the hp_replace, we pick it, otherwise read from defaul.
    '''
    para_keys = ['prod_step', 'sigma_rec_step', 'sigma_rec', 'prod_interval']
    items = []

    hp = default.get_default_hp(rule_name)
    for key in para_keys:
        try:
            items.append(hp_replace[key])
        except KeyError:
            items.append(hp[key])
    return items

if __name__ == "__main__":
    print(sys.argv)
    rule_name = sys.argv[1]
    index = int(sys.argv[2])

    train_model(rule_name, index)

