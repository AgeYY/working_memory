# train the model with specific hp. Although this file is called cluster, but is also used in training on you local computer
from os import path as ospath
from core import task
from core import network
from core import train
from core import default

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

    parent_folder = ospath.join(model_base, rule_name)
    local_folder_name = ospath.join(model_base, rule_name, 'model_'+str(index))

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
