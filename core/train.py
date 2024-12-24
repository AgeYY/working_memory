"""Main training loop"""

from __future__ import division

import numpy as np
import torch
from torch.utils.data import DataLoader

from collections import defaultdict
import time
import sys
import os

from . import train_stepper
from . import network
from . import tools
from . import dataset
from .color_manager import Degree_color
from .color_manager import Color_cell

_color_list = ['blue', 'red', 'black', 'yellow', 'pink']

def decode_color_cones(y):
    """
    Population vector read-out of spatial location.
    """
    deg_color = Degree_color()
    return deg_color.lms2deg(y)

def decode_color_tri(y):
    """
    Population vector read-out of spatial location.
    """
    from core.color_manager import Color_triangular
    ctri = Color_triangular()
    return ctri.decode(y)

def decode_color_unit(y, unit):
    """
    Population vector read-out of spatial location.
    """
    ccell = Color_cell(unit)
    deg_color = ccell.decode(y)
    return deg_color

def get_perf_color(output, rule_name, target_dir, fix_start, fix_end, fix_strength=0.5, action_threshold = -1, response_duration=int(200/20), dire_on=False, hp=None):
    """
    fix_start, fix_end: start of perception epoch and end of go_cue. During this period the RNN's output units must keep silence. Then we calculate its preferred color using priod fix_end to fix_end + response_duration
    Get performance of spatial_reproduction task
    dire_on: output the target direction and the output direction
    fail_action is useless
    the default of action_threshold = -1, so that sin cos encoding can be also counted
    """
    batch_size = output.shape[1]
    action_at_fix = np.array([np.sum(output[fix_start[i]+3:fix_end[i]-3, i, :] > fix_strength) > 0 for i in range(batch_size)])

    no_action_at_motion = np.array([np.sum(output[fix_end[i]:fix_end[i]+response_duration, i, :] > action_threshold) == 0 for i in range(batch_size)])
    fail_action = action_at_fix + no_action_at_motion

    middle_duration = int(response_duration / 3.0) # the length of middle 1 /3 of response

    response_value = np.zeros((batch_size, output.shape[-1])) # output.shape[-1] is the number of output channels
    for i in range(batch_size):
        middle_res = output[(fix_end[i] + middle_duration):(fix_end[i] + 2 * middle_duration), i, :]
        # middle_res = output[(fix_end[i]):(fix_end[i] + response_duration), i, :]
        response_value[i, :] = np.mean(middle_res, axis=0) # we choose the mean of middle 1 / 3 as the RNN's response value

    if rule_name == 'color_reproduction_delay_32':
        direction = decode_color_unit(response_value, 32)
    if rule_name == 'color_reproduction_delay_unit':
        direction = decode_color_unit(response_value, hp['num_unit'])
    elif rule_name == 'color_reproduction_delay_cones':
        direction = np.array([decode_color_cones(response_value[i, :])[0] for i in range(batch_size)])
    elif rule_name == 'color_reproduction_delay_tri':
        direction = decode_color_tri(response_value)

    direction_err = np.minimum(np.abs(direction - target_dir), 360 - np.abs(direction - target_dir))

    success_action_prob = 1 - np.sum(fail_action)/batch_size
    mean_direction_err = np.mean(direction_err[np.argwhere(1 - fail_action)])

    if dire_on:
        return direction, target_dir
    else:
        return success_action_prob, mean_direction_err

class Trainer(object):
    def __init__(self, rule_name=None, model=None, hp=None, model_dir=None, is_cuda=True, device_id=0, out_dir=None, **kwargs):
        '''
        hp[stop_cost] (double): if the cost is smaller than this one, stop training.
        '''
        tools.mkdir_p(model_dir)
        self.model_dir = model_dir
        if out_dir is None: # output directory
            self.out_dir = self.model_dir
        tools.mkdir_p(self.out_dir)

        self.rule_name = rule_name
        self.is_cuda = is_cuda
        self.device_id = device_id

        if is_cuda:
            self.device = torch.device("cuda:{}".format(device_id))
        else:
            self.device = torch.device("cpu")

        # load or create hyper-parameters
        if hp is None:
            hp = tools.load_hp(model_dir)
        # hyper-parameters for time scale
        hp['alpha'] = 1.0 * hp['dt'] / hp['tau']
        if (hp['sigma_rec'] > 0.0001) or (hp['sigma_x'] > 0.0001):
            self.stop_color_error = hp['stop_noise_color_error']
            self.stop_cost = hp['stop_noise_cost']
        else:
            self.stop_color_error = hp['stop_color_error']
            self.stop_cost = hp['stop_cost']
        self.hp = hp

        fh_fname = os.path.join(model_dir, 'hp.json')
        tools.save_hp(hp, self.out_dir)

        kwargs['rule_name'] = rule_name
        # load or create model
        if model is None:
            self.model = network.RNN(hp, is_cuda, device_id=self.device_id, **kwargs)
            self.model.load(model_dir)
        else:
            self.model = model

        # load or create log
        self.log = tools.load_log(model_dir)
        if self.log is None:
            self.log = defaultdict(list)
        self.log['model_dir'] = self.out_dir

        # trainner stepper
        self.train_stepper = train_stepper.TrainStepper(self.model, self.hp, is_cuda, device_id=self.device_id)

        #print(rule_name)
        #print(type(rule_name))
        del kwargs['rule_name']
        # data loader
        dataset_train = dataset.TaskDataset(rule_name, hp, mode='train', is_cuda=is_cuda, device_id=self.device_id, **kwargs)
        dataset_test = dataset.TaskDataset(rule_name, hp, mode='test', is_cuda=is_cuda, device_id=self.device_id, **kwargs)

        self.dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, collate_fn=collate_fn)
        self.dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

        self.min_cost = np.inf

        self.model_save_idx = 0


    def do_eval(self):
        '''Do evaluation, and then save the model
        '''

        print('Trial {:7d}'.format(self.log['trials'][-1]) +
              '  | Time {:0.2f} s'.format(self.log['times'][-1]))

        for i_batch, sample_batched in enumerate(self.dataloader_test):
            '''training'''

            clsq_tmp = list()
            creg_tmp = list()

            if self.is_cuda:
                sample_batched['inputs'] = sample_batched['inputs'].cuda(self.device_id)
        
                sample_batched['target_outputs'] = sample_batched['target_outputs'].cuda(self.device_id)
                sample_batched['cost_mask'] = sample_batched['cost_mask'].cuda(self.device_id)
                sample_batched['seq_mask'] = sample_batched['seq_mask'].cuda(self.device_id)
                sample_batched['initial_state'] = sample_batched['initial_state'].cuda(self.device_id)

            sample_batched['rule_name'] = self.rule_name

            self.train_stepper.cost_fcn(**sample_batched)

            clsq_tmp.append(self.train_stepper.cost_lsq.detach().cpu().numpy())
            creg_tmp.append(self.train_stepper.cost_reg.detach().cpu().numpy())
            self.log['cost_'].append(np.mean(clsq_tmp, dtype=np.float64))
            self.log['creg_'].append(np.mean(creg_tmp, dtype=np.float64))

            # log['perf_' + rule_test].append(np.mean(perf_tmp, dtype=np.float64))
            print('| cost_lsq {:0.6f}'.format(np.mean(clsq_tmp)) +
                  '| c_reg {:0.6f}'.format(np.mean(creg_tmp)))

            sys.stdout.flush()

            #if clsq_tmp[-1] < self.min_cost:
            #    self.min_cost = clsq_tmp[-1]
                # Saving the model and log
            print('save model!')
            self.model.save(self.out_dir)

            tools.save_log(self.log)

            # basic timing tasks
            success_action_prob,  mean_color_err = get_perf_color(self.train_stepper.outputs.detach().cpu().numpy(), self.rule_name, sample_batched['sampled_degree'], sample_batched['epochs']['stim1'][1], sample_batched['epochs']['go_cue'][1], hp=self.hp)
            success_action_prob = success_action_prob.tolist()
            mean_color_err = mean_color_err.tolist()

            self.info = dict()
            self.info['cost'] = clsq_tmp[-1].tolist()
            self.info['creg'] = creg_tmp[-1].tolist()
            self.info['success_action_prob'] = success_action_prob
            self.info['mean_color_err'] = mean_color_err

            print('| success_action_prob {:0.6f}'.format(success_action_prob) +
                  '| mean_color_err {:0.6f}'.format(mean_color_err))
            if i_batch == 0:
                return clsq_tmp[-1], creg_tmp[-1], success_action_prob, mean_color_err

    def save_final_result(self, final_name='finalResult'):
        save_path = os.path.join(self.model_dir, final_name)
        tools.mkdir_p(save_path)
        self.model.save(save_path)
        #self.info['model_dir'] = save_path
        #tools.save_log(self.info)
        tools.save_hp(self.hp, save_path)

    def train(self, max_samples=1 * 1e6, min_samples = 2e5, display_step=500, max_model_save_idx=150):
        """Train the network.
        Args:
            max_sample: int, maximum number of training samples
            display_step: int, display steps
        Returns:
            model is stored at model_dir/model.ckpt
            training configuration is stored at model_dir/hp.json
            cost (float): the final cost
        """

        # Display hp
        for key, val in self.hp.items():
            print('{:20s} = '.format(key) + str(val))

        # Record time
        t_start = time.time()
        for step, sample_batched in enumerate(self.dataloader_train):
            try:
                if self.is_cuda:

                    sample_batched['inputs'] = sample_batched['inputs'].cuda(self.device_id)
                    sample_batched['target_outputs'] = sample_batched['target_outputs'].cuda(self.device_id)
                    sample_batched['cost_mask'] = sample_batched['cost_mask'].cuda(self.device_id)
                    sample_batched['seq_mask'] = sample_batched['seq_mask'].cuda(self.device_id)
                    sample_batched['initial_state'] = sample_batched['initial_state'].cuda(self.device_id)

                sample_batched['rule_name'] = self.rule_name

                self.train_stepper.stepper(**sample_batched)

                if step % display_step == 0:
                    
                    self.log['trials'].append(step * self.hp['batch_size_train'])
                    self.log['times'].append(time.time() - t_start)
                    clsq, creg, success_action_prob, mean_color_err = self.do_eval()
                    if not np.isfinite(creg):
                        return 'error', float('nan')
                    elif (clsq < self.stop_cost) and (mean_color_err < self.stop_color_error):
                        self.save_final_result()
                        if step * self.hp['batch_size_train'] > min_samples:
                            break
                    elif self.model_save_idx > max_model_save_idx and (step * self.hp['batch_size_train'] > min_samples):
                        break

                if step * self.hp['batch_size_train'] > max_samples:
                    self.log['trials'].append(step * self.hp['batch_size_train'])
                    self.log['times'].append(time.time() - t_start)
                    clsq, creg, _, _, = self.do_eval()
                    break

            except KeyboardInterrupt:
                print("Optimization interrupted by user")
                break

        print("Optimization finished!")

        return 'OK', clsq

def collate_fn(batch):
    return batch[0]

class ReTrainer(Trainer):
    ''' Do not replace RNN_size'''
    def __init__(self, model_dir, is_cuda=True, device_id=0, out_dir=None):
        self.model_dir = model_dir
        self.is_cuda = is_cuda
        self.device_id = device_id

        hp = tools.load_hp(model_dir)
        super().__init__(rule_name=hp['rule_name'], hp=hp, model_dir = model_dir, is_cuda=is_cuda, device_id=device_id)

    def replace_hp(self, hp_replace={}):
        for key, value in hp_replace.items():
            self.hp[key] = value

        if ( (self.hp['sigma_rec'] > 0.0001) or (self.hp['sigma_x'] > 0.0001) ) and (self.hp['bias_method'] == 'uniform'):
            self.stop_color_error = self.hp['stop_noise_color_error']
            self.stop_cost = self.hp['stop_noise_cost']
        elif ( (self.hp['sigma_rec'] > 0.0001) or (self.hp['sigma_x'] > 0.0001) ) and (self.hp['bias_method'] == 'delta'):
            self.stop_color_error = self.hp['stop_delta_color_error']
            self.stop_cost = self.hp['stop_noise_cost']
        else:
            self.stop_color_error = self.hp['stop_color_error']
            self.stop_cost = self.hp['stop_cost']

        hp = self.hp
        super().__init__(rule_name=hp['rule_name'], hp=hp, model_dir = self.model_dir, is_cuda=self.is_cuda, device_id=self.device_id, out_dir=self.out_dir)

    def train(self, max_samples=None, min_samples=None, display_step=200):
        tools.save_hp(self.hp, self.out_dir)
        if max_samples is None:
            max_samples = self.hp['max_trials']
        if min_samples is None:
            min_samples = self.hp['min_trials']
        super().train(max_samples=max_samples, min_samples = min_samples, display_step=display_step)

    def set_out_dir(self, out_dir):
        self.out_dir = out_dir
        tools.mkdir_p(self.out_dir)
        self.log['model_dir'] = self.out_dir

#if __name__ == '__main__':
#    # demo for retrainer, may not work
#    model_dir = '../core/model/color_reproduction_delay_unit/model_1/noise'
#    out_dir = '../core/model/color_reproduction_delay_unit/model_1/noise_delta'
#
#    retrainer = ReTrainer(model_dir)
#    retrainer.set_out_dir(out_dir)
#
#    hp_replace = {'bias_method':'vonmises', 'l2_jac': -1}
#    retrainer.replace_hp(hp_replace=hp_replace)
#    retrainer.train()
