"""Main training loop"""

from __future__ import division


import torch


from . import train_stepper
from . import network
from . import tools
from . import dataset


_color_list = ['blue', 'red', 'black', 'yellow', 'pink']


class Runner(object):
    def __init__(self, rule_name=None, model=None, hp=None, model_dir=None, is_cuda=True, device_id=0, noise_on=True, mode='test', sigma_x=None, **kwargs):
        tools.mkdir_p(model_dir)
        self.model_dir = model_dir

        self.rule_name = rule_name
        self.is_cuda = is_cuda
        self.device_id = device_id

        # load or create hyper-parameters
        if hp is None:
            hp = tools.load_hp(model_dir)
        # hyper-parameters for time scale
        hp['alpha'] = 1.0 * hp['dt'] / hp['tau']
        self.hp = hp

        self.noise_on = noise_on

        # load or create model
        if model is None:
            if hp['rnn_type'] == 'RNN':
                self.model = network.RNN(hp, is_cuda, device_id=self.device_id, rule_name=rule_name, **kwargs)
            self.model.load(model_dir)
        else:
            self.model = model

        if not noise_on:
            self.model.sigma_rec = 0
        if sigma_x is not None:
            self.hp['sigma_x'] = sigma_x

        # trainner stepper
        self.train_stepper = train_stepper.TrainStepper(self.model, self.hp, is_cuda, device_id=self.device_id)

        self.mode = mode

    def run(self, **kwargs):
        '''
        Run the model, returns task trial and self.train_stepper
        '''

        # data loader
        self.dataset = dataset.TaskDatasetForRun(self.rule_name, self.hp, noise_on=self.noise_on, mode=self.mode, **kwargs)

        sample = self.dataset.__getitem__()

        if self.is_cuda:
            sample['inputs'] = sample['inputs'].cuda(self.device_id)
            sample['target_outputs'] = sample['target_outputs'].cuda(self.device_id)
            sample['cost_mask'] = sample['cost_mask'].cuda(self.device_id)
            sample['seq_mask'] = sample['seq_mask'].cuda(self.device_id)
            sample['initial_state'] = sample['initial_state'].cuda(self.device_id)

        sample['rule_name'] = self.rule_name

        with torch.no_grad():
            self.train_stepper.cost_fcn(**sample)

        return self.dataset.trial, self.train_stepper
