import numpy as np
import core.dataset as dataset
from core.agent import Agent
import torch
import copy

class State_Evolver():
    def read_rnn_file(self, model_dir, rule_name):
        # read in the RNN from file
        self.rule_name = rule_name
        self.sub = Agent(model_dir, rule_name)
        self.hidden_size = self.sub.hp['n_rnn']
        self.hp_replace = copy.deepcopy(self.sub.hp)

    def set_trial_para(self, prod_interval=None, pulse_duration=None, pulse_duration_go=None, response_duration=None):
        '''
        input:
          prod_interval (int): length of the delay period
          pulse_duration (int): length of the perception
          pulse_duration_go (int): length of the go cue
          response_duration (int): length of response
        '''
        if prod_interval is not None: self.hp_replace['prod_interval'] = [0, prod_interval]
        if pulse_duration is not None: self.hp_replace['pulse_duration'] = pulse_duration
        if pulse_duration_go is not None: self.hp_replace['pulse_duration_go'] = pulse_duration_go
        if response_duration is not None: self.hp_replace['response_duration'] = response_duration

    def _get_input(self, batch_size, evolve_period):
        '''
          evolve_period (list): evolve period starts from the begining of evolve_period[0] to the end of evolve_period[1]. available periods: 'fix', 'stim1', 'interval', 'go_cue', 'go', 'response'
        '''
        sampled_degree = np.zeros(batch_size) # Only realated to the delay stage we do not need here. Therefore setting arbitary sampled_degree would be fine.
        kwargs = {
            'mode': 'test',
            'batch_size': batch_size,
            'noise_on': False,
            'sampled_degree': sampled_degree,
            'prod_interval': self.hp_replace['prod_interval'][1]
        }

        dataset_test = dataset.TaskDatasetForRun(self.rule_name, self.hp_replace, **kwargs)
        dataset_test.__getitem__()

        epochs_batch = dataset_test.trial.epochs
        epochs = {}
        for key in epochs_batch:
            if key == 'fix':
                epochs[key] = [0, epochs_batch[key][1][0]]
            else:
                epochs[key] = [epochs_batch[key][0][0], epochs_batch[key][1][0]]
        input_start, input_end = epochs[evolve_period[0]][0], epochs[evolve_period[0]][1]
        inputs = dataset_test.trial.x[input_start:input_end]
        inputs = torch.as_tensor(inputs).to(device=self.sub.model.device)
        return inputs

    def evolve(self, init_state, evolve_period):
        '''
        input:
          init_state (np array of shape [n_batch, hidden_size]): initial recurrent neurons states to be evolved. Hidden_size is the number of recurrent neurons
          evolve_period (list): evolve period starts from the begining of evolve_period[0] to the end of evolve_period[1]. available periods: 'fix', 'stim1', 'interval', 'go_cue', 'go', 'response'
        output:
          state_binder (np array with shape [n_time, n_batch, hidden_size])
        '''
        batch_size = init_state.shape[0]
        inputs = self._get_input(batch_size, evolve_period=evolve_period)
        init_states_torch = torch.as_tensor(init_state).float().to(device=self.sub.model.device)

        state_collector = self.sub.model(inputs, init_states_torch)
        state_binder = torch.cat(state_collector, dim=0).view(-1, batch_size, self.hidden_size)
        state_binder = state_binder.detach().cpu().numpy()
        return state_binder


def evolve_recurrent_state(sub, rule_name, init_state, prod_interval=None, pulse_duration=None, pulse_duration_go=None, response_duration=None, evolve_period=['interval', 'interval']):
    '''
    inputs:
      sub: agent class
      rule_name: str
    '''
    ###### Get input
    hp_replace = copy.deepcopy(sub.hp)
    if prod_interval is not None: hp_replace['prod_interval'] = [0, prod_interval]
    if pulse_duration is not None: hp_replace['pulse_duration'] = pulse_duration
    if pulse_duration_go is not None: hp_replace['pulse_duration_go'] = pulse_duration_go
    if response_duration is not None: hp_replace['response_duration'] = response_duration

    batch_size = init_state.shape[0]
    hidden_size = init_state.shape[1]
    sampled_degree = np.zeros(batch_size) # Only realated to the delay stage we do not need here. Therefore setting arbitary sampled_degree would be fine.
    kwargs = {
        'mode': 'test',
        'batch_size': batch_size,
        'noise_on': False,
        'sampled_degree': sampled_degree,
        'prod_interval': hp_replace['prod_interval'][1]
    }

    dataset_test = dataset.TaskDatasetForRun(rule_name, hp_replace, **kwargs)
    dataset_test.__getitem__()

    epochs_batch = dataset_test.trial.epochs
    epochs = {}
    for key in epochs_batch:
        if key == 'fix':
            epochs[key] = [0, epochs_batch[key][1][0]]
        else:
            epochs[key] = [epochs_batch[key][0][0], epochs_batch[key][1][0]]
    input_start, input_end = epochs[evolve_period[0]][0], epochs[evolve_period[0]][1]
    inputs = dataset_test.trial.x[input_start:input_end]
    inputs = torch.as_tensor(inputs).to(device=sub.model.device)

    ###### evolve states
    init_states_torch = torch.as_tensor(init_state).float().to(device=sub.model.device)
    state_collector = sub.model(inputs, init_states_torch)
    state_binder = torch.cat(state_collector, dim=0).view(-1, batch_size, hidden_size)
    state_binder = state_binder.detach().cpu().numpy()
    return state_binder
