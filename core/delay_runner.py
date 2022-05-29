# start from some states, then run RNN though delay, go, and reproduction period. The algorithm used is similar to the rnn_decoder
from core.agent import Agent
import core.dataset as dataset
from core.train import decode_color_unit, decode_color_cones, decode_color_tri
import numpy as np
import torch


class Delay_runner():
    def read_rnn_file(self, model_dir, rule_name):
        # read in the RNN from file
        self.rule_name = rule_name
        self.sub = Agent(model_dir, rule_name)
        self.hidden_size = self.sub.hp['n_rnn']

    def read_rnn_agent(self, agent):
        # read in the RNN from agent
        self.sub = agent
        self.rule_name = self.sub.rule_name
        self.hidden_size = self.sub.hp['n_rnn']

    def delay_run(self, init_states, prod_interval, sigma_rec=None, sigma_x=None):
        '''
        inputs:
          init_states (array [float] (batch_size, hidden_size)): the hidden states
          prod_interval (int): delay time length
        outputs:
          state (array [float] [n_delay_time, batch_size, hidden_size])
        '''
        self.batch_size = init_states.shape[0]
        inputs = self._gen_inputs(self.batch_size, prod_interval, sigma_x=sigma_x).float()

        init_states_torch = torch.as_tensor(init_states).float().to(device=self.sub.model.device)

        sigma_flag = False
        if sigma_rec is not None:
            sigma_rec_temp = self.sub.model.sigma_rec
            self.sub.model.sigma_rec = sigma_rec
            sigma_flag = True
        self.state_collector = self.sub.model(inputs, init_states_torch)

        self.state_binder = torch.cat(self.state_collector, dim=0).view(-1, self.batch_size, self.hidden_size)[self.epoch_delay['interval'][0] : self.epoch_delay['interval'][1]]

        self.firing_rate = self.sub.model.act_fcn(self.state_binder).detach().cpu().numpy()
        self.state = self.state_binder.detach().cpu().numpy()

        if sigma_flag:
            self.sub.model.sigma_rec = sigma_rec_temp # restore to the original sigma_rec
        return self.state.copy()

    def _gen_inputs(self, batch_size, prod_interval=800, noise_on=True, sigma_x=None):
        '''
        generate task trials contains only go_cue and response
        output:
        X (array [float] (time, batch_size, input_size)): input to the RNN
        '''
        sampled_degree = np.zeros(batch_size) # Only realated to the delay stage we do not need here. Therefore setting arbitary sampled_degree would be fine.
        kwargs = {
            'mode': 'test',
            'batch_size': batch_size,
            'prod_interval': prod_interval,
            'noise_on': noise_on,
            'sampled_degree': sampled_degree
        }
        sigma_flag = False
        if sigma_x is not None:
            sigma_x_temp = self.sub.hp['sigma_x']
            self.sub.hp['sigma_x'] = sigma_x
            sigma_flag = True

        dataset_test = dataset.TaskDatasetForRun(self.rule_name, self.sub.hp.copy(), **kwargs)
        dataset_test.__getitem__()

        epochs_batch = dataset_test.trial.epochs
        epochs = {}
        for key in epochs_batch:
            if key == 'fix':
                epochs[key] = [0, epochs_batch[key][1][0]]
            else:
                epochs[key] = [epochs_batch[key][0][0], epochs_batch[key][1][0]]

        X = torch.as_tensor(dataset_test.trial.x[epochs['interval'][0]:])
        # shift epochs
        interval_end = epochs['interval'][1] - epochs['interval'][0]
        go_cue_start = epochs['go_cue'][0] - epochs['interval'][0]
        go_cue_end = epochs['go_cue'][1] - epochs['interval'][0]
        res_start = epochs['response'][0] - epochs['interval'][0]
        res_end = epochs['response'][1] - epochs['interval'][0]

        self.epoch_delay = {'interval': [0, interval_end] ,'go_cue': [go_cue_start, go_cue_end], 'response': [res_start, res_end]}

        if sigma_flag:
            self.sub.hp['sigma_x'] = sigma_x_temp # restore to the original sigmax

        return X.to(device=self.sub.model.device)

    def decode_from_res(self, output, res_start, res_end):
        '''decode from the output'''
        response_value = np.mean(output[res_start:res_end], axis=0)

        if self.rule_name == 'color_reproduction_delay_32':
            direction = decode_color_unit(response_value, 32)
        if self.rule_name == 'color_reproduction_delay_unit':
            direction = decode_color_unit(response_value, self.sub.hp['num_unit'])
        elif self.rule_name == 'color_reproduction_delay_cones':
            direction = np.array([decode_color_cones(response_value[i, :])[0] for i in range(self.batch_size)])
        elif self.rule_name == 'color_reproduction_delay_tri':
            direction = decode_color_tri(response_value)
        return direction
