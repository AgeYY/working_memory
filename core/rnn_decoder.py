# Give a hidden state, this program will run the rnn from the go_cue period, and output the response. This response would then be decoded by the population vector.
from core.agent import Agent
import core.dataset as dataset
from core.train import decode_color_unit, decode_color_cones, decode_color_tri
import copy
import numpy as np
import torch


class RNN_decoder():
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

    def decode(self, init_states, decoding_plane='interval'):
        '''
        inputs:
          init_states (array [float] (batch_size, hidden_size)): the hidden states
          decoding_plane (str): interval will pick the PC1-PC2 of the last states in delay, 'response' will pick last states in response
        outputs:
          output (array [float] (batch_size)): decoded color
        '''
        sigma_rec_temp = self.sub.model.sigma_rec
        self.sub.model.sigma_rec = 0
        self.batch_size = init_states.shape[0]

        # generate inputs
        if decoding_plane == 'response':
            inputs = None
        else:
            inputs = self._gen_inputs(self.batch_size).float()

        # generate outputs
        outputs = self._get_output(init_states, inputs=inputs)

        # decode outputs to color
        if decoding_plane == 'response':
            res_start, res_end = 0, 1
        else:
            res_start, res_end = self.epoch_decode['response'][0], self.epoch_decode['response'][1]

        report_color = self.decode_from_res(outputs, res_start, res_end)

        self.behaviour = {}
        self.behaviour['report_color'] = report_color

        self.sub.model.sigma_rec = sigma_rec_temp
        return report_color

    def _gen_inputs(self, batch_size):
        '''
        generate task trials contains only go_cue and response
        output:
        X (array [float] (time, batch_size, input_size)): input to the RNN
        '''
        sampled_degree = np.zeros(batch_size) # Only realated to the delay stage we do not need here. Therefore setting arbitary sampled_degree would be fine.
        kwargs = {
            'mode': 'test',
            'batch_size': batch_size,
            'prod_interval': 0,
            'noise_on': False,
            'sampled_degree': sampled_degree
        }

        dataset_test = dataset.TaskDatasetForRun(self.rule_name, self.sub.hp, **kwargs)
        dataset_test.__getitem__()

        epochs_batch = dataset_test.trial.epochs
        epochs = {}
        for key in epochs_batch:
            if key == 'fix':
                epochs[key] = [0, epochs_batch[key][1][0]]
            else:
                epochs[key] = [epochs_batch[key][0][0], epochs_batch[key][1][0]]


        X = torch.as_tensor(dataset_test.trial.x[epochs['go_cue'][0]:])
        # shift epochs
        go_cue_end = epochs['go_cue'][1] - epochs['go_cue'][0]
        res_start = epochs['response'][0] - epochs['go_cue'][0]
        res_end = epochs['response'][1] - epochs['go_cue'][0]

        self.epoch_decode = {'go_cue': [0, go_cue_end], 'response': [res_start, res_end]}

        return X.to(device=self.sub.model.device)

    def _get_output(self, init_states, inputs=None):
        '''
        inputs to the RNN
        input:
          init_states (array_like [1, n_batch, hidden_size])
          inputs (torch_tensor [n_time_step, n_batch, 13]): 13 in the last dimension means 12 perception neurons plus one go neuron. If None, init_states will be directly readout by readout matrix
        output:
          outputs (array like [n_time_step, n_batch, 12]): 12 means 12 output neurons
        '''
        init_states_torch = torch.as_tensor(init_states).float().to(device=self.sub.model.device)

        if inputs is None:
            state_collector = init_states_torch.view(-1, self.batch_size, self.hidden_size).to(device=self.sub.model.device)
            state_binder = state_collector
        else:
            state_collector = self.sub.model(inputs, init_states_torch)
            state_binder = torch.cat(state_collector, dim=0).view(-1, self.batch_size, self.hidden_size)

        firing_rate_binder = self.sub.model.act_fcn(state_binder)

        outputs = self.sub.model.out_act_fcn(torch.matmul(firing_rate_binder, self.sub.model.weight_out) + self.sub.model.bias_out)

        outputs = outputs.detach().cpu().numpy()
        return outputs

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
