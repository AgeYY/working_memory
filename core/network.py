"""Definition of the network model and various RNN cells"""
from __future__ import division

import torch
from torch import nn
from core.jacobian import JacobianReg as JReg

import os
import math
import numpy as np

from . import tools


# Create Network
class RNN(nn.Module):
    def __init__(self, hp, is_cuda=True, device_id=0, **kwargs):
        super(RNN, self).__init__()

        input_size = hp['n_input']
        hidden_size = hp['n_rnn']
        output_size = hp['n_output']
        alpha = hp['alpha']
        sigma_rec = hp['sigma_rec']
        act_fcn = hp['activation']
        out_act_fcn = hp['out_activation']

        self.hp = hp

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.device_id = device_id

        if is_cuda:
            self.device = torch.device("cuda:{}".format(device_id))
        else:
            self.device = torch.device("cpu")

        self._clamp_thres = torch.tensor(100000000., device=self.device)
        self._high_value_coeff = torch.tensor(0.5, device=self.device)

        if act_fcn == 'relu':
            self.act_fcn = lambda x: nn.functional.relu(x)
        elif act_fcn == 'softplus':
            self.act_fcn = lambda x: nn.functional.softplus(x)
        elif act_fcn == 'rec_tanh':
            self.act_fcn = lambda x: nn.functional.threshold(
                torch.tanh(x), 0, 0
            )
        elif act_fcn == 'sigmoid':
            self.act_fcn = lambda x: nn.functional.sigmoid(x)
        elif act_fcn == 'tanh':
            self.act_fcn = lambda x: torch.tanh(x)
        elif act_fcn == 'ReLU6':
            self.act_fcn = nn.ReLU6()

        if out_act_fcn == 'relu':
            self.out_act_fcn = lambda x: nn.functional.relu(x)
        elif out_act_fcn == 'softplus':
            self.out_act_fcn = lambda x: nn.functional.softplus(x)
        elif out_act_fcn == 'linear':
            self.out_act_fcn = lambda x: x
        elif out_act_fcn == 'sigmoid':
            self.out_act_fcn = lambda x: nn.functional.sigmoid(x)

        # init weight
        weight_ih = torch.empty(input_size, hidden_size).uniform_(-1./math.sqrt(2.), 1./math.sqrt(2.))
        weight_ih[0:1, :].uniform_(-1./math.sqrt(1.), 1./math.sqrt(1.)) # I don't know why its 0:1
        self.weight_ih = nn.Parameter(weight_ih)

        hh_mask = torch.ones(hidden_size, hidden_size) - torch.eye(hidden_size)
        non_diag = torch.empty(hidden_size, hidden_size).normal_(0, hp['initial_std']/math.sqrt(hidden_size))
        weight_hh = hh_mask * non_diag

        self.weight_hh = nn.Parameter(weight_hh)
        #.to(self.device)
        self.bias_h = nn.Parameter(torch.zeros(1, hidden_size))
        #.to(self.device)

        self.weight_out = nn.Parameter(torch.empty(hidden_size, output_size).normal_(0., 0.4/math.sqrt(hidden_size)))
        #.to(self.device)
        self.bias_out = nn.Parameter(torch.zeros(output_size,))
        #.to(self.device)

        self.alpha = torch.tensor(alpha, device=self.device)
        self.sigma_rec = torch.tensor(math.sqrt(2./alpha) * sigma_rec, device=self.device)

        self._0 = torch.tensor(0., device=self.device)
        self._1 = torch.tensor(1., device=self.device)

    def recurrence(self, inputs, hidden):
        """Recurrence helper."""
        #hidden_new = torch.matmul(self.act_fcn(hidden) + 1, self.weight_hh) + self.bias_h + \
        #    torch.matmul(inputs, self.weight_ih) + torch.randn_like(hidden, device=self.device) * self.sigma_rec
        hidden_new = torch.matmul(self.act_fcn(hidden), self.weight_hh) + self.bias_h + \
            torch.matmul(inputs, self.weight_ih) + torch.randn_like(hidden, device=self.device) * self.sigma_rec

        hidden_new = (self._1 - self.alpha) * hidden + self.alpha * hidden_new
        return hidden_new

    def forward(self, inputs, initial_state):

        """Most basic RNN: output = new_state = W_input * input + W_rec * act(state) + B + noise """

        # shape: (batch_size, hidden_size)
        state = initial_state
        state_collector = [state]

        mask = torch.eye(self.hidden_size, self.hidden_size, device=self.device).bool()
        weight_hh_new = self.weight_hh.masked_fill(mask, 0) # silent the diagnal element of weight_hh

        for input_per_step in inputs:
            #state_new = torch.matmul(self.act_fcn(state) + 1, weight_hh_new) + self.bias_h + \
            state_new = torch.matmul(self.act_fcn(state), weight_hh_new) + self.bias_h + \
                        torch.matmul(input_per_step, self.weight_ih) + torch.randn_like(state, device=self.device) * self.sigma_rec

            state = (self._1 - self.alpha) * state + self.alpha * state_new
            state_collector.append(state)

        return state_collector

    def out_weight_clipper(self):
        self.weight_out.data.clamp_(0.)

    def self_weight_clipper(self):
        #diag_element = self.weight_hh.diag().data.clamp_(0., 1.)
        #self.weight_hh.data[range(self.hidden_size), range(self.hidden_size)] = diag_element
        pass

    def save(self, model_dir):
        save_path = os.path.join(model_dir, 'model.pth')
        torch.save(self.state_dict(), save_path)

    def load(self, model_dir):
        if model_dir is not None:
            save_path = os.path.join(model_dir, 'model.pth')
            if os.path.isfile(save_path):
                self.load_state_dict(torch.load(save_path, map_location=lambda storage, loc: storage), strict=False)

