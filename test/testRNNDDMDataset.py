import context
from core.agent import Agent
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from core.tools import collate_fn
from core.RNN_dataset import RNN_Dataset

#class RNN_Dataset():
#    '''
#    output:
#      init_color (array [batch_size]): color in the begining of delay period
#      delay_time (array [batch_size]): delay time length
#      report_color (array [batch_size]): color in the end of delay period
#    '''
#    def __init__(self, batch_size):
#        self.batch_size = batch_size
#
#    def set_sub(self, sub):
#        self.sub = sub
#        self.dt = self.sub.hp['dt']
#        self.rnn_de = RNN_decoder()
#        self.rnn_de.read_rnn_agent(self.sub)
#
#    def __len__(self):
#        return 1
#
#    def __getitem__(self, idx):
#        # randomize input color
#        pca_degree = np.random.uniform(0, 360, size=self.batch_size)
#
#        sub.do_exp(prod_intervals=2000, ring_centers=pca_degree, sigma_rec=None, sigma_x=None)
#
#        delay_epoch = sub.epochs['interval']
#        state = sub.state[delay_epoch[0]:delay_epoch[1]] # (time_len, ring_centers, rnn_size)
#
#        delay_t = np.random.randint(0, delay_epoch[1] - delay_epoch[0], size=self.batch_size) # generate delay time interval
#
#        # decode init and end color
#        init_color = self.rnn_de.decode(state[0, :, :])
#
#        end_state = []
#        for i, t in enumerate(delay_t):
#            end_state.append(state[t, i, :])
#        end_state = np.concatenate(end_state, axis=0).reshape((self.batch_size, -1))
#
#        report_color = self.rnn_de.decode(end_state)
#
#        delay_t = torch.from_numpy(delay_t * self.dt)
#        init_color = torch.from_numpy(init_color)
#        report_color = torch.from_numpy(report_color)
#
#        return init_color, report_color, delay_t

model_dir = '../core/model_local/color_reproduction_delay_unit/model_17/noise_delta_stronger'
rule_name = 'color_reproduction_delay_unit'
batch_size = 160

input_set = RNN_Dataset(batch_size=batch_size) # 1 batch with batchsize 2
sub = Agent(model_dir, rule_name)
input_set.set_sub(sub)

input_loader = DataLoader(input_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

count = 0
for item in input_loader:
    print(item)
    count= count + 1
    if count > 10:
        break
