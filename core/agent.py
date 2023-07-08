import numpy as np
from core import run
import core.tools as tools
from core import train
import core.network as network
from core.color_error import Color_error
from os import walk
from os import path
import sys

class Agent():
    '''a agent for doing the expeiment. It is actually a rnn model. the outputs are behaviour data and neuron firing rate'''
    def __init__(self, model_dir, rule_name, is_cuda=True):
        '''
        load the model and some hyperparameters
        '''
        self.model_dir = model_dir
        self.rule_name = rule_name
        self.is_cuda = is_cuda
        hp = tools.load_hp(model_dir)
        self.hp = hp
        self.model = network.RNN(hp, is_cuda, rule_name=rule_name)

        if is_cuda:
            self.model.cuda(device=self.model.device)

        self.model.load(model_dir)
        self.model.sigma_rec = hp['sigma_rec']

    def do_exp(self, prod_intervals=1000, ring_centers = np.array([6.]), sigma_rec=None, sigma_x=None):
        '''
        run the model and get the firing rate.
        input:
          rule_name (str): rule name can be color_response_delay_32 or color_response_delay_cones
          model_dir (str):
          prod_interval (array [float] (1)): the production interval.
          noise_on (bool): the input noise. Note the hidden noise will always exists unless you set hp['noise_on'] to false
          ring_centers (array [float] (m)): the input color with unit degree.
          is_cuda (bool): run on the cuda or not
        output:
          fir_rate ( array [float] (time_len, ring_centers, rnn_size) )
          input_colors
          epochs (dic) includes:
               - fix: [0, fix_end]
               - stim1: [start, end]
               - interval: [start, end]
               - go_cue: [start, end]
               - response: [start, end]
          behaviour (dic) includes:
               - report_color: 1 dimensional array
               - target_color: 1 dimensional array. The target color is actually the same as input_color
        '''
        #### Prepare parameters
        prod_intervals = np.array(prod_intervals)
        prod_intervals, ring_centers = np.meshgrid(prod_intervals, ring_centers)
        prod_intervals = prod_intervals.flatten()
        ring_centers = ring_centers.flatten()

        batch_size = len(prod_intervals)

        sigma_rec_temp = self.model.sigma_rec
        if sigma_rec is not None:
            self.model.sigma_rec=sigma_rec

        #### Run the model
        runnerObj = run.Runner(model_dir=self.model_dir, rule_name=self.rule_name, is_cuda=self.is_cuda, noise_on=True, model=self.model, sigma_x=sigma_x)
        trial_input, run_result = runnerObj.run(batch_size=batch_size, prod_interval=prod_intervals, sampled_degree=ring_centers) # will dly_interval also be passed to the run?

        #### Transform the results
        ## In this function, the time point of each trial should be the same, so we only take a number to present the whole time points of batch (different ring centers).
        epochs_batch = trial_input.epochs
        self.epochs = {}
        for key in epochs_batch:
            if key == 'fix':
                self.epochs[key] = [0, epochs_batch[key][1][0]]
            else:
                self.epochs[key] = [epochs_batch[key][0][0], epochs_batch[key][1][0]]

        self.fir_rate = run_result.firing_rate_binder.detach().cpu().numpy()
        self.state = run_result.state_binder.detach().cpu().numpy()

        out_puts = run_result.outputs.detach().cpu().numpy()
        report_color, target_color = train.get_perf_color(out_puts, self.rule_name, trial_input.sampled_degree, trial_input.epochs['stim1'][1], trial_input.epochs['go'][1], dire_on=True, hp=runnerObj.hp)

        self.behaviour = {}
        self.behaviour['target_color'] = target_color
        self.behaviour['report_color'] = report_color

        self.model.sigma_rec = sigma_rec_temp
        return self.fir_rate, self.behaviour, self.epochs

    def do_batch_exp(self, prod_intervals=800, sigma_rec=None, sigma_x=None, batch_size=2000, sample_method='linspace', bin_size=1):
        '''
        do batch experiment
        input:
          sample_method: string or float. Currently string only supports linspace where the experiment input color will be sample by linspace from 0 to 360, with size equals to batch_size. If float, the input color will be fixed to this and repeatly do batch_size trials
        output:
          dire_df (dic): with report_dire, target_dire and error_dire
        '''
        batch_size_unit = np.minimum(200, batch_size) # too large will exceed the memory

        report_color_list = []
        target_color_list = []

        if sample_method == 'linspace':
            ring_centers = np.linspace(0, 360, batch_size)
        else:
            ring_centers = np.ones(batch_size) * sample_method

        for i in range(int(batch_size / batch_size_unit)):
            ring_centers_unit = ring_centers[i*batch_size_unit:(i+1)*batch_size_unit]
            self.do_exp(prod_intervals=prod_intervals, ring_centers=ring_centers_unit, sigma_rec=sigma_rec, sigma_x=sigma_x)
            report_color_temp, target_color_temp = self.behaviour['report_color'].copy(), self.behaviour['target_color'].copy()

            report_color_list.append(report_color_temp)
            target_color_list.append(target_color_temp)

        report_color_flat = np.array(report_color_list).flatten()
        target_color_flat = np.array(target_color_list).flatten()

        if bin_size is not None:
            target_color_flat = target_color_flat // bin_size * bin_size + bin_size // 2

        # calculate the error
        color_error = Color_error()
        color_error.add_data(report_color_flat, target_color_flat)
        error_color = color_error.calculate_error()

        self.behaviour_batch = {}
        self.behaviour_batch['report_color'] = report_color_flat
        self.behaviour_batch['target_color'] = target_color_flat
        self.behaviour_batch['error_color'] = error_color

class Agent_group():
    '''
    read agents from a directory to a list
    '''
    def __init__(self, model_dir, rule_name, sub_dir='', is_cuda=True):
        '''
        model_dir (str): a model dir has sub directories model_0/sub_dir, model_1/sub_dir, ..., each contains one model.
        '''
        self.group = []
        root, dirs, _ = next(walk(model_dir))
        for dir_name in dirs:
            full_model_dir = root + '/' + dir_name + sub_dir
            if path.exists(full_model_dir):
                self.group.append(Agent(full_model_dir, rule_name, is_cuda=is_cuda))

    def do_batch_exp(self, prod_intervals=800, sigma_rec=None, sigma_x=None, batch_size=2000, sample_method='linspace', bin_size=1):
        '''
        do batch experiment for every subjects, and concate their behavior data together. See the detail of this function in class Agent
        '''
        self.group_behaviour = {'report_color': [], 'target_color': [], 'error_color': []}
        i = 0
        for sub in self.group:
            sub.do_batch_exp(prod_intervals=prod_intervals, sigma_rec=sigma_rec, batch_size=batch_size, sigma_x=sigma_x, sample_method=sample_method, bin_size=bin_size)
            for key in self.group_behaviour:
                self.group_behaviour[key].extend(list(sub.behaviour_batch[key]))

        for key in self.group_behaviour:
            self.group_behaviour[key] = np.array(self.group_behaviour[key])

class Agent_loader():
    def read_rnn_file(self, model_dir, rule_name):
        # read in the RNN from file
        self.rule_name = rule_name
        self.sub = Agent(model_dir, rule_name)
        self.hidden_size = self.sub.hp['n_rnn']
        self.input_size = self.sub.hp['n_input']

    def read_rnn_agent(self, agent):
        # read in the RNN from agent
        self.sub = agent
        self.rule_name = self.sub.rule_name
        self.hidden_size = self.sub.hp['n_rnn']
        self.input_size = self.sub.hp['n_input']
