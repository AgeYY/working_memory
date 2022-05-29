import context
from core.manifold.state_analyzer import State_analyzer
from core.agent import Agent
import numpy as np
from core.rnn_decoder import RNN_decoder
from core.delay_runner import Delay_runner

class Space_analyzer(State_analyzer):

    def read_rnn(self, input_rnn, rule_name=None, read_from='agent'):
        # read in the RNN from agent
        if read_from == 'agent':
            self.sub = agent
        else:
            self.rule_name = rule_name
            self.sub = Agent(model_dir, rule_name)

        self.rule_name = self.sub.rule_name
        self.hidden_size = self.sub.hp['n_rnn']
        self.input_size = self.sub.hp['n_input']

        self.sub.do_exp(prod_intervals=self.prod_intervals, ring_centers=self.pca_degree, sigma_rec=self.sigma_rec, sigma_x=self.sigma_x)
        self.pca = PCA(n_components=2)
        self.pca.fit(self.sub.state[self.sub.epochs['interval'][1]])

    def neural_vel():

model_dir = '../core/model/color_reproduction_delay_unit_25/model_7/noise_delta'
rule_name = 'color_reproduction_delay_unit'
prod_interval = 800

sub = Agent(model_dir, rule_name)
sa = Space_analyzer()
sa.read_rnn_agent(sub)
rd = RNN_decoder()
rd.read_rnn_agent(sub)

color = np.array([10, 30])
#color = np.linspace(0, 360, 100)
init_states = sa.color_state(color)
de_colors = rd.decode(init_states)
print(color, de_colors)
