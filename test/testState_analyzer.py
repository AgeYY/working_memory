import context
from core.manifold.state_analyzer import State_analyzer
from core.agent import Agent
import numpy as np
from core.rnn_decoder import RNN_decoder

model_dir = '../core/model_local/color_reproduction_delay_unit_vonmise_cp7_np2/model_30/noise_delta_p2'
rule_name = 'color_reproduction_delay_unit'
prod_interval = 400

sub = Agent(model_dir, rule_name)
sa = State_analyzer()
sa.read_rnn_agent(sub)
rd = RNN_decoder()
rd.read_rnn_agent(sub)

color = np.array([10, 30])
#color = np.linspace(0, 360, 100)
init_states = sa.color_state(color)
de_colors = rd.decode(init_states)

dr = Delay_runner()
dr.read_rnn_agent(sub)
delay_states = dr.delay_run(init_states, prod_interval)
print(delay_states.shape)
de_colors = rd.decode(delay_states)
print(de_colors)
