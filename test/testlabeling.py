import context
import matplotlib.pyplot as plt
import numpy as np
from core.agent import Agent, Agent_group
from core.net_struct.labeling import Labeler, array_pipline
from core.net_struct.main import Bump_activity, bump_pipline, bin_fir, sc_dist


model_dir = '../core/model_local/color_reproduction_delay_unit/'
#sub_dir = 'model_16/noise_delta_stronger/'
sub_dir = 'model_16/noise_new/'
rule_name = 'color_reproduction_delay_unit'
prod_intervals=20 # set the delay time to 800 ms for ploring the trajectory
n_colors=180
tuned_thre = 0.5

agent_path = model_dir + sub_dir
sub = Agent(agent_path, rule_name)
laber = Labeler()
laber.read_rnn_agent(sub)

#state_list = laber.do_exp(sigma=0.05, batch_size=3, prod_intervals=800, n_colors=720) # generate data
#report_color = laber.decode_data(state_list) # decode data to color
#tuning, sense_color = laber.compute_tuning(state_list, report_color, bin_width=3) # neural response to different colors, where colors are bined
#label, t_strength = laber.prefer_color(tuning, sense_color) # labeling neurons with prefer colors

label, t_strength = laber.label_rnn_decoder(sigma=0.05, batch_size=3, prod_intervals=800, n_colors=720, bin_width=2)
tuning_sort, label_sort = array_pipline(laber.tuning, laber.label, thresh=-999, bin_width=None)

fig = plt.figure(figsize = (3, 3))
ax = fig.add_subplot(111)
ax.imshow(tuning_sort)
plt.show()

#label, t_str = laber.label_delay(prod_intervals=prod_intervals, n_colors=180)
#
#tuning_pped, label = bump_pipline(laber.bump, laber.bump.tuning.copy(), thre=tuned_thre, bin_width=None)
#
#plt.imshow(tuning_pped)
#plt.show()
##plt.hist(label)
##plt.show()
#plt.hist(t_str)
#plt.show()
