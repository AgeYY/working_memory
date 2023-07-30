import os
import context
from core.knn_decoder import KNN_decoder
from core.agent import Agent

prior_sig = 20.0
rule_name = 'color_reproduction_delay_unit'
model_dir_parent = "../core/model/model_" + str(prior_sig) + "/color_reproduction_delay_unit/"
model_dir = 'model_2/' # example RNN
sub_dir = 'noise_delta/'

f = os.path.join(model_dir_parent, model_dir, sub_dir)

kd = KNN_decoder()
kd.read_rnn_file(f, rule_name)

score = kd.score()
print(score)

sub = Agent(f, rule_name) # this is the outside agent creating data
