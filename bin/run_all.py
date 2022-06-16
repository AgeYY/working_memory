import os

# Training models, models with sigma_s = 3, 25, and 90 degree are essential to reproduce the main result of the paper.
# Alternatively, download models in https://wustl.app.box.com/file/964118053859?s=3xnt37fddxelvio2fztlawyieatf2agq
# The trained/downloaded models will/should be in /core/model/

imp_sig = [3.0, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 90.0]
n_thread = 20; n_model = 50;
for sig in imp_sig:
    os.system('mpiexec -n ' + str(n_thread) + ' python train_cluster.py --sig_s ' + str(round(sig, 1)) + \
    ' --n_model ' + str(n_model))

## Reproducing figure 1, takes 2 minites
rule_name = "color_reproduction_delay_unit" # rule name (RNN architeture and task type) through out this paper
model_dir = "../core/model/model_25.0/color_reproduction_delay_unit/" # source model
gen_data = 'Y' # generate figure data
sub_dir = "/noise_delta"

os.system('python ./figs/gaussian_error_group.py ' + model_dir + ' ' + rule_name + ' ' + sub_dir + ' ' + gen_data) # gaussian noise error
# see output figure in ./figs/fig_collect/gaussian_rnn.pdf
os.system('python ./figs/report_dist.py ' + model_dir + ' ' + rule_name + ' ' + sub_dir + ' ' + gen_data) # report distribution. \
# see output figure in ./figs/fig_collect/report_rnn.pdf
os.system('python ./figs/bias_around_common_group.py ' + model_dir + ' ' + rule_name + ' ' + sub_dir + ' ' + gen_data) # biased report. \
# see output figure in ./figs/fig_collect/unbias_rnn.pdf and ./figs/fig_collect/bias_rnn.pdf

# Reproducing figure 2 a, b
sub_dir="/model_0/noise_delta" # pick the first model as an example model
delay_len = 1000

for model_name in ['90.0', '25.0', '3.0']: # uniform, biased, and strongly biased
    model_dir = "../core/model/model_" + model_name + "/color_reproduction_delay_unit/"
    # bump activity. see output figure in ./figs/fig_collect/bump_model_name.pdf
    os.system('python ./figs/bump.py' + ' --model_dir ' + model_dir + ' --sub_dir ' + sub_dir + ' --prod_interval ' + str(delay_len) +\
             ' --file_label ' + model_name) 
    # 5 example neural activity. find output figure in ./figs/fig_collect/neural_activity_model_name.pdf
    os.system('python ./figs/simple_fig.py' + ' --model_dir ' + model_dir + ' --sub_dir ' + sub_dir + ' --prod_interval ' + str(delay_len) +\
             ' --file_label ' + model_name)

# Neural speed, figure 2 cde
sub_dir="/noise_delta"

delay = 6000
file_label = str(delay) + '_' + 'key_all'
key_part = ['90.0', '25.0', '15.0', '3.0']

os.system('python ./figs/neural_speed_more_prior.py ' + ' --prod_interval ' + str(delay) + ' --keys_part ' + ' '.join(key_part) \
          + ' --file_label ' + file_label)

# Reproducing figure 3 a, b
rule_name = "color_reproduction_delay_unit"
model_dir = "../core/model/model_25.0/color_reproduction_delay_unit/" # source model
gen_data = 'Y' # generate figure data
sub_dir = "/noise_delta"

# pca, see figure in ./figs/fig_collect/pca_explained.pdf
os.system('python ./figs/pca_explained.py ' + model_dir + ' ' + rule_name + ' ' + sub_dir + ' ' + gen_data)
sub_dir = "/model_0/noise_delta"
os.system('python ./figs/manifold.py ' + model_dir + ' ' + rule_name + ' ' + sub_dir + ' ' + gen_data)

# Reproducing figure 3 c, d, e
delay_len = 800
sub_dir="/model_0/noise_delta" # pick the first model as an example model

for model_name in ['90.0', '25.0', '3.0']: # uniform, biased, and strongly biased
    model_dir = "../core/model/model_" + model_name + "/color_reproduction_delay_unit/"
    # fixpoint ring, fig. 4c. see output figure as traj_fix_xxx.pdf
    os.system('python ./figs/fix_point_ring.py' + ' --model_dir ' + model_dir + ' --sub_dir ' + sub_dir + ' --prod_interval ' + \
               str(delay_len) + ' --file_label ' + model_name)
    # decode velocity, fig. 4c. see output figure as decode_vel_plane.pdf
    os.system('python ./figs/decode_vel_state.py' + ' --model_dir ' + model_dir + ' --sub_dir ' + sub_dir + ' --file_label ' + model_name)
    # find figures in ./figs/fig_collect/combine_xxx.pdf
    os.system('python ./figs/combine_state.py' + ' --model_dir ' + model_dir + ' --sub_dir ' + sub_dir + ' --file_label ' + model_name)

# Reproducing figure 4
sub_dir="/model_0/noise_delta" # pick the first model as an example model
model_name = '25.0'
model_dir = "../core/model/model_" + model_name + "/color_reproduction_delay_unit/"
rule_name = "color_reproduction_delay_unit"
delay_len = 6000
gen_data='Y'

# bump activity for very long delay, see output in bumpxxx_long_delay.pdf
os.system('python ./figs/bump.py' + ' --model_dir ' + model_dir + ' --sub_dir ' + sub_dir + ' --prod_interval ' + str(delay_len) +\
             ' --file_label ' + model_name + '_long_delay')
# #The trajectory of the same trial, see output in ./figs/fig_collect/traj_fix_one_xxx.pdf
os.system('python ./figs/fix_point_one_trial.py ' + model_dir + ' ' + rule_name + ' ' + sub_dir + ' ' + gen_data + \
          ' ./figs/fig_data/fixpoints_one.json ./figs/fig_collect/traj_fix_one')

# fig 5b, see output ./figs/fig_collect/manifold_2d_3.0_delay_xxx.pdf'
model_name = '3.0'
model_dir = "../core/model/model_" + model_name + "/color_reproduction_delay_unit/"

for delay_len in [60, 1000]:
    os.system('python ./figs/manifold_2d.py' + ' --model_dir ' + model_dir + ' --sub_dir ' + sub_dir + ' --prod_interval ' + \
              str(delay_len) + ' --file_label ' + model_name + '_delay_' + str(delay_len))

# reproducing figure 5, searching fix points for all RNNs (fig 6c) can take one hour or more
sub_dir = 'model_0/noise_delta'
for model_name in ['90.0', '25.0', '3.0']:
    model_dir = "../core/model/model_" + model_name + "/color_reproduction_delay_unit/"
    ## figure 6 a, b, find figures in angle_xxx_density.pdf and angle_xxx_function.pdf
    os.system('python ./figs/encode_space.py ' + model_dir + ' ' + rule_name + ' ' + sub_dir + ' ' + gen_data + ' ' + \
          "./figs/fig_collect/angle_" + model_name)
    # figure 6c, distribution of attractors, find output @ att_dis_xxx.pdf
    os.system('python ./figs/fix_point_batch.py' + ' --model_dir ' + model_dir + ' --file_label ' + model_name)
    
    ## or using mpi4py to search fixpoints
    #os.system('mpiexec -n ' + str(n_thread) + 'python ./figs/fix_point_batch_cluster.py' + ' --model_dir ' + model_dir + ' --file_label ' + model_name)

# Figure 6a, find figure as rnn_ddm_sim_25.0.pdf
model_name = '25.0'
model_dir = "../core/model/model_" + model_name + "/color_reproduction_delay_unit/"
os.system('python ./figs/rnn_ddm.py' + ' --model_dir ' + model_dir + ' --file_label ' + model_name)

#figure 6c, find figure as rnn_bay_drift_xxx.pdf
for model_name in ['90.0', '25.0', '12.5']:
    model_dir = "../core/model/model_" + model_name + "/color_reproduction_delay_unit/"
    os.system('mpiexec -n 2 python ./figs/rnn_noise_bay_drift.py' + ' --model_dir ' + model_dir + ' --file_label ' + model_name + \
              ' --sigma_s ' + model_name)
    
# figure 6d, can take a few hours
keys = ['90.0', '30.0', '27.5', '25.0', '22.5', '20.0', '17.5', '15.0', '12.5']
for model_name in keys: # generate data
    model_dir = "../core/model/model_" + model_name + "/color_reproduction_delay_unit/"
    os.system('mpiexec -n 2 python ./figs/rnn_noise_bay_drift.py' + ' --model_dir ' + model_dir + ' --file_label ' + model_name + \
              ' --sigma_s ' + model_name)
#calculate mse, find figures in rnn_bay_drift_.pdf and ./figs/fig_collect/bay_drift_xxx.pdf
os.system('python ./figs/mse_drift.py' + ' --keys ' + ' '.join(keys))
