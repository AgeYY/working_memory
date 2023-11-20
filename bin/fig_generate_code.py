import os

# Figure 4 A
'''
delay_len = 800
sub_dir="/model_40/noise_delta" # pick the first model as an example model
for noise in ['0.10','0.30']:
    model_dir = "../core/model_noise/noise_" + noise + "/model_17.5/color_reproduction_delay_unit/"

    # decode velocity, see output figure as ../bin/figs/fig_collect/decode_vel_plane_noisexxx.pdf
    os.system('python ../bin/figs/decode_vel_state.py' + ' --model_dir ' + model_dir + ' --sub_dir ' + sub_dir + ' --file_label ' + 'noise' + noise)

'''

# Figure 4 B Angle occupation
'''
sub_dir = 'model_40/noise_delta'
rule_name = "color_reproduction_delay_unit"
gen_data='Y'
for noise in ['0.10','0.30']:
    model_dir = "../core/model_noise/noise_" + noise + "/model_17.5/color_reproduction_delay_unit/"
    # find figures in ../bin/figs/fig_collect/angle_noisexxx_density.svg and angle_noisexxx_function.svg
    os.system('python ../bin/figs/encode_space.py ' + model_dir + ' ' + rule_name + ' ' + sub_dir + ' ' + gen_data + ' ' + \
          "../bin/figs/fig_collect/angle_noise" + noise)
'''

# Figure 4 C delay trajectories
'''
delay_len = 800
sub_dir="/model_40/noise_delta" # pick the first model as an example model

for noise in ['0.10','0.30']:
    model_dir = "../core/model_noise/noise_"+str(noise)+"/model_17.5/color_reproduction_delay_unit/"
    # fixpoint ring, fig. 4c. see output figure as traj_fix_xxx.pdf
    os.system('python ../bin/figs/fix_point_ring.py' + ' --model_dir ' + model_dir + ' --sub_dir ' + sub_dir + ' --prod_interval ' + \
               str(delay_len) + ' --file_label ' + 'noise'+str(noise))
'''

# Figure 4 D distribution of attractors
'''
sub_dir = 'model_40/noise_delta'
for noise in ['0.10','0.30']:
    model_dir = "../core/model_noise/noise_"+str(noise)+"/model_17.5/color_reproduction_delay_unit/"
    os.system('python ./figs/fix_point_batch_cluster.py' + ' --model_dir ' + model_dir + ' --file_label ' + 'noise'+str(noise))

    # if you want to generate data, turn on the --gen_data flag
    # os.system('python ./figs/fix_point_batch_cluster.py' + ' --model_dir ' + model_dir + ' --file_label ' + 'noise'+str(noise) + ' --gen_data ')
'''

# Figure 4E
pass

# Figure 4F

#
os.system('python ./figs/eg_go_response_plane_figure.py')
#
os.system('python ./figs/go_response_drift.py')
#
os.system('python ./figs/go_response_drift_sigmas.py')
#
os.system('python ./figs/experiment_theory_noise01to0.3.py')
#
os.system('python ./figs/readout_decode_response_plane.py')
#
os.system('python ./figs/readout_angular_occupancy_eg.py')
#
os.system('python ./figs/readout_angular_occupancy_entropy_sigmas.py')
