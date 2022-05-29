# tunning matrix, bump activity. Run this code in bin/
import context
from core.color_manager import Degree_color
import matplotlib.pyplot as plt
import numpy as np
from core.agent import Agent
from core.net_struct.main import Bump_activity, bump_pipline, bin_fir
import sys
from core.net_struct.main import circular_mean

##### input arguments

try:
    model_dir = sys.argv[1]
    rule_name = sys.argv[2]
    sub_dir = sys.argv[3]
except:
    rule_name = 'color_reproduction_delay_unit'
    model_dir = '../core/model_local/color_reproduction_delay_unit/'
    sub_dir = 'model_16/noise_delta/'

try:
    if sys.argv[4] == 'Y': # set false so it will not generate data
        gen_data = True
    else:
        gen_data = False
except:
    gen_data = False

####################

epoch = 'interval'
binwidth = 5 # binwidth for outdire and target dire
batch_size = 1
prod_intervals=2000 # set the delay time to 800 ms for ploring the trajectory
pca_degree = np.arange(0, 360, 5) # Plot the trajectories of these colors
sigma_rec = None # noise in single trial, not for calculate tunning
sigma_x = None
single_color = 180 # the color for single trial
tuned_thre = -999
bin_width = 6
diff_start_end = 40
max_single_trial = 600

# repeat trials
sub = Agent(model_dir + sub_dir, rule_name)
fir_rate_list = []
for i in range(batch_size):
    fir_rate, _, _ = sub.do_exp(prod_intervals=prod_intervals, ring_centers=pca_degree, sigma_rec=0.0, sigma_x=0.0)
    fir_rate_list.append(fir_rate)
# mean firing rate
fir_rate_list = np.concatenate(fir_rate_list).reshape(-1, *fir_rate.shape)
fir_rate_mean = np.mean(fir_rate_list, axis=0)

# get the tunning matrix
bump = Bump_activity()
bump.fit(sub.behaviour['target_color'], fir_rate_mean, sub.epochs['interval'])

#tunning_pped, label = bump_pipline(bump, bump.tunning.copy(), thre=tuned_thre, bin_width=bin_width)
#plt.imshow(tunning_pped)
#plt.savefig('./tunning.png')
#plt.show()

#### latent tuning
#from sklearn.decomposition import PCA
#n_components = 3
#pca = PCA(n_components=n_components)
#latent_tuning = pca.fit_transform(tunning_pped)
#print(pca.explained_variance_ratio_)
#for col in np.transpose(latent_tuning):
#    plt.plot(pca_degree, col)
#plt.show()


def do_one_exp(single_color):
    firing_rate, _, _= sub.do_exp(prod_intervals=prod_intervals, ring_centers=np.array(single_color), sigma_rec=sigma_rec, sigma_x=sigma_x)
    firing_rate = firing_rate[sub.epochs['interval'][0]: sub.epochs['interval'][1]]
    firing_rate = firing_rate.reshape((firing_rate.shape[0], firing_rate.shape[2])) # only one color here

    firing_rate_pped, label = bump_pipline(bump, firing_rate, thre=tuned_thre, bin_width=bin_width)
    return firing_rate_pped, label, sub.behaviour['report_color'][0], single_color

#### Try new expriment to search bump. We use population vector method to decode the color from the states in delay period. If the difference of decoded color in the begining and end are large, the bump is obvious, we interupt searching.
for i in range(max_single_trial):
    firing_rate_pped, label, report_color, target_color = do_one_exp(single_color)

    delay_start_fire = np.mean(firing_rate_pped[0:5, :], axis=0) + 1
    delay_end_fire = np.mean(firing_rate_pped[-4:, :], axis=0) + 1 # + 1 to shift the baseline of firing rate

    delay_start_color, norm = circular_mean(delay_start_fire, label)
    delay_end_color, norm = circular_mean(delay_end_fire, label)

    if abs(delay_start_color - delay_end_color) > diff_start_end:
        break

pd.DataFrame(firing_rate_pped).to_csv('./figs/fig_data/bum.csv')

print('delay_start_color: ', delay_start_color)
print('delay_end_color: ', delay_end_color)

plt.plot(np.mean(firing_rate_pped[0:5, :], axis=0))
plt.plot(np.mean(firing_rate_pped[-4:, :], axis=0))
plt.savefig('./compare_end.png')
plt.show()

from skimage.transform import resize
bottle_resized = resize(firing_rate_pped, (60, 60))
plt.imshow(bottle_resized + 1, cmap='binary')
plt.savefig('./bump.png')
plt.show()
