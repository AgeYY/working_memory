import context
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from core.color_manager import Degree_color
from core.agent import Agent, Agent_group
import sys

rule_name = 'color_reproduction_delay_unit'
model_dir = '../core/model/model_90.0/color_reproduction_delay_unit/'
sub_dir = '/noise_delta'

prod_int_short = 0
batch_size = 1000
sigma_rec = 0; sigma_x = 0 # set the noise to be default (training value)

#### Generate data
group = Agent_group(model_dir, rule_name, sub_dir=sub_dir)
group.do_batch_exp(prod_intervals=prod_int_short, sigma_rec=sigma_rec, batch_size=batch_size, sigma_x=sigma_x)
dire_df_short = pd.DataFrame(group.group_behaviour)

diff_color = np.array(dire_df_short['error_color'])
# reshape to 2D array
diff_color = diff_color.reshape(len(group.group), -1)
print(diff_color.shape)

def removeOutliers(a, outlierConstant=1.5):
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    return a[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))]

# Remove outliers row by row
diff_color_clean = np.array([removeOutliers(row) for row in diff_color])

error_color = []
for i in range(len(diff_color_clean)):
    error = np.sqrt(np.mean(diff_color_clean[i]**2))
    error_color.append(error)

error_color_mean = np.mean(error_color)
error_color_std = np.std(error_color)
print(error_color_mean)
print(error_color_std)
