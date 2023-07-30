import context
import os
import numpy as np
import matplotlib.pyplot as plt
from core.agent import Agent, Agent_group
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

def angle_to_cos_sin(angle):
    y_rad = np.deg2rad(angle)
    y_cs = np.array([np.cos(y_rad), np.sin(y_rad)]).T # map to two linear function
    return y_cs

prod_int = 20 # duration of the delay. Can be very short since only the first state of the delay will be used to train the decoder
sigma_rec = None; sigma_x = None # set the noise to be default (training value)
batch_size = 2000
prior_sig = 20.0
test_size = 0.33
n_neighbors = 10
random_seed = 42
pca_processing_n_com = 5

rule_name = 'color_reproduction_delay_unit'
model_dir_parent = "../core/model/model_" + str(prior_sig) + "/color_reproduction_delay_unit/"
model_dir = 'model_2/' # example RNN
sub_dir = 'noise_delta/'

f = os.path.join(model_dir_parent, model_dir, sub_dir)
sub = Agent(f, rule_name) # this is the outside agent creating data

# generate datasets
y = np.linspace(0, 360, batch_size) # input color
sub.do_exp(prod_intervals=prod_int, ring_centers=y, sigma_rec=sigma_rec, sigma_x=sigma_x)
y = self.sub.behaviour['report_color'] # use the report to avoid the effect of drift
X= sub.state[sub.epochs['interval'][0]] # shape is [batch_size, hidden_size], use the begining of the delay
y_cs = angle_to_cos_sin(y) # map to two linear function

# split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y_cs, test_size=test_size, random_state=random_seed)

# model
pipe = Pipeline([
    ('pca', PCA(n_components=pca_processing_n_com)),
    ('knn', KNeighborsRegressor(n_neighbors=n_neighbors))
])

pipe.fit(X_train, y_train)

y_pred_knn = pipe.predict(X_test)
score = r2_score(y_test, y_pred_knn)
print(score)

# compare to the RNN decoder
from core.rnn_decoder import RNN_decoder
rd = RNN_decoder()
rd.read_rnn_agent(sub)
y_pred_rnn = rd.decode(X_test)
y_pred_rnn = angle_to_cos_sin(y_pred_rnn)
score = r2_score(y_test, y_pred_rnn)
print(score)

# how well the knn align with RNN
align_score = r2_score(y_pred_knn, y_pred_rnn)
print(align_score)
