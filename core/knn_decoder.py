import os
import numpy as np
from core.agent import Agent
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor

class KNN_decoder():
    def __init__(self, n_neighbors=2, pca_processing_n_com=2, knn_weights='distance', batch_size=2000, test_size=0.33, sigma_rec=0, sigma_x=0, prod_int=20, random_seed=42):
        '''
        knn_decoder. Use input color as label, neural states at the begining of the delay as feature map, to map feature to label.
        input:
          n_neighbors (int): model parameter, number of neighbors in knn_regressor
          pca_processing_n_com (int): model parameter, pca is used for preprocessing. Number of principal components
          weights ('uniform' or 'distance'): weights for neighbors, check knn regressor in scikit learn for more info
          batch_size (int): dataset parameter. size of whole datasets, i.e. number of trials.
          test_size:
          sigma_rec, sigma_x (float or None): dataset parameter, noise on the agent when running trials. Set None to be the same level of training
          prod_int (int): dataset parameters. delay length of a trial. Can be very short it does not matter since only the initial neural state will be used.
        '''
        self.n_neighbors = n_neighbors
        self.pca_processing_n_com = pca_processing_n_com
        self.knn_weights = knn_weights
        self.batch_size = batch_size
        self.test_size = test_size
        self.sigma_rec = sigma_rec
        self.sigma_x = sigma_x # set the noise to be default (training value)
        self.prod_int = prod_int # duration of the delay. Can be very short since only the first state of the delay will be used to train the decoder
        self.random_seed = random_seed

    def read_rnn_file(self, model_dir, rule_name):
        # read in the RNN from file
        self.rule_name = rule_name
        self.sub = Agent(model_dir, rule_name)
        self.hidden_size = self.sub.hp['n_rnn']
        self._generate_dataset()
        self._fit_model()

    def read_rnn_agent(self, agent):
        # read in the RNN from agent
        self.sub = agent
        self.rule_name = self.sub.rule_name
        self.hidden_size = self.sub.hp['n_rnn']
        self._generate_dataset()
        self._fit_model()


    def _generate_dataset(self):
        y = np.linspace(0, 360, self.batch_size) # input color
        self.sub.do_exp(prod_intervals=self.prod_int, ring_centers=y, sigma_rec=self.sigma_rec, sigma_x=self.sigma_x)
        y = self.sub.behaviour['report_color'] # use the report to avoid the effect of drift
        X= self.sub.state[self.sub.epochs['interval'][0]] # shape is [batch_size, hidden_size], use the begining of the delay
        y_rad = np.deg2rad(y)
        y_cs = np.array([np.cos(y_rad), np.sin(y_rad)]).T # map to two linear function
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_cs, test_size=self.test_size, random_state=self.random_seed)
        return X, y

    def _fit_model(self):
        self.pipe = Pipeline([
            ('pca', PCA(n_components=self.pca_processing_n_com)),
            ('knn', KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.knn_weights))
        ])
        self.pipe.fit(self.X_train, self.y_train)

    def score(self):
        # computing the r2 score based on the testing set
        return self.pipe.score(self.X_test, self.y_test)

    def decode(self, states):
        y_pred = self.pipe.predict(states)
        angle = np.arctan2(y_pred[:, 1], y_pred[:, 0])
        angle = np.rad2deg(angle + np.pi)
        return angle
