# calculate the L2 normalizty of the prior distrbution
import scipy
from scipy.optimize import root_scalar
from core.bay_drift_loss_torch import prior_func
import numpy as np

class Normalitier():
    def __init__(self, mu, dim=20000):
        '''
        convert prior distribution defined by prior_func in bay_drift_loss_torch.py into entropy
        '''
        self.mu = mu
        self.dim = dim
        self.x = np.linspace(-np.pi, np.pi, dim) # binsize = 5 degree

    def sigma2normality(self, sigma):
        '''
        convert prior distribution defined by prior_func in bay_drift_loss_torch.py into entropy
        mu, sigma (array [m]): The output x is draw from distribution function (g(mu[0], sigma[0]) + g(mu[1], sigma[1]) + ...) / normalization, where m is the size of mu.
        output: normality value
        '''
        p = prior_func(self.x, self.mu, sigma)* (self.x[1] - self.x[0])
        return np.linalg.norm(p)

    def normality2sigma(self, normality, x0=0.5, x1=0.15):
        '''
        convert L2 normality to sigma. sigma should not exceed 2 * np.pi, or smaller than 3 / 360 * 2 * np.pi. L2 normality is bounded by 1 / sqrt[d] and 1 where d is the dimensionality of the prior distribution vector.
        normality (float)
        '''
        if (normality > 1) or (normality < 1 / np.sqrt(self.dim)):
            print('Normality Error: normality should be bounded by [1 / sqrt[d], 1]')

        def wrapper(sigma):
            return self.sigma2normality(sigma) - normality
        sol = root_scalar(wrapper, x0=x0, x1=x1)
        return sol.root

class Entropier():
    def __init__(self, mu, dim=360):
        '''
        convert prior distribution defined by prior_func in bay_drift_loss_torch.py into entropy
        '''
        self.mu = mu
        self.dim = dim
        self.x = np.linspace(-np.pi, np.pi, dim) # binsize = 5 degree

    def sigma2entropy(self, sigma):
        '''
        convert prior distribution defined by prior_func in bay_drift_loss_torch.py into entropy
        sigma (float): kappa = 1 / sigma^2
        output: entropy value
        '''
        if (sigma > 10) or (sigma < 0.01):
            print('Entropy Error: sigma should be bounded by [0.01, 10] rad, although [1, 180] degree is a more suitable')
        p = prior_func(self.x, self.mu, sigma)
        return scipy.stats.entropy(p) # entropy will automatically normalize the probability

    def sigma_arr2entropy(self, sigma):
        '''
        same as self.sigma2entropy, but the input and output are arrays
        '''
        sigma_np = np.array(sigma)
        entropy_np = np.empty(sigma_np.shape)
        for idx, sig in enumerate(sigma_np):
            entropy_np[idx] = self.sigma2entropy(sig)
        return entropy_np

    def entropy2sigma(self, entropy, x0=0.5, x1=0.02):
        '''
        convert entropy to sigma_s. sigma_s should not exceed 2 * np.pi, or smaller than 1 / 360 * 2 * np.pi
        entropy (float)
        '''
        def wrapper(sigma_s):
            return self.sigma2entropy(sigma_s) - entropy
        sol = root_scalar(wrapper, bracket=[0.01, 10], x0=x0, x1=x1)
        return sol.root
