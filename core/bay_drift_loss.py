import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises
from core.tools import find_nearest

def prior_func(x, mu, sigma, baseline=0.1):
    '''
    Prior distribution of the stimulus. It is the summation of vonmises functions and baseline, where baseline is to control the widness of the distribution. Vonmises functions have different mean value mu and variance sigma. We call the maximum points of prior distribution as common stimulus. mu and sigma should be array with the same size.
    input:
      x (array or float, [n]): stimulus
      mu, sigma (array [m]): The output x is draw from distribution function (g(mu[0], sigma[0]) + g(mu[1], sigma[1]) + ...) / normalization, where m is the size of mu.
    output:
      y (array, [n]): probability density function value
    '''

    x_temp = np.asarray(x)
    kappa = 1 / sigma**2

    n_center = len(mu)
    y = np.zeros(x_temp.shape)
    for i in range(n_center):
        y = y +  vonmises.pdf(x_temp, kappa[i], loc=mu[i])

    y = (y + baseline) / (4 + baseline * 2 * np.pi) # divide 4 is nomalization for vonmises, second term is for baseline

    return y

def prior_s_table(mu, sigma, n_mesh=300, baseline=0.1):
    '''
    compute a mesh of prior, the output 
    input:
      n_mesh (int): number of mesh points of the stimulus
    output:
      ydx: ydx is the pdf of prior * dx. So the sum(ydx) = 1 which can be used in random.choice
    '''
    x = np.linspace(-np.pi, np.pi, n_mesh, endpoint=False)

    y = prior_func(x, mu, sigma, baseline)

    ydx = y / np.sum(y)
    return x, ydx

def likelihood_func(x, mu, delay_t, sigma_n):
    '''
    pdf of f(noisy_s | s, delay_t) as different s
    input:
      x (array or float [n_x]): noisy_s
      mu (array or float [n_mu]): s.
      delay_t (array or float [n_t]): delay_t. Only one of x, mu, and delay_t can be an array
      sigma_n (float): unit sigma for noise
    output:
      y (array [n_x] or [n_mu] or [n_t]): pdf
    '''
    sigma = np.sqrt(delay_t) * sigma_n
    kappa = 1 / sigma**2
    if kappa > 600:
        kappa = 600 # to avoid overflow.
    y = vonmises.pdf(x, kappa, loc=mu)
    return y

def likelihood_table(noisy_s=0, delay_t=0, n_prior_s=100, sigma_n=0.01):
    '''
    input:
      noisy_s (float):
      delay_t (float):
      n_prior_s (int):
    output:
      prior_s (array [n_prior_s]): np.linspace(-np.pi, np.pi, n_prior_s)
      y (array [n_prior_s]): pdf of f(noisy_s | prior_s, delay_t)
    '''
    prior_s = np.linspace(-np.pi, np.pi, n_prior_s)
    y = likelihood_func(noisy_s, prior_s, delay_t, sigma_n)

    return prior_s, y

class Posterior():
    '''
    f(prior_s | noisy_s, delay_t)
    '''
    def __init__(self, c_center, sigma_s, sigma_n, baseline=0.1):
        '''
        input:
          c_center, sigma_s (array [m]): The prior distribution function (g(mu[0], sigma[0]) + g(mu[1], sigma[1]) + ...) / normalization, where m is the size of mu.
          baseline (float):
          sigma_n (float): unit sigma for noise
        '''
        self.c_center = c_center
        self.sigma_s = sigma_s
        self.baseline = baseline

        self.sigma_n = sigma_n

    def norm_table(self, n_noisy_s=200, n_prior_s=200, dt=20, delay_t_max=2000, t0=1):
        '''
        f(prior_s | noisy_s, delay_t) = f(noisy_s | prior_s, delay_t) * f(prior_s) * N, where N is normalization = int{ds} f(noisy_s | s, delay_t) * f(s). So normalization is a function of delay_t and noisy_s. We construct a normalization table accordingly
        baseline (float): baseline for prior distribution
        t0: to avoid time equal to 0
        n_noisy_s, n_prior_s (array): number of mesh points in computation. At time near 0, the width of likehood is narrow. If one use too small n_noisy_s, the value may not be captured. If the n_noisy_s = 100. ntable[0, 0] would apear be a bit higher than other peaks in the prior distribution, because noisy_s = -np.pi is sampled, but not noisy_s = 0. This makes considerable difference if the time is small
        '''
        delay_t_max = delay_t_max + 1

        noisy_s = np.linspace(-np.pi, np.pi, n_noisy_s)
        delay_t = np.arange(0, delay_t_max, dt)
        delay_t[0] = t0

        y = np.zeros((n_noisy_s, len(delay_t)))

        mesh_s, pdf_s_ds = prior_s_table(c_center, sigma_s, n_mesh=n_prior_s, baseline=self.baseline) # generate pdf for prior distribution
        for i, n_s in enumerate(noisy_s):
            for j, d_t in enumerate(delay_t):
                mesh_s, pdf_n = likelihood_table(n_s, d_t, n_prior_s=n_prior_s)
                y[i, j] = np.dot(pdf_n, pdf_s_ds)

        self.noisy_s_mesh = noisy_s
        self.delay_t_mesh = delay_t
        self.norm_table = y

        return noisy_s, delay_t, y

    def forward(self, s, noisy_s, delay_t):
        '''
        posterior
        s (array)
        noisy_s (float)
        delay_t (float)
        '''
        # look the table and find the normalization factor
        i = find_nearest(noisy_s, self.noisy_s_mesh)
        j = find_nearest(delay_t, self.delay_t_mesh)
        n_factor = self.norm_table[i, j]
        post = prior_func(s, self.c_center, self.sigma_s, baseline=self.baseline) * likelihood_func(noisy_s, s, delay_t, self.sigma_n) / n_factor

        return post
