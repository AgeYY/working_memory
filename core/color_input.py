# generate the input for color task.
import numpy as np
from scipy.stats import vonmises

def vonmises_prior(degree_input, bias_centers, width=10):
    '''
    Prior distribution of the stimulus. It is the summation of vonmises functions. Vonmises functions have different mean value mu and variance sigma. We call the maximum points of prior distribution as common stimulus. mu and sigma should be array with the same size.
    input:
      x (array or float, [n]): stimulus, from -np.pi to np.pi
      mu, sigma (array [m]): The output x is draw from distribution function (g(mu[0], sigma[0]) + g(mu[1], sigma[1]) + ...) / normalization, where m is the size of mu.
      sigma (float): for some numerical issue, sigma should not be smaller than 8 degree
      center_prob (float): probability within 2 sigama of vonmises + the area of baseline
    output:
      y (array, [n]): probability density function value. Unnormalized
    '''
    # convert degree to rad
    rad_input = degree_input / 360 * 2 * np.pi - np.pi
    bias_centers_rad = np.array(bias_centers) / 360 * 2 * np.pi - np.pi
    width_rad = width / 360 * 2 * np.pi

    n_center = len(bias_centers_rad)
    kappa = 1 / width_rad**2
    y = np.zeros(rad_input.shape)
    for i in range(n_center):
        y = y +  vonmises.pdf(rad_input, kappa, loc=bias_centers_rad[i])

    y = y / n_center # divide 4 is nomalization for vonmises, second term is for baseline
    return y

def gaussian_func(x, mu=0, sig=1):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
def delta_func(x, centers, width=10, center_prob=0.5):
    '''
    generate delta probability distribution of the color
    n * p(bias - sig <= degree < bias + sig) = center_prob, where bias arbitary single bias, n is the number of centers.
    Input:
    degree_input (np.array [float from 0 to 360]): sampled degree. For example, [0, 50, 100] means the color can only be 0, 50 or 100 (degree). Then the goal of this function is to assign the probability distribution of this three colors.
    '''
    n_center = len(centers)
    height = center_prob / (1 - center_prob) * (360 - n_center * width * 2) / (n_center * width * 2) # this makes sure there are center_prob chance of hitting the center colors
    score = np.ones(len(x))
    for center in centers:
        score[(x < center + width) * (x >= center - width)] = height
    return score

def input_color_dist(degree_input, bias_centers, method, sig=10, center_prob=0.5):
    '''
    generate the input distribution. The color is indicated by angle.
    Input:
    degree_input (np.array [float from 0 to 360]): sampled degree. For example, [0, 50, 100] means the color can only be 0, 50 or 100 (degree). Then the goal of this function is to assign the probability distribution of this three colors.
    bias_center (np.array [float]): centers for gaussian or delta method
    sig (float): width of centers. In the gaussian case this would be the standard deviation, in the delta case this would be the width of the rectanct
    center_prob (float): in the delta case. There was a 50% chance (defaule) that the color of that sample would be drawn from a biased dis- tribution. This parameter is only for delta
    method (str: [uniform, gaussian, delta]).
        'uniform': p(degree) = 1 / n_colors
        'gaussian': p(degree) is propotional to sum_{bias_centers}{exp^{(x - bias_center)^2 / 2 / sig^2}}
        'delta': n * p(bias - sig <= degree < bias + sig) = center_prob, where bias arbitary single bias, n is the number of centers.
        'vonmises': p(degree) is propotional to the sum of vonmises functions
    '''
    center_p_input = np.zeros(degree_input.shape) # probability distribution of input color
    if method == 'uniform':
        center_p_input = center_p_input + np.ones(center_p_input.shape) # the probability distribution of input centers at bc
    elif method == 'gaussian':
        for bc in bias_centers:
            center_p_input = center_p_input + gaussian_func(degree_input, bc, sig = sig) # the probability distribution of input centers at bc. sig is fixed here.
    elif method == 'delta':
        center_p_input = delta_func(degree_input, bias_centers, width=sig, center_prob=center_prob)
    elif method == 'vonmises':
        center_p_input = vonmises_prior(degree_input, bias_centers, width=sig)
    else:
        raise ValueError('Unknown method')

    center_p_input = center_p_input / np.sum(center_p_input) # normalization
    return center_p_input

class Color_input():
    '''
    Create input colors' degrees
    '''
    def add_samples(self, n_degree=180):
        self.degree_input = np.arange(0., 360., 360. / n_degree) # possible input color degrees
        return self.degree_input

    def prob(self, bias_centers=None, method='uniform', sig=10, center_prob=0.5):
        '''add probability distribution to each color in degree_input. The detail of this function see input_color_dist'''
        self.center_p_input = input_color_dist(self.degree_input, bias_centers, method=method, sig=sig, center_prob=center_prob)
        return self.center_p_input

    def out_color_degree(self, batch_size, random_state=np.random.RandomState(1000)):
        # output n colors
        sampled_degree = random_state.choice(self.degree_input, batch_size, p=self.center_p_input)
        return sampled_degree
