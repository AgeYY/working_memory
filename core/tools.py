"""Utility functions."""

import os
import errno
import json
#import pickle
import numpy as np
import torch
import pandas as pd
from scipy.linalg import circulant
from numpy.linalg import matrix_power
from scipy.stats.mstats import mquantiles
from sklearn.decomposition import PCA

from . import default


def load_hp(model_dir):
    """Load the hyper-parameter file of model save_name"""
    fname = os.path.join(model_dir, 'hp.json')

    if os.path.isfile(fname):
        with open(fname, 'r') as f:
            hp = json.load(f)
    else:
        print(fname)
        hp = default.get_default_hp()

    # Use a different seed aftering loading,
    # since loading is typically for analysis
    hp['seed'] = np.random.randint(0, 1000000)
    hp['rng'] = np.random.RandomState(hp['seed'])
    return hp


def save_hp(hp, model_dir):
    """Save the hyper-parameter file of model save_name"""
    hp_copy = hp.copy()
    hp_copy.pop('rng')  # rng can not be serialized
    with open(os.path.join(model_dir, 'hp.json'), 'w') as f:
        json.dump(hp_copy, f, cls=NpEncoder)


def mkdir_p(path):
    """
    Portable mkdir -p

    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_log(log, log_name='log.json'):
    """Save the log file of model."""
    model_dir = log['model_dir']
    fname = os.path.join(model_dir, log_name)
    with open(fname, 'w') as f:
        json.dump(log, f)


def load_log(model_dir, log_name='log.json'):
    """Load the log file of model save_name"""
    fname = os.path.join(model_dir, log_name)
    if not os.path.isfile(fname):
        return None

    with open(fname, 'r') as f:
        log = json.load(f)
    return log

def save_dic(dic, fname='./out.json'):
    index = find_char_index('/', fname)
    model_dir = fname[:index[-1]]
    mkdir_p(model_dir)
    with open(fname, 'w') as f:
        json.dump(dic, f, cls=NpEncoder)

def load_dic(fname='./out.json'):
    if not os.path.isfile(fname):
        return None

    with open(fname, 'r') as f:
        dic = json.load(f)
    return dic


def load_pickle(file):
    try:
        with open(file, 'rb') as f:
            pass
            #data = pickle.load(f)
    except Exception as e:
        print('Unable to load data ', file, ':', e)
        raise
    return data


def sequence_mask(lens):
    '''
    Input: lens: numpy array of integer

    Return sequence mask
    Example: if lens = [3, 5, 4]
    Then the return value will be
    tensor([[1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0]], dtype=torch.uint8)
    :param lens:
    :return:
    '''
    max_len = max(lens)
    # return torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1)
    return torch.t(torch.arange(max_len).expand(len(lens), max_len) < torch.tensor(np.expand_dims(lens, 1), dtype=torch.float32))

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.complex):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def para_reader(comb_path):
    comb = pd.read_csv(comb_path, index_col=0)

    n_combination = comb.shape[0]

    hp_replace_list = []

    for i in range(n_combination):
        hp_replace = comb.iloc[i, :].to_dict()

        # we isolate this term since its data type is untuitable for json
        try:
            hp_replace['is_cuda'] = bool(hp_replace['is_cuda'])
        except:
            hp_replace['is_cuda'] = True

        try:
            hp_replace['noise_delta_model'] = bool(hp_replace['noise_delta_model'])
        except:
            hp_replace['noise_delta_model'] = True

        try: # the number of input and output should be consistent with num_unit, if hp_replace['num_unit'] is setted.
            if hp_replace['rule_name'] == 'color_reproduction_delay_unit':
                hp_replace['n_output'] = hp_replace['num_unit']
                hp_replace['n_input'] = hp_replace['num_unit'] + 1
        except:
            pass

        try:
            hp_replace['prod_interval'] = [0, hp_replace['prod_interval_end']]
        except:
            pass
        hp_replace_list.append(hp_replace)

    return hp_replace_list

def reject_outliers(data, m = 3.5):
    '''
    This is a good idea from https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list and https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    '''
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = 0.6745 * d/mdev if mdev else 0.
    return data[s<m]

# calculate the mean and se
def quantile_curve(x, y, epsilon = 1e-5, prob=[0.25, 0.5, 0.75]):
    '''
    compute the quantiles of data
    '''
    x = np.array(x); y = np.array(y)
    b = x.copy()
    b.sort()
    d = np.append(9999999, np.diff(b))
    target = b[d>epsilon]

    upy, mdy, loy = [], [], []
    for uq in target:
        sub_list = y[np.abs(x - uq) < epsilon]
        quantiles = mquantiles(sub_list, axis=0, prob=prob)
        loy.append(quantiles[0]); mdy.append(quantiles[1]); upy.append(quantiles[2]);
    loy, mdy, upy = np.array(loy), np.array(mdy), np.array(upy)

    return target, loy, mdy, upy


def mean_se(x, y, epsilon = 1e-5, sd=False, remove_outlier=False, m=3.5):
    '''
    calculate the mean and standard error or standard deviation if sd is True of the mean
    x (array [float]): [0,0,0, 1,1,1, 2,2,2, ...] or [2.5,2.5,2.5, 2.6,2.6,2.6, ...]
    y (array [float]): the value y[i] is the y value of x[i]. Note that there are many repetitions in the x. This function will calculate the mean and se in every x value
    epsilon: the difference of x smaller than epsilon is considered as same value
    remove_outlier (bool): dataset in one bin, like y[x==1] might have outlier
    m (float): more then m * standard deviation are considered as outlier
    '''
    # unique works poorly in case x is float, we use epsilon to select the value. code comes from https://stackoverflow.com/questions/5426908/find-unique-elements-of-floating-point-array-in-numpy-with-comparison-using-a-d
    x = np.array(x); y = np.array(y)
    b = x.copy()
    b.sort()
    d = np.append(9999999, np.diff(b))
    target = b[d>epsilon]

    mean_y = []
    se_y  = []
    for uq in target:
        sub_list = y[np.abs(x - uq) < epsilon]

        if remove_outlier:
            sub_list = removeOutliers(sub_list)

        mean_y.append(sub_list.mean())
        if sd:
            se_y.append(sub_list.std())
        else:
            se_y.append(sub_list.std() / np.sqrt(sub_list.shape[0]))

    mean_y = np.array(mean_y)
    se_y = np.array(se_y)
    return target, mean_y, se_y

def smooth(x,window_len=11,window='hanning'):
    '''
    Codes adapted from Kyle Brandt. https://stackoverflow.com/questions/5515720/python-smooth-time-series-data
    '''
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:  
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]

def find_char_index(c, s):
    '''
    c (char): the target char
    s (string): the searching space
    return:
    index (list [int]): all char's positions that match c
    '''
    return [pos for pos, char in enumerate(s) if char == c]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def select_unique(x, delta):
    '''
    input:
      x (array): an array like [1, 1.2, 0.9, 2, 3]. If abs(x_i - x_j) < delta, then we think x_i and x_j represent the same number, we only pick x_i.
    output:
      x_sel (array): selected array, with only different elements
    '''
    x = np.array(x)
    b = x.copy()
    b.sort()

    unique_list = []
    sort_list = list(b)
    j = 0
    for i in range(j, len(sort_list)):
        if (sort_list[i] - sort_list[j]) > delta:
            unique_list.append(sort_list[j])
            j = i

    print(len(unique_list))
    return np.array(unique_list)
    #'''
    #input:
    #  x (array): an array like [1, 1.2, 0.9, 2, 3]. If abs(x_i - x_j) < delta, then we think x_i and x_j represent the same number, we only pick x_i.
    #output:
    #  x_sel (array): selected array, with only different elements
    #'''
    #x = np.array(x)
    #b = x.copy()
    #b.sort()
    #d = np.append(9999999, np.diff(b))
    #target = b[d>delta]
    #return target

def arg_select_unique(x, delta):
    '''
    input:
      x (array): an array like [1, 1.2, 0.9, 2, 3]. If abs(x_i - x_j) < delta, then we think x_i and x_j represent the same number, we only pick x_i.
    output:
      arg_sel (array): index of selected array, with only different elements
    '''
    x = np.array(x)
    b = x.copy()
    b_arg = np.argsort(b) # sort b
    b = b[b_arg]

    j = 0
    idx = np.zeros(x.shape[0], dtype=bool)
    idx[0] = True
# bool?
    for i in range(0, b.shape[0]):
        if (b[i] - b[j]) > delta:
            idx[j] = True
            j = i

    b_arg_inv = np.argsort(b_arg) # inverse sort index
    return idx[b_arg_inv] #idx for original x

def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
     gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi
    
    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)
    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches

def dif_matrix(x):
    '''
    differential matrix d^n / dx^n f(x) = d_x_n f_x, where d_x_n is nth order differential matrix and f_x is the function array. This function will only output d_x_n.
    input:
      x (array): x mesh. Data points must equally spaced
      del_end (bool): delete end point

    In case for len(x) = 5, n = 1:
    dif = np.array(
    [[1, 0, 0, 0, 0],
     [-1, 1, 0, 0, 0],
     [0, -1, 1, 0, 0],
     [0, 0, -1, 1, 0],
     [0, 0, 0, -1, 1]])
    ref: http://gappyfacets.com/2016/03/30/python-differentiation-matrix/
    '''
    dx = x[1] - x[0]
    # Diagonal elements are 1.
    dif_now = np.diag(np.ones(len(x)))

    # Left elements of diagonal are -1.
    dif_pre_ones = np.ones(len(x)-1) * - 1 # -1 vector.
    dif_pre = np.diag(dif_pre_ones, k=-1) # Diagonal matrix shiftedto left.

    dif = (dif_now + dif_pre) / dx
    return dif

def dif_matrix_circular(x, n=1):
    '''
    same above, but x[-1] is connected to x[0]
    '''
    dx = x[1] - x[0]
    df = np.zeros(x.shape[0])
    #df[0] = 1.0 / dx; df[1] = -1.0 / dx;
    df[1] = -1.0 / dx / 2.0; df[-1] = 1.0 / dx / 2.0;
    diff = circulant(df)
    diff = matrix_power(diff, n)
    return diff

def dif_matrix_circular_2nd(x):
    '''
    this is a common method to represent second order of derivertive
    '''
    dx = x[1] - x[0]
    df = np.zeros(x.shape[0])
    df[0] = -2.0 / dx**2; df[-1] = 1.0 / dx**2; df[1] = 1.0 / dx**2
    diff = circulant(df)
    return diff

def collate_fn(batch):
    '''batch is generated from dataset, not data loader. This function is to flat the data'''
    return batch[0]

def complex_mat2arr(mat):
    '''
    convert a complex matrix ([n, m]) into two arrs (size = n*m). One is real part another is imaginary part
    '''
    arr = mat.flatten()
    arr_real = np.real(arr)
    arr_imag = np.imag(arr)
    return arr_real, arr_imag

def state_to_angle(states, pca=None, state_type='data', verbose=False):
    '''
    compute the angle of each states in their pc1-pc2 plane
    state (array shape [n_states, n_features])
    state_type: 'data' or 'vector'. Angle is computed by projecting state to the pc1-pc2 plane, then compute the angle within the pc1-pc2 plane. However, there are two types of projection. One is 'data' type, which is more traditional. A data wil be projected to the pc1-pc2 plane. Consider a fitted pca transformer with mean vector R and pc1, pc2 vector. the data vector will be firstly be substracted by R, then substracted vector will be projected to the pc1, pc2 vectors. Intuitively it's like projecting a data point to the pc1-pc2 plane. The second type is vector projection. In this case we want to directly compute the projection of that vector to two pcs, without substracting the mean.
    verbose: if true, output fitted pca as well
    '''
    if pca is None:
        pca = PCA(n_components=2)
        pca.fit(states)

    if state_type == 'data':
        states_pca = pca.transform(states)
    elif state_type == 'vector':
        states_pca = pca.components_ @ states.T
        states_pca = states_pca.T
    else:
        print('The state type can only be data or vector')
        quit()
    angle = np.arctan2(states_pca[:, 1], states_pca[:, 0]) # calculate the angle of states_pca
    angle = np.mod(angle, 2*np.pi) / 2.0 / np.pi * 360.0

    if verbose:
        return angle, pca
    else:
        return angle

def find_indices(full_list, target_values):
    '''
    # Example usage:
    full_list = [10, 20, 30, 40, 50]
    target_values = [30, 50, 70]
    result = find_indices(full_list, target_values)
    print("Indices of target values:", result)
    '''
    indices = []
    for target in target_values:
        try:
            index = full_list.index(target)
            indices.append(index)
        except ValueError:
            indices.append(None)  # Target value not found in the full list
    return indices

def removeOutliers(a, outlierConstant=1.5):
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant # 1.5
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    return a[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))]

def align_center_multiline(text):
    lines = text.split('\n')
    width = max(len(line) for line in lines)
    return '\n'.join(line.center(width) for line in lines)
