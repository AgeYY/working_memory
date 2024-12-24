# convert degree color to the input of RNN and also the output of RNN to the degree color. Basically this file contains color encoder and decoder.
# Currently we have:
# 1. Encode/Decode degree to LMS space
# 2. Encode/Decode degree to the firing rate of color cell (virtual) (under construction).
import numpy as np
import colormath.color_objects as colorob
import colormath.color_conversions as colorcv
from scipy.stats import vonmises
from matplotlib import cm

# normalized to D65
MAT_LMS_NORM = np.array([[0.4002, -0.2263, 0], [0.7076, 1.1653, 0], [-0.0808, 0, 0.9182]])
MAT_LMS_NORM_INV = np.array(
[[ 1.86006661,  0.36122292,  0.        ],
 [-1.12948008,  0.63880431, -0.        ],
 [ 0.16368262,  0.03178699,  1.08908734]]
)

class Degree_color():
    '''
    output several format of color which is defined by a circle in cielab space.
    '''
    def __init__(self, center_l=80, center_a=22, center_b=14, radius=60):
        '''
        center_l, center_a, center_b (int): the center of color circle
        radius (float): the raius of color circle
        The default value comes from experiment 1a from paper error-correcting ...
        '''
        self.center_l, self.center_a, self.center_b = center_l, center_a, center_b
        self.radius = radius
    def out_color(self, degree, fmat='LAB'):
        '''
        input:
          degree (np.array or int): the unit is degree. if the input is int, the output color will have degrees np.arange(0, 360, degree)
          fmat (str): output color format, can be LAB, RGBA (where A is fixed as 1), LMS
        output:
          color (n * c matrix): where n is the length of degree, c is the number of channel in of the color format.
        '''
        # calculate the color on LAB space
        i = 0
        if not hasattr(degree, '__iter__'):
            degree = np.linspace(0, 360, degree)

        lab = self.deg2lab(degree)
        if fmat == 'LAB':
            return lab
        elif fmat == 'RGBA':
            return lab2rgba(lab)
        elif fmat == 'XYZ':
            return lab2xyz(lab)
        elif fmat == 'LMS':
            xyz = lab2xyz(lab)
            return xyz2lms(xyz)
        elif fmat == 'RGBA_my':
            cmap = cm.get_cmap('hsv')
            degree_norm = degree / 360.0
            color_list = cmap(degree_norm)
            #alpha = 0.3
            #color_list[:, -1] = alpha * np.ones(color_list.shape[0])
            #light = 0.6
            #color_list[:, 0:3] *= light
            return color_list

    def set_centers(self, center_l, center_a, center_b):
        self.center_l, self.center_a, self.center_b = center_l, center_a, center_b
    def set_radius(self, radius):
        self.radius = radius

    def deg2lab(self, degree):
        rads = np.deg2rad(degree) # convert to rad so that can be calculated by np.cos

        n_sample = len(rads)
        lab = np.zeros((n_sample, 3)) # 3 channels l, a, b
        i = 0
        for rad in rads:
            lab[i, :] = np.array([self.center_l, self.center_a + self.radius * np.cos(rad), self.center_b + self.radius * np.sin(rad)])
            i = i + 1
        return lab
    def lab2deg(self, lab):
        n_sample = lab.shape[0]
        degree = np.zeros(n_sample)
        for i in range(n_sample):
            l, a, b = lab[i]
            temp_sin = (b - self.center_b) / self.radius
            temp_cos = (a - self.center_a) / self.radius
            loc = np.arctan2(temp_sin, temp_cos)
            loc = np.mod(loc, 2*np.pi)
            degree[i] = loc
        degree = degree / (2 * np.pi) * 360
        return degree
    def deg2rgba(self, degree):
        lab = self.deg2lab(degree)
        return lab2rgba
    def lms2deg(self, lms):
        '''
        lms (n * 3 matrix): where n is the number of colors, 3 means the three channels of l, m, s
        '''
        xyz = lms2xyz(lms)
        lab = xyz2lab(xyz)
        degree = self.lab2deg(lab)
        return degree

class Color_cell():
    '''
    convert degree to the firing rate of color cell (doesn't exist)
    '''
    def __init__(self, n_unit, sigma = None):
        # n_unit (int): the number of color cell which are uniformly distributed from 0 to 360
        self.n_unit = n_unit
        if sigma is None:
            self.sigma = 360 / n_unit / 2 # default sigma
        else:
            self.sigma = sigma
        self.yi = np.arange(0, 360, 360 / n_unit)

    def fire(self, degree):
        '''
        input:
        degree (np.array [float](n)): n input degree
        output:
        ri (np.array [float](n, m)): m is the number of color cell
        '''
        n_input = degree.shape[0] # the number of degrees
        yi_tile = np.tile(self.yi, (n_input, 1)) # expand yi by repeating itself (as column) n times, where n is the number of degrees
        degree_tile = np.tile(degree.reshape(-1, 1), (1, self.n_unit)) # expand yi by repeating itself (as column) n times, where n is the number of degrees
        return self.ri_func(yi_tile, degree_tile)

    def ri_func(self, y, x):
        kappa = 1 / np.deg2rad(self.sigma * 2)**2
        r =  vonmises.pdf(np.deg2rad(y), kappa, loc=np.deg2rad(x))
        return r

    def decode(self, y):
        '''
        decode the degree from the firing of color cell
        input:
          y (np.array [float](n_color, self.n_unit): input firing rate of color cells. n_color means the number of color for decoding.
        output:
          deg (np.array [float](n_color)): color
        '''
        n_color = y.shape[0]
        deg = np.zeros(n_color)
        pref = self.yi / 360. * 2 * np.pi
        for i in range(n_color):
            temp_cos = np.sum(y[i, :]*np.cos(pref))
            temp_sin = np.sum(y[i, :]*np.sin(pref))
            loc = np.arctan2(temp_sin, temp_cos)
            deg[i] = np.mod(loc, 2*np.pi) / 2.0 / np.pi * 360.0
        return  deg # convert to degree

class Color_triangular():
    def fire(self, degree):
        '''
        input:
        degree (np.array [float](n)): n input degree
        output:
        ri (np.array [float](n, 2))
        '''
        n_color = degree.shape[0]
        ri = np.zeros((n_color, 2))
        ri[:, 0] = np.cos(np.deg2rad(degree))
        ri[:, 1] = np.sin(np.deg2rad(degree))
        return ri

    def decode(self, y):
        '''
        decode the degree from the firing of color cell
        input:
          y (np.array [float](n_color, 2): input firing rate. The first column is cos and the second one is sin
        output:
          deg (np.array [float](n_color)): color
        '''
        n_color = y.shape[0]
        deg = np.zeros(n_color)

        loc = np.arctan2(y[:, 1], y[:, 0])
        deg = np.mod(loc, 2*np.pi) / 2.0 / np.pi * 360.0
        return deg

def lab2rgba(lab):
    '''
    lab (n * 3 matrix): where n is the number of colors, 3 means the three channels of l, a, b
    '''
    n_sample = lab.shape[0]
    lab_instance = colorob.LabColor(0, 0, 0, observer='2', illuminant='d65')
    RGBA = np.zeros((n_sample, 4)) # 4 channels R, G, B, A
    for i in range(n_sample):
        lab_instance.lab_l, lab_instance.lab_a, lab_instance.lab_b = lab[i, :]
        sRGB_sample = colorcv.convert_color(lab_instance, colorob.sRGBColor)
        RGBA[i, :3] = sRGB_sample.clamped_rgb_r, sRGB_sample.clamped_rgb_g, sRGB_sample.clamped_rgb_b
        RGBA[i, 3] = 1
    return RGBA

def lab2rgb(lab):
    '''
    lab (n * 3 matrix): where n is the number of colors, 3 means the three channels of l, a, b
    '''
    n_sample = lab.shape[0]
    lab_instance = colorob.LabColor(0, 0, 0, observer='2', illuminant='d65')
    RGB = np.zeros((n_sample, 3)) # 4 channels R, G, B, A
    for i in range(n_sample):
        lab_instance.lab_l, lab_instance.lab_a, lab_instance.lab_b = lab[i, :]
        sRGB_sample = colorcv.convert_color(lab_instance, colorob.AdobeRGBColor)
        RGB[i] = sRGB_sample.clamped_rgb_r, sRGB_sample.clamped_rgb_g, sRGB_sample.clamped_rgb_b
    return RGB

def lab2xyz(lab):
    '''
    input:
      lab (n * 3 matrix): where n is the number of colors, 3 means the three channels of l, a, b
    return:
      xyz (n * 3 matrix): where n is the number of colors, 3 means the three channels of x, y, z
    '''
    n_sample = lab.shape[0]
    lab_instance = colorob.LabColor(0, 0, 0, observer='2', illuminant='d65')
    xyz = np.zeros((n_sample, 3)) # 4 channels R, G, B, A
    for i in range(n_sample):
        lab_instance.lab_l, lab_instance.lab_a, lab_instance.lab_b = lab[i, :]
        xyz_sample = colorcv.convert_color(lab_instance, colorob.XYZColor)
        xyz[i, :] = xyz_sample.get_value_tuple()
    return xyz

def xyz2lms(xyz):
    '''
    xyz (n * 3 matrix): where n is the number of colors, 3 means the three channels of x, y, z
    '''
    lms = np.dot(MAT_LMS_NORM, xyz.T)
    # reshape to 2-D image
    lms = lms.T.reshape(xyz.shape)
    return lms

def lms2xyz(lms):
    '''
    lms (n * 3 matrix): where n is the number of colors, 3 means the three channels of l, m, s
    '''
    xyz = np.dot(MAT_LMS_NORM_INV, lms.T)
    xyz = xyz.T.reshape(lms.shape)
    return xyz

def xyz2lab(xyz):
    '''
    input:
      xyz (n * 3 matrix): where n is the number of colors, 3 means the three channels of x, y, z
    return:
      lab (n * 3 matrix): where n is the number of colors, 3 means the three channels of l, a, b
    '''
    xyz = xyz.reshape((-1, 3))
    n_sample = xyz.shape[0]
    xyz_instance = colorob.XYZColor(0, 0, 0, observer='2', illuminant='d65')
    lab = np.zeros((n_sample, 3))
    for i in range(n_sample):
        xyz_instance.xyz_x, xyz_instance.xyz_y, xyz_instance.xyz_z = xyz[i, :]
        lab_sample = colorcv.convert_color(xyz_instance, colorob.LabColor)
        lab[i, :] = lab_sample.get_value_tuple()
    return lab
