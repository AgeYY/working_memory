from __future__ import division

import torch
import numpy as np
from matplotlib import pyplot as plt


from .. import dataset
from .. import task
from .. import default
from .. import train

import seaborn as sns

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from sklearn.decomposition import PCA
from matplotlib import cm
from core.color_error import Color_error

import os
from scipy import stats
from numba import jit

from .. import run

class behavior_ploter():
    def load_behavior(self, report_color, target_color):
        '''
        load the behavior data
        report_color, target_color: one dimension array.
        self:
          error: report_color - target_color
        '''
        self.report_color = report_color
        self.target_color = target_color

        # calculate the error
        color_error = Color_error()
        color_error.add_data(report_color, target_dire)
        self.error = color_error.calculate_error()
