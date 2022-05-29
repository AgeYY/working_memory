# color error related calculation. Unit of color is degree
import numpy as np

class Color_error():
    def add_data(self, output_color, target_color):
        '''
        output_color, target_color (1D np.array [float])
        '''
        self.output_color, self.target_color = np.array(output_color).astype(float), np.array(target_color).astype(float)
    def calculate_error(self):
        '''
        output - target
        '''
        error = self.output_color - self.target_color
        error[error > 180.] = error[error > 180.] - 360
        error[error < -180.] = error[error < -180.] + 360
        return error

class Circular_operator():
    # the variable are bounded by c_min and c_max.
    def __init__(self, c_min=0, c_max=2 * np.pi):
        self.c_min = c_min; self.c_max = c_max
    def add(self, a, b):
        '''
        input:
        a, b  (array): element-wise operation
        '''
        ap = self.shift_zero(a); bp = self.shift_zero(b)
        c = np.mod(ap + bp, c_max)
        return self.shift_c_min(c)

    def diff(self, a, b):
        '''
        a - b, the difference is bounded by -(c_max - c_min) / 2 to (c_max - c_min) / 2
        '''
        ap = self.shift_zero(a); bp = self.shift_zero(b)
        c = ap - bp

        minus_lower = -(self.c_max - self.c_min) / 2
        minus_upper = -minus_lower

        c[c > minus_upper] = c[c > minus_upper] - (self.c_max - self.c_min)
        c[c <= minus_lower] = c[c <= minus_lower] + (self.c_max - self.c_min)

        return c

    def shift_zero(self, x):
        '''
        shift variable to 0, c_max - c_min
        '''
        return x - self.c_min

    def shift_c_min(self, x):
        return x + self.c_min
