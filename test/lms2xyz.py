import numpy as np
from core.color_manager import Degree_color

deg_color = Degree_color()
deg = np.array([60, 65])
lms = deg_color.out_color(deg, fmat='LMS')
print(lms)
deg1 = deg_color.lms2deg(lms)
print(deg, deg1)

