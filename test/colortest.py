# test the colormanager
import context
import matplotlib.pyplot as plt
import numpy as np
import colormath.color_objects as colorob
import colormath.color_conversions as colorcv
from core.color_manager import Degree_color

# normalized to D65
MAT_LMS_NORM = np.array([[0.4002, -0.2263, 0], [0.7076, 1.1653, 0], [-0.0808, 0, 0.9182]])

########## lab comparision ##########
#### tranditional method
lab_l, lab_a, lab_b = 60, 22, 14
n_sample = 360 # 360 sample colors
radius_color = 52
lab = np.zeros((n_sample, 3)) # 3 channels l, a, b
i = 0
for angle in np.linspace(0, 2 * np.pi, n_sample):
    lab[i, :] = np.array([lab_l, lab_a + radius_color * np.cos(angle), lab_b + radius_color * np.sin(angle)])
    i = i + 1
##### new method
color = Degree_color()
degrees = np.linspace(0, 360, n_sample)
lab_degree = color.out_color(degrees)
#### compare the result of these two methods
if (lab - lab_degree < 10e-5).all():
    print('LAB pass')

########## RGBA comparision ##########
##### Tranditional method
RGBA = np.zeros((n_sample, 4)) # 3 channels R, G, B, A
cielib_sample = colorob.LabColor(lab_l, lab_a, lab_b, observer='2', illuminant='d65')
for i in range(n_sample):
    cielib_sample.lab_l, cielib_sample.lab_a, cielib_sample.lab_b = lab[i, :]
    sRGB_sample = colorcv.convert_color(cielib_sample, colorob.sRGBColor)
    RGBA[i, :3] = sRGB_sample.clamped_rgb_r, sRGB_sample.clamped_rgb_g, sRGB_sample.clamped_rgb_b
    RGBA[i, 3] = 1
##### new method
RGBA_degree = color.out_color(degrees, fmat='RGBA')
if (RGBA - RGBA_degree < 10e-5).all():
    print('RGBA pass')
########## Show LMS ##########
lab_degree = color.out_color(np.array([0]))
lms_degree = color.out_color(np.array([0]), fmat='LMS')
print('lab_degree is: ', lab_degree)
print('lms_degree is: ', lms_degree)

######### Show the color map
from matplotlib.colors import ListedColormap
from matplotlib import cm
import matplotlib as mpl

n_sample = 360
color = Degree_color()
RGBA = color.out_color(n_sample, fmat='RGBA')
ring_map = ListedColormap(RGBA)

fig = plt.figure()

display_axes = fig.add_axes([0.1,0.1,0.8,0.8], projection='polar')
display_axes._direction = 2*np.pi ## This is a nasty hack - using the hidden field to 
                                  ## multiply the values such that 1 become 2*pi
                                  ## this field is supposed to take values 1 or -1 only!!

norm = mpl.colors.Normalize(0.0, 2*np.pi)

# Plot the colorbar onto the polar axis
# note - use orientation horizontal so that the gradient goes around
# the wheel rather than centre out

quant_steps = 360
cb = mpl.colorbar.ColorbarBase(display_axes, cmap=ring_map,
                                   norm=norm,
                                   orientation='horizontal')

# aesthetics - get rid of border and axis labels. The black line means 0 degree. Increasing degree by anticolockwise rotation
cb.outline.set_visible(False) # comment this line to show a black line indicate 0 degree
display_axes.set_axis_off()
display_axes.set_rlim([-1,0.3])

plt.show() # Replace with plt.savefig if you want to save a file

########## fill_between color
degs = np.arange(360).astype(int) # the index of RAGB is also a index
degs_samples = np.random.choice(degs, 10000)
n_bins = 180
hist, bin_edges = np.histogram(degs_samples, bins=n_bins, density=True) # distribution density of degs_samples. Totally len(bins) - 1 bins.

fig, ax = plt.subplots(1, 1)
for i in range(n_bins):
    ax.fill_between(bin_edges[i:i+2], hist[i:i+2], color=RGBA[2 * i])

########### Reproduce the paper's figure
#import pandas as pd
#
#exp_data = pd.read_excel('../data/exp_paper.xlsx', sheet_name='Figure1c')
#human_res = exp_data['human'][1:].to_numpy().astype(float)
#human_res[human_res >= 360] = human_res[human_res >= 360] % 360
#
#hist, bin_edges = np.histogram(human_res, bins=n_bins, density=True) # distribution density of degs_samples. Totally len(bins) - 1 bins.
#
#fig, ax = plt.subplots(1, 1)
#for i in range(n_bins):
#    ax.fill_between(bin_edges[i:i+2], hist[i:i+2], color=RGBA[2 * i])
#
#plt.show()

########## degree color test ##########
deg = np.arange(0, 360, 52)

cDeg = Degree_color()
lms_degree = color.out_color(deg, fmat='LMS')
deg_decode = cDeg.lms2deg(lms_degree)
print('########## Degree color ##########')
print('encoded deg: ', deg)
print('decoded deg: ', deg_decode)


########## color cell test ##########
from core.color_manager import Color_cell

n_unit = 30

ccell = Color_cell(n_unit)
fr = ccell.fire(deg)
deg_decode = ccell.decode(fr)
print('########## Color cell ##########')
print('encoded deg: ', deg)
print('decoded deg: ', deg_decode)
