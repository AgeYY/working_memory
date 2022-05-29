import context
import matplotlib.pyplot as plt
from core.color_manager import Color_cell
import numpy as np
from scipy.stats import vonmises

#centers = [-np.pi * (1 - 0.2), -np.pi * (1 - 0.5), np.pi * (1 - 0.4)]
centers = np.linspace(-np.pi, np.pi, 5, endpoint=False)
x = np.linspace(-np.pi, np.pi, 100)
x_deg = (x + np.pi) / 2 / np.pi * 360
kappa = 1.0 / 0.2

fig = plt.figure(figsize=(3, 3))
ax = fig.add_axes([0.23, 0.2, 0.6, 0.7])

for ci in centers:
    yi = vonmises.pdf(x, kappa, loc=ci)
    ax.plot(x_deg, yi)

ax.set_xlabel('Stimuli')
ax.set_xlabel('Stimuli (degree)')
ax.set_ylabel('Response (a.u.)')
ax.set_xticks([0, 360])
ax.set_yticks([0, 0.5, 1])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
