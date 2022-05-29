import context
from core.color_input import vonmises_prior
import numpy as np
import matplotlib.pyplot as plt
from core.bay_drift_loss_torch import prior_func

deg_input = np.linspace(0, 360, 100)
bias_center = np.array([40, 130, 220, 310])
sigma = 10
center_prob = 0.7

pdf_input = vonmises_prior(deg_input, bias_center, width=sigma, center_prob=center_prob)
pdf_input = pdf_input / sum(pdf_input)

rad_input = deg_input / 360.0 * 2 * np.pi - np.pi
bias_center = bias_center / 360.0 * 2 * np.pi - np.pi
sigma = sigma / 360.0 * 2 * np.pi

pdf_bay = prior_func(rad_input, bias_center, sigma=sigma, center_prob=center_prob)
pdf_bay = pdf_bay / sum(pdf_bay)

plt.plot(deg_input, pdf_input)
plt.plot((rad_input + np.pi) * 360.0 / 2 / np.pi, pdf_bay)
plt.ylim([0, 1])
plt.show()
