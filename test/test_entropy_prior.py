import context
from core.normalitier import Normalitier, Entropier
import numpy as np
import matplotlib.pyplot as plt

mu = np.array([40, 130, 220, 310])
mu = mu / 360 * 2 * np.pi - np.pi
nter = Normalitier(mu)

sigma = 3 / 360 * 2 * np.pi

#print(sigma)
#normality = nter.sigma2normality(sigma)
#print(normality)
#print(nter.normality2sigma(normality))

eper = Entropier(mu, dim=360)

print(sigma)
entropy = eper.sigma2entropy(sigma)
print(entropy)
print(eper.entropy2sigma(entropy))

sigma = np.linspace(3, 90, 50) / 360 * 2 * np.pi
entropy = eper.sigma_arr2entropy(sigma)
#entropy = np.empty(sigma)
#for idx, sig in enumerate(sigma):
#    entropy[i]

sigma = sigma/2.0 / np.pi * 360
plt.figure()
plt.plot(sigma, entropy)
plt.show()
