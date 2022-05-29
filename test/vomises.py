from scipy.stats import vonmises
import numpy as np
import matplotlib.pyplot as plt

y = np.ones((2, 2))
x = np.random.randn(2, 2)

y = np.array([[1, 0], [0, 0]])
x = np.array([[0, 0], [0, 3.15]])

n_ring = 10
sigma = 2 * np.pi / n_ring / 2 * 2
kappa = 1 / sigma**2
x_list = np.linspace(-np.pi, np.pi, n_ring, endpoint=False)
y = np.linspace(-np.pi, np.pi, 100)

for x in x_list:
    #r = vonmises.pdf(y, kappa, loc=x)
    r = 1 / 2 * np.exp(kappa * ( np.cos(y - x)  - 1 ))
    plt.plot(y, r)

plt.scatter(sigma, 0)
plt.show()
